"""
cnn.py — 频谱图 + CNN 的杂音检测器。

为什么要做
----------
工程特征 + 树模型的基线在 CirCor 上只有 AUC 0.715(默认阈值敏感度 0.340,
漏掉 2/3 的杂音),见 docs/VALIDATION.md。实测还表明**瓶颈不是分段**:给它
官方标注当作完美的心动周期参考,AUC 反而降到 0.685。瓶颈是「高频能量比」
这个手工特征本身太粗糙——它把整个收缩期压成一个标量,丢掉了杂音的频谱形状
与时频结构,而这恰恰是听诊学用来区分杂音的东西。

所以正确的下一步是换表示:用**对数梅尔频谱图**保留时频结构,让 CNN 自己学。

设计取舍(数据量只有 ~3000 条录音 / ~900 名患者,极易过拟合)
--------------------------------------------------------------
- **小模型**:4 个卷积块、约 20 万参数。不用 ResNet/预训练大模型。
- **窗级训练 + 录音级聚合**:每条录音切成若干重叠窗,窗级预测后取分位数
  聚合。既扩增了样本量,也符合"杂音只在部分心动周期明显"的实际。
- **按患者分组**:同一患者的多个听诊区必须整体划入同一折,否则泄漏。
- **强增广**:时间掩蔽 / 频率掩蔽 / 随机增益 / 随机时移。
- **class_weight**:阳性窗远少于阴性窗。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import numpy as np

# --- 特征:对数梅尔频谱图 ---------------------------------------------------

FS = 2000.0                 # 与 HeartSoundAnalyzer 一致的处理采样率
N_FFT = 256                 # 128ms 窗 @2kHz
HOP = 64                    # 32ms 跳,时间分辨率够刻画收缩期内的形状
N_MELS = 64
FMIN, FMAX = 20.0, 900.0    # 覆盖 S1/S2(20-200)到杂音高频(可达 900)
WIN_S = 4.0                 # 每个训练窗 4 秒(约 4-6 个心动周期)
HOP_S = 2.0                 # 窗间跳 2 秒(50% 重叠)


def _mel_filterbank(n_mels: int, n_fft: int, fs: float,
                    fmin: float, fmax: float) -> np.ndarray:
    """梅尔滤波器组。自己实现以避免引入 librosa 依赖。"""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    # 面积归一化,避免高频滤波器因带宽大而系统性偏亮
    area = fb.sum(axis=1, keepdims=True)
    return fb / np.maximum(area, 1e-9)


_FB = _mel_filterbank(N_MELS, N_FFT, FS, FMIN, FMAX)


def logmel(x: np.ndarray) -> np.ndarray:
    """波形(已重采样到 FS)-> 对数梅尔频谱图 (n_mels, frames)。"""
    from scipy import signal as sg
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < N_FFT:
        x = np.pad(x, (0, N_FFT - x.size))
    _, _, Z = sg.stft(x, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP,
                      window="hann", padded=False, boundary=None)
    p = np.abs(Z) ** 2
    m = _FB @ p
    return np.log(m + 1e-10).astype(np.float32)


def windows(x: np.ndarray, fs: float, analyzer=None) -> list[np.ndarray]:
    """把一条录音切成固定长度的重叠窗,返回各窗的对数梅尔谱。

    切窗前先做与传统管线**相同**的重采样和带通,保证两条路线吃到的是同一个
    前处理结果——否则性能差异会混入前处理差异。
    """
    from .analyzer import HeartSoundAnalyzer
    an = analyzer or HeartSoundAnalyzer()
    xr = an.resample(x, fs)
    if xr.size < 2:
        return []
    # 只做去均值 + 幅度归一化,不做窄带滤波:CNN 需要看到完整频谱
    xr = xr - xr.mean()
    peak = np.max(np.abs(xr))
    if peak > 0:
        xr = xr / peak
    n, hop = int(WIN_S * an.fs), int(HOP_S * an.fs)
    if xr.size < n:                      # 太短则补零成一个窗
        xr = np.pad(xr, (0, n - xr.size))
    return [logmel(xr[i:i + n]) for i in range(0, xr.size - n + 1, hop)]


# --- 模型 -------------------------------------------------------------------

def build_model(n_mels: int = N_MELS):
    """小型 CNN。参数量刻意压到 ~20 万,匹配 ~3000 条录音的数据规模。"""
    import torch.nn as nn

    def block(cin, cout, pool):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    return nn.Sequential(
        block(1, 16, (2, 2)),
        block(16, 32, (2, 2)),
        block(32, 64, (2, 2)),
        block(64, 64, (2, 2)),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(64, 1),
    )


def load_model(path: str, device: str = "cpu"):
    """加载 train_cnn.py --save 存下的模型,返回 (model, meta)。"""
    import torch
    d = torch.load(path, map_location=device, weights_only=False)
    m = build_model()
    m.load_state_dict(d["state_dict"])
    m.to(device).eval()
    return m, d


def score_recording(model, x, fs, device: str = "cpu", q: float = 0.8) -> float:
    """对一条录音打分:切窗 -> 逐窗预测 -> 高分位聚合。"""
    import torch
    ws = windows(x, fs)
    if not ws:
        return float("nan")
    T = min(w.shape[1] for w in ws)
    X = torch.from_numpy(np.stack([w[:, :T] for w in ws]).astype(np.float32))
    X = X.unsqueeze(1)
    X = (X - X.mean(dim=(2, 3), keepdim=True)) / (
        X.std(dim=(2, 3), keepdim=True) + 1e-5)
    with torch.no_grad():
        p = torch.sigmoid(model(X.to(device)).squeeze(1)).cpu().numpy()
    return float(np.quantile(p, q))


def augment(batch, rng):
    """SpecAugment 风格增广:频率掩蔽 + 时间掩蔽 + 随机增益。"""
    import torch
    b, _, nm, nt = batch.shape
    out = batch.clone()
    for i in range(b):
        if rng.random() < 0.5:                       # 频率掩蔽
            f = rng.integers(1, max(2, nm // 8))
            f0 = rng.integers(0, nm - f)
            out[i, :, f0:f0 + f, :] = 0
        if rng.random() < 0.5:                       # 时间掩蔽
            t = rng.integers(1, max(2, nt // 8))
            t0 = rng.integers(0, nt - t)
            out[i, :, :, t0:t0 + t] = 0
        if rng.random() < 0.5:                       # 随机增益(对数域即加常数)
            out[i] = out[i] + float(rng.normal(0, 0.5))
    return torch.nan_to_num(out)
