"""滤波、包络、心率估计等与临床无关的信号处理

由 heartbeat.py 拆分而来。方法体逐字保留,行为与拆分前完全一致——
拆分只改代码组织,不改算法。
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _movavg(v: np.ndarray, n: int) -> np.ndarray:
    """滑动平均,边缘用端点值延拓。

    直接 `np.convolve(..., mode="same")` 会在首尾按零补齐,使包络两端被人为
    压低——在短窗(如 110ms 的 S2 窗)上这会制造出虚假的峰谷。
    """
    if n <= 1 or v.size < 2:
        return v
    n = min(int(n), v.size)
    pad = n // 2
    vp = np.pad(v, pad, mode="edge")
    return np.convolve(vp, np.ones(n) / n, mode="same")[pad: pad + v.size]

class DSPMixin:
    """滤波、包络、心率估计等与临床无关的信号处理"""

    def resample(self, x: np.ndarray, fs_in: float) -> np.ndarray:
        """把输入波形重采样到内部处理采样率。"""
        x = np.asarray(x, dtype=np.float64).ravel()
        if abs(fs_in - self.fs) < 1e-6:
            return x
        # resample_poly 内部自动用 gcd 约分,44100->1000 即 10/441。
        return signal.resample_poly(x, int(self.fs), int(round(fs_in)))

    @staticmethod
    def _filtfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
        """零相位滤波;信号短于 padlen 时退回去均值。

        统一三条滤波支路的短信号容错。此前 `bandpass` 用 try/except,而
        `analyze` 里的高/低频带直接调 sosfiltfilt、仅用 `len(xr) > 50` 保护,
        这个数字与 SOS 的实际 padlen 无关。
        """
        try:
            return signal.sosfiltfilt(sos, x)
        except ValueError:
            return x - np.mean(x)

    def bandpass(self, x: np.ndarray) -> np.ndarray:
        """零相位带通,保留 S1/S2 能量,去除工频/呼吸/摩擦。"""
        return self._filtfilt(self.sos, x)

    def shannon_envelope(self, x: np.ndarray) -> np.ndarray:
        """归一化平均香农能量包络(Liang 1997)。

        E = -s·log(s),s 为归一化后的瞬时能量。相比简单整流,香农能量
        能突出中等强度的心音、压制低幅噪声和个别尖峰。
        """
        peak = np.max(np.abs(x)) + 1e-12
        xn = x / peak
        s = xn ** 2
        E = -s * np.log(s + 1e-12)
        E = _movavg(E, self.env_win)
        # 标准化:零均值单位方差,便于设阈值/自相关
        E = (E - E.mean()) / (E.std() + 1e-12)
        return E

    def estimate_bpm(self, env: np.ndarray) -> tuple[float | None, float]:
        """用包络自相关估计心率。

        返回 (bpm, confidence)。confidence 是该 lag 处的归一化自相关系数
        (~0..1),信号越规律越接近 1;过低时应判定为"不可靠"。

        选峰不能只取搜索区内的最大值。心音包络每个周期含 S1、S2 两个脉冲,
        自相关在**舒张间期**(S2→下一 S1)处也有峰;该 lag 落在 40-180BPM
        搜索区内(72BPM 时约 533ms → 112BPM),在干扰下会盖过真周期。
        真周期的判据是它在 2×lag 处同样有峰,而舒张间期没有——因此对每个
        候选峰做一次谐波确认后再比较。
        """
        env = env - env.mean()
        ac = signal.correlate(env, env, mode="full")
        ac = ac[len(ac) // 2:]              # 取非负 lag
        ac = ac / (ac[0] + 1e-12)           # 归一化,ac[0]=1

        lag_min = int(self.fs * 60.0 / self.bpm_range[1])  # 高 BPM -> 小 lag
        lag_max = int(self.fs * 60.0 / self.bpm_range[0])  # 低 BPM -> 大 lag
        lag_max = min(lag_max, len(ac) - 1)
        if lag_max <= lag_min:
            return None, 0.0

        seg = ac[lag_min:lag_max + 1]
        peak_rel = int(np.argmax(seg))
        cand, _ = signal.find_peaks(seg, height=0.5 * seg[peak_rel])
        cand = sorted(set(cand.tolist()) | {peak_rel})

        def _score(rel: int) -> float:
            """候选 lag 的谐波确认得分:自身自相关 + 2×lag 处的确认峰。"""
            lag = rel + lag_min
            s = float(seg[rel])
            j = 2 * lag
            if j < len(ac):                 # 允许 ±5% 周期抖动
                w = max(1, int(0.05 * lag))
                s += 0.5 * float(ac[max(0, j - w): min(len(ac), j + w + 1)].max())
            return s

        rel = max(cand, key=_score)
        lag = rel + lag_min
        conf = float(ac[lag])

        # 抛物线插值提高 lag 分辨率(亚采样精度)
        if 0 < rel < len(seg) - 1:
            y0, y1, y2 = seg[rel - 1], seg[rel], seg[rel + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-12:
                lag = lag + 0.5 * (y0 - y2) / denom

        bpm = 60.0 * self.fs / lag
        return bpm, conf

    def detect_beats(
        self, env: np.ndarray, k: float = 0.5, refractory_ms: float = 200.0
    ) -> np.ndarray:
        """在包络上检测心音事件(S1/S2 峰),返回样本索引。

        阈值 = mean + k·std;refractory 为最小峰间距,避免重复计数。
        (注意:这里检测的是单个心音,一个心动周期通常含 S1+S2 两个峰;
        心率应以 estimate_bpm 的周期为准。)
        """
        thr = env.mean() + k * env.std()
        min_dist = max(1, int(refractory_ms / 1000.0 * self.fs))
        peaks, _ = signal.find_peaks(env, height=thr, distance=min_dist)
        return peaks
