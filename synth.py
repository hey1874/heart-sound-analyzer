"""
synth.py — 合成心音信号生成器,用于离线自测与 ML 管线验证。

从 `heartbeat.py` 中抽出:它是测试/演示脚手架而非核心算法,原先以私有名
`_synthesize` 被三个模块跨文件导入。这里转为公开 API `synthesize()`。

⚠️ 合成信号只能验证"算法方向是否正确",不能替代真实心音上的验证。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = ["synthesize"]


def synthesize(
    bpm: float = 72.0,
    secs: float = 8.0,
    fs: float = 2000.0,
    noise: float = 0.05,
    seed: int = 0,
    irregular: float = 0.0,
    murmur: float = 0.0,
    murmur_shape: str = "plateau",
    murmur_phase: str = "systolic",
    s3: float = 0.0,
    s2_split_ms: float = 0.0,
    hum_hz: float = 0.0,
    hum: float = 0.0,
) -> tuple[np.ndarray, float]:
    """合成一段含 S1/S2 的心音。返回 (波形, 采样率)。

    参数
    ----
    irregular:    每拍周期随机抖动的相对幅度(模拟节律不齐)。
    murmur:       杂音幅度;>0 时在指定时相注入带限(200-600Hz)噪声。
    murmur_shape: plateau(平台) / diamond(钻石) / crescendo(递增) / decrescendo(递减)。
    murmur_phase: systolic(收缩期) / diastolic(舒张期)。
    s3:           早舒张期 S3 奔马律幅度。
    s2_split_ms:  S2 第二分量(A2-P2 分裂)间隔,毫秒。
    hum_hz/hum:   工频干扰频率与幅度(如 hum_hz=50, hum=0.5),用于验证 SQI 门控。
    """
    rng = np.random.default_rng(seed)
    n = int(secs * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    hp = signal.butter(4, [200, min(600, fs / 2 * 0.95)], "band", fs=fs, output="sos")
    # 杂音噪声源只生成一次:原实现放在逐拍循环内,每拍都要滤一遍整段噪声,
    # 复杂度 O(拍数 × 总长度) 且无必要。
    burst = signal.sosfilt(hp, rng.standard_normal(n)) if murmur > 0 else None

    base = 60.0 / bpm
    t1 = 0.0
    while t1 < secs:
        t2 = t1 + 0.30                       # S2 "dub",收缩期约 0.3s
        for tc, amp, f in ((t1, 1.0, 50.0), (t2, 0.6, 70.0)):
            idx = (t > tc) & (t < tc + 0.10)
            tau = t[idx] - tc
            x[idx] += amp * np.sin(2 * np.pi * f * tau) * np.exp(-tau / 0.02)
        if s2_split_ms > 0:                  # S2 第二分量(分裂)
            tc = t2 + s2_split_ms / 1000.0
            idx = (t > tc) & (t < tc + 0.10)
            tau = t[idx] - tc
            x[idx] += 0.5 * np.sin(2 * np.pi * 70 * tau) * np.exp(-tau / 0.02)
        if s3 > 0:                           # 早舒张期 S3 低频音(~35Hz)
            tc = t2 + 0.14
            idx = (t > tc) & (t < tc + 0.08)
            tau = t[idx] - tc
            x[idx] += s3 * np.sin(2 * np.pi * 35 * tau) * np.exp(-tau / 0.03)
        period = base * (1.0 + irregular * rng.standard_normal())
        period = max(0.3, period)
        if murmur > 0:                       # 湍流噪声,按形状加权
            if murmur_phase == "diastolic":
                lo, hi = t2 + 0.12, t1 + period - 0.03
            else:
                lo, hi = t1 + 0.04, t2 - 0.03
            mask = (t > lo) & (t < hi)
            if hi > lo and mask.any():
                p = (t[mask] - lo) / (hi - lo)       # 相位内相对位置 0..1
                if murmur_shape == "crescendo":
                    w = p
                elif murmur_shape == "decrescendo":
                    w = 1 - p
                elif murmur_shape == "diamond":
                    w = 1 - np.abs(2 * p - 1)        # 中间高两端低
                else:                                # plateau
                    w = np.ones_like(p)
                x[mask] += murmur * w * burst[mask]
        t1 += period

    if hum > 0 and hum_hz > 0:               # 工频干扰(含 3 次谐波)
        x += hum * np.sin(2 * np.pi * hum_hz * t)
        x += 0.3 * hum * np.sin(2 * np.pi * 3 * hum_hz * t)
    x += noise * rng.standard_normal(n)
    return x, fs
