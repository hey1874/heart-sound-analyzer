"""S1/S2 分段与心动周期重建

由 heartbeat.py 拆分而来。方法体逐字保留,行为与拆分前完全一致——
拆分只改代码组织,不改算法。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

class SegmentationMixin:
    """S1/S2 分段与心动周期重建"""

    def segment_s1s2(self, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把交替出现的心音峰分成 S1 和 S2。

        依据:正常/偏慢心率下,收缩期(S1→S2)比舒张期(S2→S1)短且更稳定。
        假设峰严格交替,比较两种相位假设,取"收缩间期更短"的那种。
        ⚠️ 漏检/多检会打乱交替,所以仅在 confidence 较高、节律规则时可靠。
        """
        if len(beats) < 4:
            return np.array([], dtype=int), np.array([], dtype=int)
        iv = np.diff(beats).astype(float)
        mean_a = iv[0::2].mean()                       # 假设 beats[0]=S1
        mean_b = iv[1::2].mean() if len(iv) > 1 else np.inf
        if mean_a <= mean_b:
            return beats[0::2], beats[1::2]
        return beats[1::2], beats[0::2]

    def _cycles(self, s1: np.ndarray, s2: np.ndarray) -> list[tuple[int, int, int]]:
        """重建心动周期 (S1_start, S2, S1_end),每周期恰含一个 S2。"""
        out = []
        for i in range(len(s1) - 1):
            a, b = int(s1[i]), int(s1[i + 1])
            mid = s2[(s2 > a) & (s2 < b)]
            if len(mid) == 1:
                out.append((a, int(mid[0]), b))
        return out

    def segment_hsmm(
        self, env: np.ndarray, bpm: float | None, fs_d: float = 50.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """时长约束的 4 状态分段(S1→收缩→S2→舒张),返回 (S1, S2) 起点索引。

        相比"短间期=收缩"启发式,这里用生理时长先验(由心率推出各相时长)
        约束一个分段 Viterbi,对漏检/噪声更稳健。它是 Springer 2016 HSMM 的
        轻量无监督版:发射用包络(高→心音/低→间歇),时长用高斯先验。
        在降采样到 fs_d(默认 50Hz,~20ms 分辨率)的包络上做 DP 以保证速度。
        """
        if bpm is None or len(env) < 20:
            return np.array([], dtype=int), np.array([], dtype=int)
        T = 60.0 / bpm
        # 实际抽取因子取整,并据此回算真实 fs_d,保证"降采样"与"反映射"一致
        d = max(1, int(round(self.fs / fs_d)))
        fs_d = self.fs / d
        e = signal.resample_poly(env, 1, d)
        e = (e - e.mean()) / (e.std() + 1e-12)
        M = len(e)
        if M < 8:                       # 降采样后过短,DP 无法凑齐心动周期
            return np.array([], dtype=int), np.array([], dtype=int)
        p_sound = 1.0 / (1.0 + np.exp(-2.0 * e))         # 越高越像心音
        cum_s = np.concatenate([[0.0], np.cumsum(np.log(p_sound + 1e-9))])
        cum_q = np.concatenate([[0.0], np.cumsum(np.log(1 - p_sound + 1e-9))])
        EMIS = {0: cum_s, 1: cum_q, 2: cum_s, 3: cum_q}  # 0,2=心音 1,3=间歇

        sys_int = min(0.40, max(0.20, 0.20 + 0.12 * T))  # S1→S2 间期(生理近似)
        durs = {                                          # (均值 s, 标准差 s)
            0: (0.11, 0.03),                              # S1
            1: (max(0.05, sys_int - 0.11), 0.04),         # 收缩静默
            2: (0.09, 0.03),                              # S2
            3: (max(0.06, T - sys_int - 0.09), 0.09),     # 舒张静默
        }
        prev = {0: 3, 1: 0, 2: 1, 3: 2}                  # 强制循环转移
        NEG = -1e18

        f = np.full((M + 1, 4), NEG)
        bp_i = np.full((M + 1, 4), -1, dtype=int)        # 该段起点
        for j in range(1, M + 1):
            for st in range(4):
                mu, sg = durs[st]
                dmin = max(1, int(round((mu - 3 * sg) * fs_d)))
                dmax = max(dmin, int(round((mu + 3 * sg) * fs_d)))
                cum = EMIS[st]
                hi = j - dmin
                lo = max(0, j - dmax)
                for i in range(lo, hi + 1):
                    base = 0.0 if i == 0 else f[i][prev[st]]  # i==0 可为起始段
                    if i != 0 and base <= NEG / 2:
                        continue
                    val = base + (cum[j] - cum[i]) \
                        - 0.5 * (((j - i) - mu * fs_d) / (sg * fs_d)) ** 2
                    if val > f[j][st]:
                        f[j][st] = val
                        bp_i[j][st] = i
        end_st = int(np.argmax(f[M]))
        if f[M][end_st] <= NEG / 2:
            return np.array([], dtype=int), np.array([], dtype=int)
        segs, j, st = [], M, end_st
        while j > 0 and st >= 0:
            i = bp_i[j][st]
            if i < 0:
                break
            segs.append((st, i))
            st = prev[st] if i != 0 else -1
            j = i
        segs.reverse()
        s1 = [int(round(i * d)) for s, i in segs if s == 0]
        s2 = [int(round(i * d)) for s, i in segs if s == 2]
        return np.array(s1, dtype=int), np.array(s2, dtype=int)
