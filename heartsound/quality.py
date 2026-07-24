"""信号质量:削波、工频、SQI、可采信判定

由 heartbeat.py 拆分而来。方法体逐字保留,行为与拆分前完全一致——
拆分只改代码组织,不改算法。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

class QualityMixin:
    """信号质量:削波、工频、SQI、可采信判定"""

    @staticmethod
    def clip_fraction(x_raw: np.ndarray) -> float:
        """削波样本比例。**必须在重采样之前**对原始波形计算。

        重采样的抗混叠滤波会把削波平顶磨圆:同一段硬削波信号,以 2000Hz 直接
        输入时测得 clip=0.027(正确拒绝),以 44kHz 输入(即声卡的实际情况)
        再重采样后测得 clip=0.000,门控完全失效。
        """
        x = np.asarray(x_raw, dtype=np.float64).ravel()
        # 这是个**相对**度量:任何信号至少有 1 个样本等于最大值,所以比例天然
        # 不低于 1/N。N ≤ 50 时该下限就已经 ≥0.02(门限),未削波的短信号也会
        # 被判成削波。短于 200 样本的片段本来就短到无法分析(会先被心率门控
        # 拦下),直接返回 0 比给一个假阳性更安全。
        if x.size < 200:
            return 0.0
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        mx = float(np.max(np.abs(x)))
        if mx <= 0:
            return 0.0
        return float(np.mean(np.abs(x) >= 0.99 * mx))

    def hum_ratio(self, f: np.ndarray, p: np.ndarray) -> float:
        """50/60Hz 工频及其谐波贡献的**超出本底**能量,占带内总能量的比例。

        工频落在 20-200Hz 心音带内,所以 band_ratio 遇到工频反而会升高——
        实测 50Hz 干扰使 band_ratio 从 0.74 升到 1.00、SQI 仍判 ok,而心率
        已经完全错了。这里单独度量工频线谱,用于抵消 band_ratio 的这一反常。

        工频是**线谱**(1-3 个频点),心音是**瞬态**(谱峰宽约 16Hz),因此以
        谱线邻域的中位数为本底,只统计超出本底的部分,避免误伤心音本体。
        """
        if len(f) < 4:
            return 0.0
        in_band = (f >= self.band[0]) & (f <= self.band[1])
        total = float(p[in_band].sum()) + 1e-20
        df = float(f[1] - f[0])
        half = max(df, 1.5)                  # 每根谱线取 ±1.5Hz
        excess = 0.0
        for f0 in (50.0, 60.0):
            for k in range(1, int(self.band[1] // f0) + 1):
                fc = k * f0
                if not (self.band[0] <= fc <= self.band[1]):
                    continue
                line = np.abs(f - fc) <= half
                if not line.any():
                    continue
                # 本底:谱线两侧 3-10Hz 的中位数
                side = (np.abs(f - fc) > 3.0) & (np.abs(f - fc) <= 10.0) & in_band
                base = float(np.median(p[side])) if side.any() else 0.0
                excess += max(0.0, float(p[line].sum()) - base * int(line.sum()))
        return float(min(1.0, excess / total))

    def signal_quality(
        self, x: np.ndarray, conf: float, x_raw: np.ndarray | None = None
    ) -> dict:
        """估计信号质量指数 SQI(0..1),用于剔除坏段。

        综合四项启发式:
          periodicity 周期性(自相关置信度)—— 心音应规律重复;
          band_ratio  带内能量占比 —— 心音能量应集中在 20-200Hz,而非宽带噪声;
          hum         50/60Hz 工频线谱占比 —— 抵消工频对 band_ratio 的反常抬升;
          clip        削波/饱和比例 —— 过载会失真(在 x_raw 上算,见 clip_fraction)。

        x 为重采样后的处理波形;x_raw 为原始输入波形(缺省时退回用 x 算削波,
        此时削波检测在高采样率输入下不可靠)。
        """
        x = np.asarray(x, dtype=np.float64)
        clip = self.clip_fraction(x if x_raw is None else x_raw)
        f, p = signal.welch(x, fs=self.fs, nperseg=min(len(x), 2048))
        in_band = (f >= self.band[0]) & (f <= self.band[1])
        band_ratio = float(p[in_band].sum() / (p.sum() + 1e-12))
        hum = self.hum_ratio(f, p)
        periodicity = float(max(0.0, min(1.0, conf)))
        # 工频占掉的那部分带内能量不算"有效心音能量"
        band_clean = band_ratio * (1.0 - hum)
        sqi = float(0.5 * periodicity + 0.4 * band_clean + 0.1 * (1 - clip))
        return {
            "sqi": sqi, "periodicity": periodicity,
            "band_ratio": band_ratio, "band_clean": band_clean,
            "hum": hum, "clip": clip,
            "ok": bool(sqi >= self.sqi_thr and clip < 0.02 and hum < self.hum_thr),
        }

    def _reliability(self, res: dict) -> tuple[bool, str | None]:
        """汇总可采信性,并给出人能看懂的失败原因(用于提示如何改善采集)。

        ⚠️ `band_ratio` 只用来**解释**原因,不能单独作为拒绝判据:响亮的杂音
        本身就有大量能量落在 150-600Hz(20-200Hz 带外),会把 band_ratio 压到
        0.22 左右。若拿它硬性拒绝,最该被检出的病例反而先被挡在门外。
        它对质量的贡献已经通过 0.4 的权重体现在综合 SQI 里了。
        """
        sqi = res.get("sqi") or {}
        if res.get("bpm") is None:
            return False, "录音过短或无法估计心率"
        # 硬缺陷:与病理无关,只可能是采集问题
        if sqi.get("clip", 0.0) >= 0.02:
            return False, "削波/过载(调低输入增益)"
        if sqi.get("hum", 0.0) >= self.hum_thr:
            return False, "工频干扰(检查屏蔽/接地,或改用电池供电)"
        if res.get("confidence", 0.0) < self.conf_thr:
            return False, "心音不规律或信号弱(调整听诊器位置、压紧)"
        if not sqi.get("ok", False):
            why = ("环境噪声/摩擦音偏大" if sqi.get("band_ratio", 1.0) < 0.30
                   else "周期性差")
            return False, f"信号质量不足(SQI {sqi.get('sqi', 0):.2f},{why})"
        return True, None
