"""
analyzer.py — HeartSoundAnalyzer 主类:构造、一站式 analyze()、标定加载。

具体算法按职责拆到四个 mixin(dsp / quality / segmentation / features)。
拆分只改代码组织,方法体逐字保留,行为与拆分前完全一致。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import DSPMixin
from .features import FeatureMixin
from .quality import QualityMixin
from .segmentation import SegmentationMixin


class HeartSoundAnalyzer(DSPMixin, QualityMixin, SegmentationMixin, FeatureMixin):
    """心音分析核心算法。

    参数
    ----
    fs_proc:     内部处理采样率(Hz)。心音 <200Hz,1000Hz 足够且高效。
    band:        心音带通范围 (低, 高) Hz。
    bpm_range:   合理心率范围 (min, max),用于约束自相关搜索区间。
    env_smooth_ms: 包络平滑窗口(毫秒)。

    判定与门控阈值
    ------------------------------------------------------------------
    murmur_thr / extra_thr / split_narrow_ms / split_fixed_iqr_ms
                                            —— 决定"疑似 XX"的判定;
    sqi_thr / conf_thr / hum_thr            —— 决定信号是否可采信。

    ⚠️ 这些数字目前是**经验值,未经任何真实心音数据标定**。在标定之前,本类
    输出的判定文字只能当作提示,不能当作筛查结论。标定方法见
    docs/ROADMAP.md 阶段 4。作为构造参数暴露,是为了让使用者能按自己的设备
    与数据集重新标定,而不是改源码里的魔数。

    分段使用**独立的窄带**(seg_band,默认 30-160Hz / 8 阶)而非主带 20-200Hz,
    以挡住强杂音对 HSMM 发射模型的污染;取值经真实数据 + 合成用例双重扫参
    选定,见 __init__ 中的对照表。
    """

    def __init__(
        self,
        fs_proc: float = 2000.0,
        band: tuple[float, float] = (20.0, 200.0),
        seg_band: tuple[float, float] = (30.0, 160.0),
        seg_order: int = 8,
        murmur_band: tuple[float, float] = (150.0, 600.0),
        bpm_range: tuple[float, float] = (40.0, 180.0),
        env_smooth_ms: float = 30.0,
        murmur_thr: float = 0.15,
        extra_thr: float = 0.20,
        # 生理性分裂:呼气 <30ms,吸气末可达 50-60ms。取 45ms 作为"超出生理
        # 范围"的下限(留出呼气/吸气均值的余量),而非原来的 40ms——40ms 会把
        # 正常吸气相判成异常。
        split_narrow_ms: float = 45.0,
        # 判"固定"的逐周期 IQR 上限:低于此说明分裂几乎不随呼吸变化。
        split_fixed_iqr_ms: float = 8.0,
        sqi_thr: float = 0.35,
        conf_thr: float = 0.30,
        hum_thr: float = 0.80,
        # 输入电平下限(峰值 dBFS,按 ±1.0 满量程的浮点音频约定)。
        # -50 dBFS ≈ 满量程的 0.3%:低于此说明增益没开对,此时信号与采集
        # 噪声无法区分。见 _reliability 中的说明。
        level_thr_dbfs: float = -50.0,
    ) -> None:
        self.fs = float(fs_proc)
        self.band = band
        self.seg_band = seg_band
        self.murmur_band = murmur_band
        self.bpm_range = bpm_range
        self.murmur_thr = murmur_thr
        self.extra_thr = extra_thr
        self.split_narrow_ms = split_narrow_ms
        self.split_fixed_iqr_ms = split_fixed_iqr_ms
        self.sqi_thr = sqi_thr
        self.conf_thr = conf_thr
        # 工频硬拒绝门限。定为 0.80 而非更严的值:实测工频幅度 0.1~0.2 时
        # hum 指标已达 0.43~0.75,但心率估计仍完全正确(72.0/72.2,置信度
        # 0.85/0.58)——此时拒收是**过度拒绝**,会把好录音挡在门外、逼用户
        # 去用 --force,反而架空门控。心率真正开始出错(72→109)是在幅度
        # 0.3~0.5、hum 指标 0.87~0.95 处。中间地带交给综合 SQI 判(它已按
        # (1-hum) 折算带内占比)。
        self.hum_thr = hum_thr
        self.level_thr_dbfs = level_thr_dbfs
        # 带通滤波器用二阶节(SOS)形式,数值更稳定;阶数 4。
        # 低带:提取 S1/S2 心音用于包络/节律。
        self.sos = signal.butter(
            4, [band[0], band[1]], btype="band", fs=self.fs, output="sos"
        )
        # 分段专用带:30-160Hz / 8 阶,只留 S1/S2 主能量区并挡住杂音。
        #
        # 问题:分段包络原先取自主带 20-200Hz,4 阶巴特沃斯在 200Hz 滚降不够
        # 陡,**响亮的杂音漏进包络**,把 HSMM 的"心音/间歇"发射模型带偏,
        # 时相归属判反(舒张期杂音幅度 ≥1.6 时被报成收缩期杂音)。
        #
        # 取值是**扫参数选的**,不是拍脑袋。在 274 条 CirCor 真实录音(对官方
        # S1 标注)与合成强杂音两个指标上同时评估:
        #
        #   分段带            真实 F1   合成强杂音判反
        #   20-200 / 4阶      0.9278    1.6, 2.0, 2.5 全判反
        #   20-200 / 8阶      0.9310    1.6, 2.0, 2.5 全判反
        #   25-180 / 6阶      0.9240    2.5 判反
        #   25-150 / 6阶      0.9230    无
        #   30-160 / 8阶      0.9310    无            ← 采用
        #
        # 关键教训:第一版只把带宽收窄到 25-150/6 阶,合成用例过了,但真实
        # F1 从 0.9278 掉到 0.9230——**用合成场景的修复换来了真实数据的倒退**。
        # 真正起作用的是**滚降陡度**(阶数),不是转折频率。
        self.sos_seg = signal.butter(
            seg_order, list(seg_band), btype="band", fs=self.fs, output="sos"
        )
        # 高带:杂音(湍流噪声)主要落在这里,用于杂音能量指数。
        hi = min(murmur_band[1], self.fs / 2 * 0.95)
        self.sos_murmur = signal.butter(
            4, [murmur_band[0], hi], btype="band", fs=self.fs, output="sos"
        )
        # 低带:S3/S4 奔马律(舒张期低频额外音)主要在 15-60Hz。
        self.sos_low = signal.butter(
            4, [15.0, 60.0], btype="band", fs=self.fs, output="sos"
        )
        self.env_win = max(1, int(env_smooth_ms / 1000.0 * self.fs))

    @classmethod
    def from_json(cls, path: str, operating_point: str = "youden",
                  **overrides) -> "HeartSoundAnalyzer":
        """从 calibrate.py 产出的 thresholds.json 构造已标定的分析器。

        operating_point 选 ROC 上的工作点:
          "default"  沿用代码里的经验阈值(只做对照用);
          "youden"   敏感度+特异度之和最大(平衡);
          "sens80" / "sens90"  保证该敏感度下特异度最优——筛查通常选这类,
                     因为漏诊比误报贵,但要接受特异度大幅下降。
        """
        import json
        with open(path, encoding="utf-8") as fh:
            cal = json.load(fh)
        kw = {}
        pt = (cal.get("murmur", {}).get("points", {}) or {}).get(operating_point)
        if pt:
            kw["murmur_thr"] = float(pt["thr"])
        kw.update(overrides)
        an = cls(**kw)
        an.calibration = cal
        an.operating_point = operating_point
        return an

    def analyze(self, x: np.ndarray, fs_in: float, screen: bool = True) -> dict:
        """对一段波形做完整分析,返回结果字典。

        screen=True 时额外做信号质量、节律与杂音的筛查级分析(仅提示,非诊断)。

        `res["reliable"]` 汇总"这段结果能不能采信":心率估出来了、置信度够、
        且 SQI 通过。所有下游(实时输出、predict.py)都应先看这一项——此前
        实时路径做了置信度门控而 predict.py 完全不做,两条路径的严谨程度不
        一致,导致低质量录音也会得到一个自信的判定。
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        # 空/极短输入直接短路:下游多处要取 max/median,在空数组上会抛
        # ValueError,而调用方(尤其 predict.py)期望拿到一个结构完整的结果。
        if x.size < 2:
            return self._empty_result("输入为空或过短")
        # 非有限值(NaN/inf)会经 FFT 卷积污染整段结果,且只发出一条
        # RuntimeWarning 就继续算下去,最终得到全 NaN 的"结果"。损坏的音频
        # 文件、某些解码器都可能产生。少量则置零后继续,大量则判定为坏数据。
        finite = np.isfinite(x)
        if not finite.all():
            n_bad = int((~finite).sum())
            if n_bad > 0.01 * x.size:
                return self._empty_result(
                    f"输入含 {n_bad}/{x.size} 个非有限值(NaN/inf),超过 1%")
            x = np.where(finite, x, 0.0)
        xr = self.resample(x, fs_in)
        if xr.size < 2:
            return self._empty_result("重采样后过短")
        xb = self.bandpass(xr)
        env = self.shannon_envelope(xb)
        bpm, conf = self.estimate_bpm(env)
        beats = self.detect_beats(env)
        res = {
            "bpm": bpm,                 # 估计心率,可能为 None
            "confidence": conf,         # 0..1,建议 >0.3 才采信
            "n_beats": int(len(beats)), # 窗口内检测到的心音数
            "beats": beats,             # 峰索引(基于 fs_proc)
            "env": env,                 # 主带包络(心率/可视化用)
            "env_seg": None,            # 分段专用窄带包络(见 sos_seg)
            "filtered": xb,             # 带通后波形(可视化用)
            "resampled": xr,            # 重采样后的原始波形(频谱图用:需完整频段)
            "fs_proc": self.fs,
            "sqi": None, "rhythm": None, "sti": None, "murmur": None,
            "extra_sounds": None, "s2_split": None, "hrv_freq": None,
            "spectrum": None,
            "reliable": False, "unreliable_reason": None,
        }
        if screen:
            # 削波必须在重采样前的原始波形上判定,见 clip_fraction
            res["sqi"] = self.signal_quality(xr, conf, x_raw=x)
            # 分段用**更窄的低带**包络,避免强杂音漏进来带偏 HSMM(见 sos_seg)
            env_seg = self.shannon_envelope(self._filtfilt(self.sos_seg, xr))
            res["env_seg"] = env_seg      # 暴露出来,便于外部复现分段结果
            s1, s2 = self.segment_hsmm(env_seg, bpm)
            if len(s1) < 3:
                s1, s2 = self.segment_s1s2(self.detect_beats(env_seg))
            cyc = self._cycles(s1, s2)
            res["rhythm"] = self.analyze_rhythm(s1)
            res["sti"] = self.systolic_intervals(cyc)
            res["hrv_freq"] = self.hrv_frequency(s1)
            x_high = self._filtfilt(self.sos_murmur, xr)
            mu = self.murmur_index(x_high, xb, cyc)
            # 形状/时相**无条件**计算。此前只在 systolic>阈值 时才算,导致
            # occupancy/peak_pos 在正常样本上 100% 缺失、异常样本上仅 15% 缺失
            # ——"是否缺失"本身成了标签的代理,树模型可零成本作弊(实测仅凭
            # 这一个缺失指示位即可达到 92.5% 准确率)。
            mu.update(self.murmur_shape(x_high, cyc))
            res["murmur"] = mu
            # 频谱轮廓:比 murmur_index 的单标量信息量大得多(见 spectral_profile)
            res["spectrum"] = self.spectral_profile(xr, cyc)
            x_low = self._filtfilt(self.sos_low, xr)
            res["extra_sounds"] = self.detect_extra_sounds(x_low, cyc)
            res["s2_split"] = self.detect_s2_split(xb, s2)
            res["reliable"], res["unreliable_reason"] = self._reliability(res)
        return res

    def _empty_result(self, reason: str) -> dict:
        """结构与 analyze() 一致的"数据不足"结果,便于调用方统一处理。"""
        return {
            "bpm": None, "confidence": 0.0, "n_beats": 0,
            "beats": np.array([], dtype=int),
            "env": np.array([]), "env_seg": np.array([]),
            "filtered": np.array([]), "resampled": np.array([]),
            "fs_proc": self.fs,
            "sqi": None, "rhythm": None, "sti": None, "murmur": None,
            "extra_sounds": None, "s2_split": None, "hrv_freq": None,
            "spectrum": None,
            "reliable": False, "unreliable_reason": reason,
        }
