"""
heartbeat.py — 通过(接听诊器的)麦克风实时采集心音并估计心率(BPM)。

设计要点
--------
1. 核心算法 `HeartSoundAnalyzer` 与音频采集(`RealtimeHeartRate`)完全解耦:
   前者只吃一段 numpy 波形 + 采样率,便于离线单测、复用到文件分析。
2. 听诊器贴在麦克风上时,心音(S1/S2)与胸壁耦合良好,主要能量在 20-200 Hz。

处理管线
--------
  重采样 -> 带通滤波 -> 香农能量包络 -> 自相关估心率 -> 峰值检测(每拍事件)

依赖:
  pip install numpy scipy sounddevice

⚠️ 仅用于学习/演示,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import threading

import numpy as np
from scipy import signal

from synth import synthesize

# 向后兼容:旧版本以私有名 `_synthesize` 从本模块导入合成器,现已移到 synth.py。
_synthesize = synthesize


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


# ---------------------------------------------------------------------------
# 核心算法:与采集无关,只处理一段波形
# ---------------------------------------------------------------------------
class HeartSoundAnalyzer:
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

    # --- 各步骤(单独暴露,方便调试/可视化) ---------------------------------

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

    # --- 筛查级分析:节律 & 杂音 --------------------------------------------

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

    def analyze_rhythm(self, s1: np.ndarray) -> dict:
        """从 S1 间期(RR)评估节律规则性。

        输出心率变异系数 CV、Poincaré 的 SD1/SD2、早搏计数与文字判读。
        """
        if len(s1) < 4:
            return {"classification": "数据不足", "n_rr": 0}
        rr = np.diff(s1) / self.fs * 1000.0            # 毫秒
        mean_rr = float(rr.mean())
        cv = float(rr.std() / (mean_rr + 1e-12))
        d = np.diff(rr)                                 # 相邻 RR 之差
        sd1 = float(np.std(d) / np.sqrt(2))             # Poincaré 短期变异
        sd2 = float(np.sqrt(max(0.0, 2 * rr.var() - 0.5 * np.var(d))))  # 长期
        # 标准时域 HRV(Task Force 1996)
        sdnn = float(rr.std())                          # RR 总体标准差
        rmssd = float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0
        pnn50 = float(np.mean(np.abs(d) > 50.0) * 100) if len(d) else 0.0
        # 早搏:相对局部中位数偏离 >20% 的间期
        med = np.median(rr)
        ectopic = int(np.sum(np.abs(rr - med) > 0.20 * med))

        if cv < 0.05:
            cls = "规则"
        elif cv < 0.12:
            cls = "轻度不齐(可能为窦性心律不齐,常见且多属正常)"
        else:
            cls = "明显不规则(建议就医排查早搏/房颤)"
        return {
            "classification": cls, "n_rr": int(len(rr)),
            "mean_bpm": 60000.0 / (mean_rr + 1e-12), "cv": cv,
            "sd1_ms": sd1, "sd2_ms": sd2,
            "sdnn_ms": sdnn, "rmssd_ms": rmssd, "pnn50_pct": pnn50,
            "ectopic": ectopic,
        }

    def murmur_index(
        self, x_high: np.ndarray, x_low: np.ndarray,
        cycles: list[tuple[int, int, int]],
    ) -> dict:
        """杂音能量指数:收缩/舒张期"安静段"的高频能量 / S1 心音响度。

        正常时收缩、舒张期应基本安静,指数接近 0;出现湍流杂音时升高。
        分母用低频带 S1 的实际响度(而非高频带),即"杂音相对心音的响度",
        接近临床杂音分级的直觉,也避免归一化分母虚小。
        ⚠️ 纯启发式,阈值需按设备标定,假阳/假阴都不少,只作提示。
        """
        if not cycles:
            return {"systolic": None, "diastolic": None,
                    "asymmetry": None, "noise_suspect": None, "flag": "数据不足"}
        e = x_high ** 2                                  # 高频带瞬时能量
        e_lo = x_low ** 2                                # 低频带(心音本体)能量
        g = max(1, int(0.04 * self.fs))                # 避开心音本体的护带
        s1_ref, sys_e, dia_e = [], [], []
        for a, s2, b in cycles:
            s1_ref.append(e_lo[max(0, a - g): a + g].mean())  # S1 响度(低带)
            if s2 - g > a + g:
                sys_e.append(e[a + g: s2 - g].mean())   # 收缩期安静段(高带)
            if b - g > s2 + g:
                dia_e.append(e[s2 + g: b - g].mean())   # 舒张期安静段(高带)
        ref = np.median(s1_ref) + 1e-12
        sys_idx = float(np.median(sys_e) / ref) if sys_e else None
        dia_idx = float(np.median(dia_e) / ref) if dia_e else None

        # 时相不对称度:环境宽带噪声均匀铺满整个心动周期,收缩期与舒张期的
        # 高频能量近似相等;真杂音则集中在某一时相。实测 20 段仅加白噪声
        # (noise=0.25)的正常信号,20 段全部同时报"收缩期+舒张期杂音"——
        # 这个组合是噪声的指纹,而非病理。以两者的相对差作为判别量。
        asym, noise_suspect = None, None
        if sys_idx is not None and dia_idx is not None:
            asym = float(abs(sys_idx - dia_idx) / (sys_idx + dia_idx + 1e-12))
            noise_suspect = bool(
                max(sys_idx, dia_idx) > self.murmur_thr and asym < 0.20
            )

        flags = []
        if sys_idx is not None and sys_idx > self.murmur_thr:
            flags.append("疑似收缩期杂音")
        if dia_idx is not None and dia_idx > self.murmur_thr:
            flags.append("疑似舒张期杂音")
        if noise_suspect:
            # 不静默抑制,也不过度断言。双期能量接近有三种可能,本模块**分不清**:
            #   1. 宽带噪声污染(最常见);
            #   2. 分段不准——杂音响过心音时 HSMM 会被带偏,收缩/舒张窗口错位;
            #   3. 真的双期升高(连续性杂音如 PDA、狭窄合并反流)。
            # 因此只陈述观察到的现象和排查方向,把两个数值都留在结果里。
            flags = ["收缩期与舒张期高频能量接近(不对称度低):"
                     "可能为宽带噪声污染、分段不准或双期杂音,建议改善采集后复测"]
        return {
            "systolic": sys_idx, "diastolic": dia_idx,
            "asymmetry": asym, "noise_suspect": noise_suspect,
            "flag": "、".join(flags) if flags else "未见明显杂音",
        }

    def systolic_intervals(self, cycles: list[tuple[int, int, int]]) -> dict:
        """收缩时间间期(STI):收缩期/舒张期时长与比值。

        S1→S2 起点间期 ≈ 机械收缩期(含左室射血时间 LVET 的近似),临床上与
        心肌收缩力相关(Weissler)。这里给出可解释的时长量,非精确 LVET。
        """
        if not cycles:
            return {"systole_ms": None, "diastole_ms": None}
        f = self.fs / 1000.0
        sys_ms = np.array([(s2 - a) / f for a, s2, b in cycles])
        dia_ms = np.array([(b - s2) / f for a, s2, b in cycles])
        return {
            "systole_ms": float(np.median(sys_ms)),    # ≈LVET 近似
            "diastole_ms": float(np.median(dia_ms)),
            "sys_dia_ratio": float(np.median(sys_ms) / (np.median(dia_ms) + 1e-9)),
            "n_cycles": int(len(cycles)),
        }

    def murmur_shape(
        self, x_high: np.ndarray, cycles: list[tuple[int, int, int]], n: int = 30
    ) -> dict:
        """杂音的形状与时相分类(收缩期)。

        把每个周期的收缩期高频能量包络重采样到固定长度并平均,据其形状判:
          递增-递减(钻石型)→ 喷射性,见于主动脉/肺动脉瓣狭窄;
          全收缩期平台型     → 见于二尖瓣反流 / 室间隔缺损;
          递增 / 递减型      → 见于部分反流 / 二尖瓣脱垂等。
        时相:按能量峰位置与占空比判 早/中/晚/全收缩期。
        ⚠️ 启发式提示,非诊断;形状-病变对应仅为听诊学常识,需医师判读。
        """
        e = x_high ** 2
        g = max(1, int(0.04 * self.fs))
        segs = []
        for a, s2, b in cycles:
            lo, hi = a + g, s2 - g                      # 收缩期安静段
            if hi - lo < n:
                continue
            seg = e[lo:hi]
            seg = signal.resample(seg, n)
            seg = np.clip(seg, 0, None)
            mx = seg.max() + 1e-12
            segs.append(seg / mx)
        if not segs:
            return {"shape": "数据不足", "timing": None}
        env = np.mean(segs, axis=0)
        env = env / (env.max() + 1e-12)
        t1, t2, t3 = env[:n // 3].mean(), env[n // 3:2 * n // 3].mean(), \
            env[2 * n // 3:].mean()
        peak_pos = float(np.argmax(env) / n)            # 0(早)..1(晚)
        occupancy = float(np.mean(env > 0.5))           # 高能量占收缩期比例
        # 平坦度:中间 80% 的变异系数。原判据是 `env.max()-env.min() < 0.5`,
        # 但 env 已归一化(max 恒为 1),而收缩期两端天然衰减使 min 接近 0,
        # 该条件恒为假——"全收缩期平台型"这一分支从未触发过(实测平台型杂音
        # occupancy 已达 0.90,仍被判为"不定形")。
        core = env[max(1, n // 10): n - max(1, n // 10)] if n >= 10 else env
        flatness = float(core.std() / (core.mean() + 1e-12))

        if occupancy > 0.7 and flatness < 0.25:
            shape = "全收缩期平台型(疑二尖瓣反流/VSD类)"
        elif t2 > t1 * 1.15 and t2 > t3 * 1.15:
            shape = "递增-递减/钻石型(疑主动脉/肺动脉瓣狭窄类)"
        elif t3 > t1 * 1.3:
            shape = "递增型(crescendo)"
        elif t1 > t3 * 1.3:
            shape = "递减型(decrescendo)"
        else:
            shape = "不定形"
        timing = ("全收缩期" if occupancy > 0.7 else
                  "早收缩期" if peak_pos < 0.33 else
                  "中收缩期" if peak_pos < 0.66 else "晚收缩期")
        return {"shape": shape, "timing": timing, "peak_pos": peak_pos,
                "occupancy": occupancy, "flatness": flatness}

    def detect_extra_sounds(
        self, x_low: np.ndarray, cycles: list[tuple[int, int, int]]
    ) -> dict:
        """检测舒张期低频额外音 S3 / S4(奔马律)。

        S3(室性奔马律):早舒张期低频音,见于心衰/容量负荷;
        S4(房性奔马律):晚舒张期(收缩前)低频音,见于心室顺应性下降。
        在低频带(15-60Hz)能量上,取舒张期早 1/3 与末 1/3 的峰,
        归一化到 S1 低频响度。
        ⚠️ 启发式提示,非诊断。
        """
        if not cycles:
            return {"s3": None, "s4": None, "flag": "数据不足"}
        e = x_low ** 2
        g = max(1, int(0.04 * self.fs))
        s1_ref, s3_vals, s4_vals = [], [], []
        for a, s2, b in cycles:
            s1_ref.append(e[max(0, a - g): a + g].max())
            lo, hi = s2 + g, b - g
            if hi - lo < int(0.12 * self.fs):          # 舒张期太短(快心率)跳过
                continue
            dia = e[lo:hi]
            L = len(dia)
            s3_vals.append(dia[:L // 3].max())          # 早舒张 → S3
            s4_vals.append(dia[2 * L // 3:].max())      # 晚舒张(收缩前)→ S4
        ref = np.median(s1_ref) + 1e-12
        s3 = float(np.median(s3_vals) / ref) if s3_vals else None
        s4 = float(np.median(s4_vals) / ref) if s4_vals else None
        flags = []
        if s3 is not None and s3 > self.extra_thr:
            flags.append("疑似S3(室性奔马律,见于心衰)")
        if s4 is not None and s4 > self.extra_thr:
            flags.append("疑似S4(房性奔马律)")
        return {"s3": s3, "s4": s4,
                "flag": "、".join(flags) if flags else "未见明显额外音"}

    def detect_s2_split(self, x_band: np.ndarray, s2_idx: np.ndarray) -> dict:
        """估计 S2 分裂(A2–P2 间期,ms)。

        S2 含主动脉瓣(A2)与肺动脉瓣(P2)两个分量,吸气时生理性增宽。
        宽而固定(不随呼吸变化)的分裂提示房间隔缺损(ASD)。
        在每个 S2 窗内对细包络(Hilbert)找两个亚峰,取间隔中位数。

        ⚠️ 此前版本用「高度 > 0.4·峰值 + 最小间隔 15ms」选峰,测到的其实是
        希尔伯特包络的**纹波**而非 A2-P2:S2 主频约 70Hz,振荡周期 14ms,与
        15ms 的最小间隔基本重合,导致未注入任何分裂的正常信号也稳定报出
        18-21ms 的"分裂"(13%-27% 的周期检出双峰)。现改为:
          - 平滑窗 5ms → 10ms(零点落在 100Hz,压制 70Hz 附近纹波);
          - 最小间隔 15ms → 20ms(仍远小于 40ms 的宽分裂判据);
          - 增加**突出度**(prominence)判据——纹波峰骑在主瓣上,突出度很低,
            这是区分真双峰与纹波的关键;
          - 取最突出的两个峰而非最靠前的两个。
        """
        if len(s2_idx) == 0:
            return {"split_ms": None, "split_frac": None, "flag": "数据不足"}
        w = int(0.11 * self.fs)                         # S2 窗 ~110ms
        sm = max(1, int(0.010 * self.fs))               # ~10ms 平滑
        mind = max(1, int(0.020 * self.fs))             # 两峰最小间隔 ~20ms
        intervals, double, usable = [], 0, 0
        for s2 in s2_idx:
            a = int(s2)
            bb = min(len(x_band), a + w)
            if bb - a < int(0.05 * self.fs):
                continue
            usable += 1
            env = _movavg(np.abs(signal.hilbert(x_band[a:bb])), sm)
            span = float(env.max() - env.min())
            if span <= 0:
                continue
            pk, props = signal.find_peaks(
                env, height=0.4 * env.max(), distance=mind,
                prominence=0.20 * span,
            )
            if len(pk) >= 2:
                top = pk[np.argsort(props["prominences"])[-2:]]
                p0, p1 = sorted(int(v) for v in top)
                double += 1
                intervals.append((p1 - p0) / self.fs * 1000.0)
        if not usable:
            return {"split_ms": None, "split_frac": None,
                    "split_iqr_ms": None, "flag": "数据不足"}
        if not intervals:
            return {"split_ms": 0.0, "split_frac": 0.0, "split_iqr_ms": 0.0,
                    "flag": "S2 单一(未见明显分裂)"}
        split = float(np.median(intervals))
        frac = double / usable
        # 逐周期离散度。呼吸会周期性调制 A2-P2:吸气时右室回心血量增加、
        # 肺动脉瓣关闭延后,分裂增宽。所以**正常人的分裂在多个呼吸周期上应有
        # 明显起伏**,而 ASD 的分裂"固定"——这才是教科书上的鉴别点。
        iqr = float(np.subtract(*np.percentile(intervals, [75, 25]))) \
            if len(intervals) >= 4 else None
        # 记录时长要够覆盖数个呼吸周期(静息 12-20 次/分,一次 3-5 秒)
        span_s = len(x_band) / self.fs
        return {"split_ms": split, "split_frac": float(frac),
                "split_iqr_ms": iqr, "n_split": len(intervals),
                "flag": self._split_flag(split, frac, iqr, span_s)}

    def _split_flag(self, split: float, frac: float,
                    iqr: float | None, span_s: float) -> str:
        """把分裂间期与其呼吸性变异翻译成人话。

        文献依据:生理性分裂呼气时 <30ms,**吸气末可达 50-60ms**;宽分裂约
        40-50ms。因此单看"宽度 >40ms"会把正常人的吸气相当成异常——本方法原先
        就是这么做的。真正的鉴别点是**固定性**(不随呼吸变化),而固定性无法
        从单一宽度值判断。

        本模块没有呼吸相标注,只能用逐周期离散度(IQR)作为呼吸性变异的**代理**:
        分裂宽、且跨多个呼吸周期几乎不变 → 才提示需要排查 ASD。
        """
        if split <= self.split_narrow_ms:
            return f"S2 分裂(~{split:.0f}ms,在生理范围内)"
        if frac <= 0.5:
            return f"S2 分裂(~{split:.0f}ms,仅 {frac:.0%} 的周期检出,证据弱)"
        if span_s < 15.0:
            return (f"S2 分裂 ~{split:.0f}ms(偏宽);录音仅 {span_s:.0f}s,"
                    "不足以覆盖数个呼吸周期,无法判断是否固定")
        if iqr is None:
            return f"S2 分裂 ~{split:.0f}ms(偏宽);周期数不足,无法评估变异性"
        if iqr < self.split_fixed_iqr_ms:
            return (f"S2 宽分裂 ~{split:.0f}ms 且逐周期变异小"
                    f"(IQR {iqr:.0f}ms)——符合「固定分裂」形态,"
                    "建议就医排查房间隔缺损")
        return (f"S2 分裂 ~{split:.0f}ms(偏宽)但随周期变化"
                f"(IQR {iqr:.0f}ms),更像生理性呼吸调制")

    def hrv_frequency(
        self, s1: np.ndarray, min_rr: int = 100, min_secs: float = 120.0
    ) -> dict:
        """频域 HRV:LF/HF 比(自主神经平衡)。

        将 RR 间期序列重采样为均匀 4Hz 心动周期图后做 Welch 谱,
        积分 LF(0.04–0.15Hz)与 HF(0.15–0.40Hz)。

        ⚠️ 门限与文档一致性:此前 `min_rr=20`,在 72BPM 下只需 **18 秒**录音就
        判"可用",而 LF 下界 0.04Hz 意味着单段至少要 25 秒才谈得上分辨——与
        docstring 声称的"≥1–2 分钟"矛盾。现按 Task Force 1996 的短时程标准,
        同时要求 ≥100 个心动周期**且** ≥120 秒。

        异位搏动(早搏)会在 RR 序列上造成尖峰,严重污染频谱,按惯例先用局部
        中位数替换后再做谱分析,并报告修正了几个。
        """
        if len(s1) < 2:
            return {"available": False, "note": "心动周期不足"}
        rr = np.diff(s1) / self.fs                       # 秒
        dur = float(rr.sum())
        if len(rr) < min_rr or dur < min_secs:
            return {"available": False,
                    "note": (f"需 ≥{min_rr} 个心动周期且 ≥{min_secs:.0f} 秒"
                             f"(当前 {len(rr)} 个 / {dur:.0f} 秒)")}
        med = float(np.median(rr))
        ectopic = np.abs(rr - med) > 0.20 * med
        rr = np.where(ectopic, med, rr)                  # 异位搏动替换为中位数
        t = np.cumsum(rr)
        fs_t = 4.0
        ti = np.arange(0, t[-1], 1.0 / fs_t)
        rri = np.interp(ti, t, rr)
        rri = rri - rri.mean()
        f, p = signal.welch(rri, fs=fs_t, nperseg=min(len(rri), 256))
        lf = float(p[(f >= 0.04) & (f < 0.15)].sum())
        hf = float(p[(f >= 0.15) & (f < 0.40)].sum())
        return {"available": True, "lf": lf, "hf": hf,
                "lf_hf": float(lf / (hf + 1e-12)),
                "n_rr": int(len(rr)), "duration_s": dur,
                "ectopic_corrected": int(ectopic.sum())}

    # --- 信号质量 SQI:分析前的门控(临床数据处理的标准步骤) ----------------

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

    # --- HSMM-lite 心音分段(时长约束的分段 Viterbi) ------------------------

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

    # --- 一站式入口 ---------------------------------------------------------

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
            "filtered": xb,             # 滤波后波形(可视化用)
            "fs_proc": self.fs,
            "sqi": None, "rhythm": None, "sti": None, "murmur": None,
            "extra_sounds": None, "s2_split": None, "hrv_freq": None,
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
            "filtered": np.array([]),
            "fs_proc": self.fs,
            "sqi": None, "rhythm": None, "sti": None, "murmur": None,
            "extra_sounds": None, "s2_split": None, "hrv_freq": None,
            "reliable": False, "unreliable_reason": reason,
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


# ---------------------------------------------------------------------------
# 实时采集:环形缓冲 + 周期性分析
# ---------------------------------------------------------------------------
class RealtimeHeartRate:
    """从麦克风实时读流,每隔 hop 秒对最近 window 秒做一次分析。"""

    def __init__(
        self,
        device: int | str | None = None,
        samplerate: int | None = None,
        window_s: float = 6.0,     # 分析窗:越长越稳,越短越跟手
        hop_s: float = 1.0,        # 每隔多久输出一次
        conf_gate: float = 0.30,   # 置信度门限,低于此判"信号不清"
    ) -> None:
        self.device = device
        self.samplerate = samplerate
        self.window_s = window_s
        self.hop_s = hop_s
        self.conf_gate = conf_gate
        # 门控逻辑统一由 analyzer 持有,避免实时路径与 predict.py 各写一套
        self.analyzer = HeartSoundAnalyzer(conf_thr=conf_gate)

    def run(self) -> None:
        import sounddevice as sd  # 延迟导入,核心算法本身不依赖它

        info = sd.query_devices(self.device, "input")
        fs = int(self.samplerate or info["default_samplerate"])
        win_n = int(self.window_s * fs)
        # 索引式环形缓冲:写指针推进,只拷贝新到的那一块。
        # 此前用 np.roll 整体搬移,每次回调都要新分配并拷贝整个窗口
        # (44.1kHz/6s 实测 313µs,占 512 样本回调预算的 2.7%),
        # 而且在音频回调线程里做分配本身就是实时音频的忌讳。
        ring = np.zeros(win_n, dtype=np.float32)
        widx = 0                       # 下一个写入位置
        filled = 0
        since_hop = 0
        xruns = 0                      # 回调内不做 I/O,只累计,由主线程报告
        hop_n = int(self.hop_s * fs)
        lock = threading.Lock()        # 音频回调线程与主线程共享缓冲,需互斥

        def callback(indata, frames, time_info, status):
            nonlocal widx, filled, since_hop, xruns
            if status:
                xruns += 1
            block = indata[:, 0]       # 取单声道
            if len(block) >= win_n:    # 块比窗口还长时只保留最新的一窗
                block = block[-win_n:]
            n = len(block)
            with lock:
                end = widx + n
                if end <= win_n:
                    ring[widx:end] = block
                else:                  # 跨越缓冲末端,分两段写
                    k = win_n - widx
                    ring[widx:] = block[:k]
                    ring[:end - win_n] = block[k:]
                widx = end % win_n
                filled = min(win_n, filled + n)
                since_hop += n

        print(f"🎙  设备={info['name']}  fs={fs}Hz  窗={self.window_s}s")
        print("把听诊器贴稳,保持安静。Ctrl+C 退出。\n")

        with sd.InputStream(
            device=self.device, channels=1, samplerate=fs,
            dtype="float32", callback=callback,
        ):
            try:
                reported_xruns = 0
                while True:
                    sd.sleep(int(self.hop_s * 1000))
                    with lock:         # 在锁内取快照并复位计数,锁外再做分析
                        ready = since_hop >= hop_n and filled >= win_n
                        if ready:
                            # 按时间顺序展开环形缓冲(分配发生在主线程)
                            snapshot = np.concatenate((ring[widx:], ring[:widx]))
                            since_hop = 0
                        n_xrun = xruns
                    if n_xrun > reported_xruns:
                        print(f"⚠️  音频溢出/欠载 ×{n_xrun - reported_xruns}",
                              flush=True)
                        reported_xruns = n_xrun
                    if not ready:
                        continue
                    self._report(self.analyzer.analyze(snapshot, fs))
            except KeyboardInterrupt:
                print("\n已停止。")

    def _report(self, res: dict) -> None:
        sqi = res.get("sqi") or {}
        if not res.get("reliable"):
            reason = res.get("unreliable_reason") or "信号不清"
            print(f"[{'·' * 20}] 信号不清({reason},SQI {sqi.get('sqi', 0):.2f})",
                  flush=True)
            return
        bpm, conf = res["bpm"], res["confidence"]
        n = int(round(conf * 20))
        parts = [f"[{'█' * n + '·' * (20 - n)}] {bpm:5.1f} BPM",
                 f"置信度 {conf:4.2f}", f"窗内心音 {res['n_beats']}"]

        rh = res.get("rhythm") or {}
        if rh.get("n_rr", 0) >= 3:
            s = f"节律:{rh['classification']}"
            if rh.get("ectopic"):
                s += f"(早搏{rh['ectopic']})"
            parts.append(s)

        mu = res.get("murmur") or {}
        if mu.get("flag") not in (None, "数据不足", "未见明显杂音"):
            s = f"杂音:{mu['flag']}"
            # 形状/时相现在无条件计算(避免特征缺失泄漏),但只在**收缩期**窗口
            # 上有意义,故仅在确实疑似收缩期杂音、且不像噪声时才展示。
            if ("收缩期杂音" in mu["flag"] and not mu.get("noise_suspect")
                    and mu.get("shape") not in (None, "数据不足", "不定形")):
                s += f"({mu['shape']}/{mu['timing']})"
            parts.append(s)

        ex = res.get("extra_sounds") or {}
        if ex.get("flag") not in (None, "数据不足", "未见明显额外音"):
            parts.append(ex["flag"])

        sp = res.get("s2_split") or {}
        if (sp.get("split_ms") or 0) > self.analyzer.split_narrow_ms:
            parts.append(sp["flag"])

        print("   ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Windows 控制台默认 GBK,统一切到 UTF-8 避免 emoji/中文报错
    try:   # Windows 控制台默认 GBK;stderr 也要一起切,否则报错信息乱码
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if "--selftest" in sys.argv:
        # 自测已移到 selftest.py:原来的内联版本无论成败都返回退出码 0,
        # 无法接入 CI;现在按失败数返回非零退出码,并被 tests/ 复用。
        from selftest import main
        sys.exit(main())
    else:
        RealtimeHeartRate().run()
