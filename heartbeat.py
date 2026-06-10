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
    """

    def __init__(
        self,
        fs_proc: float = 2000.0,
        band: tuple[float, float] = (20.0, 200.0),
        murmur_band: tuple[float, float] = (150.0, 600.0),
        bpm_range: tuple[float, float] = (40.0, 180.0),
        env_smooth_ms: float = 30.0,
    ) -> None:
        self.fs = float(fs_proc)
        self.band = band
        self.murmur_band = murmur_band
        self.bpm_range = bpm_range
        # 带通滤波器用二阶节(SOS)形式,数值更稳定;阶数 4。
        # 低带:提取 S1/S2 心音用于包络/节律。
        self.sos = signal.butter(
            4, [band[0], band[1]], btype="band", fs=self.fs, output="sos"
        )
        # 高带:杂音(湍流噪声)主要落在这里,用于杂音能量指数。
        hi = min(murmur_band[1], self.fs / 2 * 0.95)
        self.sos_murmur = signal.butter(
            4, [murmur_band[0], hi], btype="band", fs=self.fs, output="sos"
        )
        self.env_win = max(1, int(env_smooth_ms / 1000.0 * self.fs))

    # --- 各步骤(单独暴露,方便调试/可视化) ---------------------------------

    def resample(self, x: np.ndarray, fs_in: float) -> np.ndarray:
        """把输入波形重采样到内部处理采样率。"""
        x = np.asarray(x, dtype=np.float64).ravel()
        if abs(fs_in - self.fs) < 1e-6:
            return x
        # resample_poly 内部自动用 gcd 约分,44100->1000 即 10/441。
        return signal.resample_poly(x, int(self.fs), int(round(fs_in)))

    def bandpass(self, x: np.ndarray) -> np.ndarray:
        """零相位带通,保留 S1/S2 能量,去除工频/呼吸/摩擦。"""
        # sosfiltfilt 需 len(x) > padlen(≈3·(2·n_sections+1));过短则仅去均值。
        try:
            return signal.sosfiltfilt(self.sos, x)
        except ValueError:
            return x - np.mean(x)

    def shannon_envelope(self, x: np.ndarray) -> np.ndarray:
        """归一化平均香农能量包络(Liang 1997)。

        E = -s·log(s),s 为归一化后的瞬时能量。相比简单整流,香农能量
        能突出中等强度的心音、压制低幅噪声和个别尖峰。
        """
        peak = np.max(np.abs(x)) + 1e-12
        xn = x / peak
        s = xn ** 2
        E = -s * np.log(s + 1e-12)
        # 滑动平均平滑
        kernel = np.ones(self.env_win) / self.env_win
        E = np.convolve(E, kernel, mode="same")
        # 标准化:零均值单位方差,便于设阈值/自相关
        E = (E - E.mean()) / (E.std() + 1e-12)
        return E

    def estimate_bpm(self, env: np.ndarray) -> tuple[float | None, float]:
        """用包络自相关估计心率。

        返回 (bpm, confidence)。confidence 是该 lag 处的归一化自相关系数
        (~0..1),信号越规律越接近 1;过低时应判定为"不可靠"。
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
        rel = int(np.argmax(seg))
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
        d = np.diff(rr)
        sd1 = float(np.std(d) / np.sqrt(2))
        sd2 = float(np.sqrt(max(0.0, 2 * rr.var() - 0.5 * np.var(d))))
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
            "sd1_ms": sd1, "sd2_ms": sd2, "ectopic": ectopic,
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
            return {"systolic": None, "diastolic": None, "flag": "数据不足"}
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

        flags = []
        if sys_idx is not None and sys_idx > 0.15:
            flags.append("疑似收缩期杂音")
        if dia_idx is not None and dia_idx > 0.15:
            flags.append("疑似舒张期杂音")
        return {
            "systolic": sys_idx, "diastolic": dia_idx,
            "flag": "、".join(flags) if flags else "未见明显杂音",
        }

    # --- 信号质量 SQI:分析前的门控(临床数据处理的标准步骤) ----------------

    def signal_quality(self, x: np.ndarray, conf: float) -> dict:
        """估计信号质量指数 SQI(0..1),用于剔除坏段。

        综合三项启发式:
          periodicity 周期性(自相关置信度)—— 心音应规律重复;
          band_ratio  带内能量占比 —— 心音能量应集中在 20-200Hz,而非宽带噪声;
          clip        削波/饱和比例 —— 过载会失真。
        ⚠️ band_ratio 无法区分 50/60Hz 工频(也落在带内),必要时另加陷波。
        """
        x = np.asarray(x, dtype=np.float64)
        mx = np.max(np.abs(x)) + 1e-12
        clip = float(np.mean(np.abs(x) >= 0.99 * mx))
        f, p = signal.welch(x, fs=self.fs, nperseg=min(len(x), 2048))
        in_band = (f >= self.band[0]) & (f <= self.band[1])
        band_ratio = float(p[in_band].sum() / (p.sum() + 1e-12))
        periodicity = float(max(0.0, min(1.0, conf)))
        sqi = float(0.5 * periodicity + 0.4 * band_ratio + 0.1 * (1 - clip))
        return {
            "sqi": sqi, "periodicity": periodicity,
            "band_ratio": band_ratio, "clip": clip,
            "ok": bool(sqi >= 0.35 and clip < 0.02),
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
        """
        xr = self.resample(x, fs_in)
        xb = self.bandpass(xr)
        env = self.shannon_envelope(xb)
        bpm, conf = self.estimate_bpm(env)
        beats = self.detect_beats(env)
        res = {
            "bpm": bpm,                 # 估计心率,可能为 None
            "confidence": conf,         # 0..1,建议 >0.3 才采信
            "n_beats": int(len(beats)), # 窗口内检测到的心音数
            "beats": beats,             # 峰索引(基于 fs_proc)
            "env": env,                 # 包络(可视化用)
            "filtered": xb,             # 滤波后波形(可视化用)
            "fs_proc": self.fs,
            "sqi": None, "rhythm": None, "murmur": None,
        }
        if screen:
            res["sqi"] = self.signal_quality(xr, conf)
            # 优先用 HSMM 分段;失败(段数过少)回退到启发式
            s1, s2 = self.segment_hsmm(env, bpm)
            if len(s1) < 3:
                s1, s2 = self.segment_s1s2(beats)
            res["rhythm"] = self.analyze_rhythm(s1)
            x_high = signal.sosfiltfilt(self.sos_murmur, xr) \
                if len(xr) > 50 else xr
            res["murmur"] = self.murmur_index(x_high, xb, self._cycles(s1, s2))
        return res


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
        self.analyzer = HeartSoundAnalyzer()

    def run(self) -> None:
        import sounddevice as sd  # 延迟导入,核心算法本身不依赖它

        info = sd.query_devices(self.device, "input")
        fs = int(self.samplerate or info["default_samplerate"])
        win_n = int(self.window_s * fs)
        ring = np.zeros(win_n, dtype=np.float32)
        filled = 0
        since_hop = 0
        hop_n = int(self.hop_s * fs)
        lock = threading.Lock()        # 音频回调线程与主线程共享缓冲,需互斥

        def callback(indata, frames, time_info, status):
            nonlocal ring, filled, since_hop
            if status:
                print("⚠️", status, flush=True)
            block = indata[:, 0]  # 取单声道
            n = len(block)
            with lock:
                ring = np.roll(ring, -n)
                ring[-n:] = block
                filled = min(win_n, filled + n)
                since_hop += n

        print(f"🎙  设备={info['name']}  fs={fs}Hz  窗={self.window_s}s")
        print("把听诊器贴稳,保持安静。Ctrl+C 退出。\n")

        with sd.InputStream(
            device=self.device, channels=1, samplerate=fs,
            dtype="float32", callback=callback,
        ):
            try:
                while True:
                    sd.sleep(int(self.hop_s * 1000))
                    with lock:         # 在锁内取快照并复位计数,锁外再做分析
                        ready = since_hop >= hop_n and filled >= win_n
                        if ready:
                            snapshot = ring.copy()
                            since_hop = 0
                    if not ready:
                        continue
                    self._report(self.analyzer.analyze(snapshot, fs))
            except KeyboardInterrupt:
                print("\n已停止。")

    def _report(self, res: dict) -> None:
        bpm, conf = res["bpm"], res["confidence"]
        sqi = res.get("sqi") or {}
        if bpm is None or conf < self.conf_gate or sqi.get("ok") is False:
            bar = "·" * 20
            why = "削波/过载" if sqi.get("clip", 0) >= 0.02 else \
                  ("噪声大" if sqi.get("band_ratio", 1) < 0.3 else "不规律")
            print(f"[{bar}] 信号不清 ({why}, SQI {sqi.get('sqi', 0):.2f}) "
                  f"— 调整听诊器位置/保持安静", flush=True)
            return
        filled = int(round(conf * 20))
        bar = "█" * filled + "·" * (20 - filled)
        line = (f"[{bar}] {bpm:5.1f} BPM   置信度 {conf:4.2f}   "
                f"窗内心音 {res['n_beats']}")
        rh, mu = res.get("rhythm"), res.get("murmur")
        if rh and rh.get("n_rr", 0) >= 3:
            line += f"   节律:{rh['classification']}"
            if rh.get("ectopic"):
                line += f"(早搏{rh['ectopic']})"
        if mu and mu.get("flag") not in (None, "数据不足"):
            line += f"   杂音:{mu['flag']}"
        print(line, flush=True)


# ---------------------------------------------------------------------------
# 自测:无麦克风时用合成心音验证算法正确性
# ---------------------------------------------------------------------------
def _synthesize(bpm=72.0, secs=8.0, fs=2000.0, noise=0.05, seed=0,
                irregular=0.0, murmur=0.0):
    """合成一段含 S1/S2 的心音,用于离线验证。

    irregular>0: 每拍周期随机抖动(模拟节律不齐)。
    murmur>0:    收缩期(S1→S2)注入带限噪声(模拟杂音),幅度=该值。
    """
    rng = np.random.default_rng(seed)
    n = int(secs * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    base = 60.0 / bpm
    t1 = 0.0
    while t1 < secs:
        t2 = t1 + 0.30             # S2 "dub",收缩期约 0.3s
        for tc, amp, f in ((t1, 1.0, 50.0), (t2, 0.6, 70.0)):
            idx = (t > tc) & (t < tc + 0.10)
            tau = t[idx] - tc
            x[idx] += amp * np.sin(2 * np.pi * f * tau) * np.exp(-tau / 0.02)
        if murmur > 0:             # 收缩期湍流噪声(高频)
            sysmask = (t > t1 + 0.04) & (t < t2 - 0.03)
            hp = signal.butter(4, [200, 600], "band", fs=fs, output="sos")
            burst = signal.sosfilt(hp, rng.standard_normal(n))
            x[sysmask] += murmur * burst[sysmask]
        period = base * (1.0 + irregular * rng.standard_normal())
        t1 += max(0.3, period)     # 防止周期过短
    x += noise * rng.standard_normal(n)
    return x, fs


if __name__ == "__main__":
    import sys

    # Windows 控制台默认 GBK,统一切到 UTF-8 避免 emoji/中文报错
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if "--selftest" in sys.argv:
        an = HeartSoundAnalyzer()

        # 1) 心率精度
        true_bpm = 72.0
        x, fs = _synthesize(bpm=true_bpm, secs=12)
        r = an.analyze(x, fs)
        err = abs(r["bpm"] - true_bpm)
        print(f"[1] 心率: 真值 {true_bpm} -> 估计 {r['bpm']:.1f} BPM "
              f"(置信度 {r['confidence']:.2f})  "
              + ("✅" if err < 3 else f"❌ 误差 {err:.1f}"))

        # 2) 规则节律应判"规则"
        r2 = an.analyze(x, fs)["rhythm"]
        print(f"[2] 规则节律: CV={r2['cv']:.3f} -> {r2['classification']}  "
              + ("✅" if r2["cv"] < 0.05 else "❌"))

        # 3) 不规则节律应判 CV 偏大
        xi, _ = _synthesize(bpm=72, secs=14, irregular=0.18, seed=1)
        r3 = an.analyze(xi, fs)["rhythm"]
        print(f"[3] 不规则节律: CV={r3['cv']:.3f} -> {r3['classification']}  "
              + ("✅" if r3["cv"] > 0.08 else "❌"))

        # 4) 杂音检出:有杂音的收缩期指数应明显高于正常
        xm, _ = _synthesize(bpm=72, secs=12, murmur=0.5, seed=2)
        mu_n = an.analyze(x, fs)["murmur"]["systolic"]
        mu_m = an.analyze(xm, fs)["murmur"]["systolic"]
        ok = (mu_n is not None and mu_m is not None and mu_m > 3 * mu_n)
        print(f"[4] 杂音指数: 正常={mu_n:.3f}  含杂音={mu_m:.3f}  "
              + ("✅" if ok else "❌"))

        # 5) HSMM 分段: S1 起点应对齐真实心跳(周期 T 的整数倍)
        rh = an.analyze(x, fs)
        s1, _ = an.segment_hsmm(rh["env"], rh["bpm"])
        T = 60.0 / true_bpm
        if len(s1) >= 4:
            t_s1 = np.sort(s1) / fs
            # 与最近的真实 S1 时刻(k*T)对齐,统计相位误差
            phase_err = np.mean(np.abs(((t_s1 - t_s1[0] + T / 2) % T) - T / 2))
            ok5 = phase_err < 0.05
        else:
            phase_err, ok5 = 9.9, False
        print(f"[5] HSMM分段: 检出 {len(s1)} 个 S1, 相位误差 {phase_err*1000:.0f}ms  "
              + ("✅" if ok5 else "❌"))

        # 6) SQI: 干净信号应判 ok,强噪声应判 not ok
        xn, _ = _synthesize(bpm=72, secs=12, noise=2.0, seed=3)
        q_ok = an.analyze(x, fs)["sqi"]
        q_bad = an.analyze(xn, fs)["sqi"]
        ok6 = q_ok["ok"] and not q_bad["ok"]
        print(f"[6] SQI门控: 干净 SQI={q_ok['sqi']:.2f}(ok={q_ok['ok']})  "
              f"噪声 SQI={q_bad['sqi']:.2f}(ok={q_bad['ok']})  "
              + ("✅" if ok6 else "❌"))
    else:
        RealtimeHeartRate().run()
