"""临床特征:节律、STI、杂音、频谱轮廓、额外音、S2 分裂、HRV

由 heartbeat.py 拆分而来。方法体逐字保留,行为与拆分前完全一致——
拆分只改代码组织,不改算法。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import _movavg  # noqa: F401  (features/segmentation 中用到)


class FeatureMixin:
    """临床特征:节律、STI、杂音、频谱轮廓、额外音、S2 分裂、HRV"""

    SPEC_SUBBANDS = ((20, 60), (60, 120), (120, 200),
                     (200, 300), (300, 450), (450, 700))

    def analyze_rhythm(self, s1: np.ndarray) -> dict:
        """从 S1 间期(RR)评估节律规则性。

        输出心率变异系数 CV、Poincaré 的 SD1/SD2、早搏计数与文字判读。
        """
        if len(s1) < 4:
            return {"classification": "数据不足", "n_rr": 0}
        rr = np.diff(s1) / self.fs * 1000.0            # 毫秒
        mean_rr = float(rr.mean())
        cv = float(rr.std() / (mean_rr + 1e-12))
        # 稳健 CV:用四分位距估标准差(/1.349),对离群 RR 不敏感。
        #
        # 为什么需要它:分段 F1 约 0.93,即约 7% 的 S1 检漏/多检,而**每一个
        # 分段错误都会同时污染两个 RR 间期**,产生一个约两倍或半倍长的离群值。
        # CV 是二阶统计量,离群值对它的影响远大于对心率中位数的影响——这就是
        # 为什么分段 F1 0.93 足以支撑心率和间期中位数,却支撑不住变异性指标。
        #
        # CirCor 实测(269 条,真值由官方 S1 标注算出):原始 CV 中位 0.082 而
        # 真值 0.044,三档判读一致率仅 43%;改用稳健 CV 后中位 0.051、
        # 一致率 65%。
        #
        # ⚠️ 代价:稳健估计同样会削掉**真实的**早搏/房颤造成的离群 RR。本模块
        # 分不清"分段错误"和"心律失常"——两者在 RR 序列上长得一样。因此
        # cv/cv_robust 都保留在输出里,`ectopic` 计数也一并给出,由使用者判断。
        q1, q3 = np.percentile(rr, [25, 75])
        cv_robust = float((q3 - q1) / 1.349 / (np.median(rr) + 1e-12))
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

        # 判读用稳健 CV(见上),原始 cv 一并输出供对照
        if cv_robust < 0.05:
            cls = "规则"
        elif cv_robust < 0.12:
            cls = "轻度不齐(可能为窦性心律不齐,常见且多属正常)"
        else:
            cls = "明显不规则(建议就医排查早搏/房颤;也可能是分段不准)"
        return {
            "classification": cls, "n_rr": int(len(rr)),
            "mean_bpm": 60000.0 / (mean_rr + 1e-12),
            "cv": cv, "cv_robust": cv_robust,
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

    def _spectral_descriptors(self, seg: np.ndarray) -> list[float]:
        """一段波形的频谱描述量。子带占比 + 形状统计量。"""
        n = len(self.SPEC_SUBBANDS) + 6
        if len(seg) < 64:
            return [np.nan] * n
        f, p = signal.welch(seg, fs=self.fs, nperseg=min(len(seg), 256))
        tot = float(p.sum()) + 1e-20
        out = [float(p[(f >= lo) & (f < hi)].sum() / tot)
               for lo, hi in self.SPEC_SUBBANDS]
        pn = p / tot
        centroid = float((f * pn).sum())                       # 谱质心
        spread = float(np.sqrt(((f - centroid) ** 2 * pn).sum()))
        cs = np.cumsum(pn)
        rolloff = float(f[min(np.searchsorted(cs, 0.85), len(f) - 1)])
        gm = float(np.exp(np.mean(np.log(p + 1e-20))))
        flatness = gm / (float(p.mean()) + 1e-20)               # 谱平坦度
        slope = float(np.polyfit(f, np.log(p + 1e-20), 1)[0])   # 谱倾斜
        hi_lo = float(p[f >= 200].sum()
                      / (p[(f >= 20) & (f < 200)].sum() + 1e-20))
        return out + [centroid, spread, rolloff, flatness, slope, hi_lo]

    def spectral_profile(
        self, x_band: np.ndarray, cycles: list[tuple[int, int, int]]
    ) -> dict:
        """收缩期与舒张期安静段的频谱轮廓。

        为什么需要它:`murmur_index` 把整个收缩期的频谱压成**一个标量**(高频
        能量比),丢掉了杂音的频谱形状——而形状恰恰是听诊学区分杂音的依据。
        CirCor 实测,单靠 murmur_index 的 AUC 只有 0.715;把这里的 24 个描述量
        加进特征向量后,通过门控的录音上 AUC 升到 **0.840**(仅用这 24 维即可),
        与工程特征合并后 0.838。**没有用任何神经网络。**

        每个心动周期分别算,再用**中位数与 90 分位数**两种方式聚合。

        为什么要两个聚合量:杂音常只在**部分**心动周期清晰可闻(呼吸调制、
        体位、噪声干扰),中位数会把它冲淡。这本质上是个多示例问题——一条
        录音阳性当且仅当**存在**含杂音的周期。CirCor 实测(885 条录音):
        仅中位数 AUC 0.853,加上 90 分位数升到 **0.879**;再加标准差/10 分位
        则降到 0.874-0.875(维度翻倍但无新信息,反而稀释)。
        """
        stats = [f"b{lo}_{hi}" for lo, hi in self.SPEC_SUBBANDS] + [
            "centroid", "spread", "rolloff85", "flatness", "slope", "hi_lo"]
        keys = [f"{ph}_{ag}_{k}" for ph in ("sys", "dia")
                for ag in ("med", "p90") for k in stats]
        if not cycles:
            return dict.fromkeys(keys, None)
        g = max(1, int(0.04 * self.fs))
        vals = []
        for segs in ([x_band[a + g: s2 - g] for a, s2, b in cycles
                      if s2 - g > a + g],
                     [x_band[s2 + g: b - g] for a, s2, b in cycles
                      if b - g > s2 + g]):
            if not segs:
                vals += [None] * (2 * len(stats))
                continue
            d = np.array([self._spectral_descriptors(s) for s in segs], float)
            # 逐列先看有没有有效值再聚合。`_spectral_descriptors` 对短于 64
            # 样本的段返回全 NaN;心率很高时(护带 ±40ms 一扣,安静段可能只剩
            # 几十个样本)整列都是 NaN,nanmedian 会刷出 "All-NaN slice" 警告。
            #
            # 注意外面那层 np.errstate **拦不住**它——errstate 管的是浮点异常
            # (除零、溢出),而这是 warnings 模块的 RuntimeWarning。真实录音上
            # 一次就刷了 48 条。
            ok = np.isfinite(d).any(axis=0)
            agg = np.full(2 * d.shape[1], np.nan)
            if ok.any():
                sub = d[:, ok]
                agg[np.flatnonzero(ok)] = np.nanmedian(sub, axis=0)
                agg[d.shape[1] + np.flatnonzero(ok)] = np.nanpercentile(
                    sub, 90, axis=0)
            vals += [None if not np.isfinite(v) else float(v) for v in agg]
        return dict(zip(keys, vals))

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
