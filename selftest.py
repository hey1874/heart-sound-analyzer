"""
selftest.py — 用合成心音验证算法方向正确性。

从 `heartbeat.py` 的 `__main__` 块抽出,并修掉两个问题:
  1. 原版无论成败都返回退出码 0,无法接入 CI —— 现按失败数返回非零;
  2. 原版**只有阳性用例**(注入 X 应检出 X),缺阴性对照,导致假阳性缺陷
     长期看不出来。例如 S2 分裂:未注入分裂的正常信号本来就会稳定报出
     18-21ms 的"分裂",而原自测只检查"注入 50ms 能否测出 50ms",通过。

每个检查是一个返回 (是否通过, 说明文字) 的函数,`CHECKS` 同时被
`tests/test_selftest.py` 复用,避免两套断言各写一遍。

⚠️ 合成信号只能验证"算法方向对不对",**不能**替代真实心音上的验证与阈值标定。
"""

from __future__ import annotations

import numpy as np

from heartsound import HeartSoundAnalyzer
from heartsound.synth import synthesize

FS = 2000.0
CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


_an = HeartSoundAnalyzer()
_cache: dict[tuple, dict] = {}


def _analyze(**kw) -> dict:
    """带缓存的合成+分析,避免同一段信号在多个检查里重复计算。"""
    key = tuple(sorted(kw.items()))
    if key not in _cache:
        x, fs = synthesize(**kw)
        _cache[key] = _an.analyze(x, fs)
    return _cache[key]


# --- 阳性用例:注入什么就应检出什么 ----------------------------------------

@check("心率精度")
def _c_bpm():
    r = _analyze(bpm=72.0, secs=12)
    err = abs(r["bpm"] - 72.0)
    return err < 3, f"真值 72.0 -> 估计 {r['bpm']:.1f} BPM(置信度 {r['confidence']:.2f})"


@check("心率全量程")
def _c_bpm_range():
    errs = []
    for bpm in (45, 60, 90, 120, 150, 170):
        r = _analyze(bpm=float(bpm), secs=12, seed=11)
        errs.append(abs(r["bpm"] - bpm))
    return max(errs) < 3, f"45-170 BPM 全量程最大误差 {max(errs):.2f} BPM"


@check("规则节律")
def _c_regular():
    r = _analyze(bpm=72.0, secs=12)["rhythm"]
    return r["cv_robust"] < 0.05,         f"稳健CV={r['cv_robust']:.3f}(原始 {r['cv']:.3f})-> {r['classification']}"


@check("不规则节律")
def _c_irregular():
    r = _analyze(bpm=72.0, secs=14, irregular=0.18, seed=1)["rhythm"]
    return r["cv_robust"] > 0.08,         f"稳健CV={r['cv_robust']:.3f}(原始 {r['cv']:.3f})-> {r['classification']}"


@check("杂音检出")
def _c_murmur():
    n = _analyze(bpm=72.0, secs=12)["murmur"]["systolic"]
    m = _analyze(bpm=72.0, secs=12, murmur=0.5, seed=2)["murmur"]["systolic"]
    return m > 3 * n, f"正常={n:.3f}  含杂音={m:.3f}"


@check("HSMM 分段")
def _c_hsmm():
    r = _analyze(bpm=72.0, secs=12)
    # 必须用 env_seg(analyze 内部实际用于分段的窄带包络),不是主带 env
    # ——否则这项测的就不是生产路径。
    s1, _ = _an.segment_hsmm(r["env_seg"], r["bpm"])
    if len(s1) < 4:
        return False, f"仅检出 {len(s1)} 个 S1"
    T = 60.0 / 72.0
    t = np.sort(s1) / FS
    err = float(np.mean(np.abs(((t - t[0] + T / 2) % T) - T / 2)))
    return err < 0.05, f"检出 {len(s1)} 个 S1,相位误差 {err * 1000:.0f}ms"


@check("STI 收缩间期")
def _c_sti():
    s = _analyze(bpm=72.0, secs=12)["sti"]
    ok = (s["systole_ms"] is not None and abs(s["systole_ms"] - 300) < 40
          and s["sys_dia_ratio"] < 1.0)
    return ok, (f"收缩期≈{s['systole_ms']:.0f}ms 舒张期≈{s['diastole_ms']:.0f}ms "
                f"比值={s['sys_dia_ratio']:.2f}")


@check("杂音形状(四型)")
def _c_shape():
    """四种形状都要判对。

    原自测只测钻石型,于是"全收缩期平台型"这一分支从未被覆盖——它的判据
    `env.max()-env.min() < 0.5` 作用在已归一化(max 恒为 1)的包络上而恒为假,
    README 里宣传的这项能力实际从未触发过(平台型 occupancy 达 0.90 仍被判
    "不定形")。
    """
    want = {"plateau": "平台型", "diamond": "钻石型",
            "crescendo": "递增型", "decrescendo": "递减型"}
    bad, detail = [], []
    for shape, kw in want.items():
        m = _analyze(bpm=72.0, secs=12, murmur=0.6, murmur_shape=shape,
                     seed=4)["murmur"]
        got = m.get("shape") or ""
        detail.append(f"{shape}->{got.split('(')[0]}")
        if kw not in got:
            bad.append(shape)
    return not bad, "  ".join(detail)


@check("舒张期杂音")
def _c_diastolic():
    """单时相判别不能只对收缩期成立。"""
    m = _analyze(bpm=72.0, secs=12, murmur=0.5, murmur_phase="diastolic",
                 seed=9)["murmur"]
    ok = "舒张期杂音" in m["flag"] and not m.get("noise_suspect")
    return ok, (f"收缩期={m['systolic']:.2f} 舒张期={m['diastolic']:.2f} "
                f"-> 「{m['flag']}」")


@check("强杂音下时相归属不判反")
def _c_loud_phase():
    """回归:杂音响过心音本体时,分段被带偏导致时相判反。

    改动前用主带 20-200Hz / 4 阶的包络做分段,滚降不够陡,强杂音漏进包络
    污染 HSMM 发射模型;舒张期杂音幅度 1.6 时被报成收缩期。现改用独立窄带
    30-160Hz / 8 阶——该取值在真实数据 F1 与本项合成用例上同时最优。
    """
    bad, detail = [], []
    for amp in (0.5, 1.0, 1.6, 2.0):
        m = _analyze(bpm=72.0, secs=12, murmur=amp, noise=0.06,
                     murmur_phase="diastolic", seed=7)["murmur"]
        ok = (m["diastolic"] or 0) > (m["systolic"] or 0)
        detail.append(f"amp={amp}:{m['systolic']:.2f}/{m['diastolic']:.2f}"
                      + ("" if ok else "✗"))
        if not ok:
            bad.append(amp)
    return not bad, "舒张期杂音 收缩/舒张指数  " + "  ".join(detail)


@check("S3 奔马律")
def _c_s3():
    n = _analyze(bpm=72.0, secs=12)["extra_sounds"]
    s = _analyze(bpm=72.0, secs=12, s3=0.5, seed=5)["extra_sounds"]
    ok = (s["s3"] is not None and n["s3"] is not None
          and s["s3"] > 0.20 and s["s3"] > 2 * n["s3"])
    return ok, f"正常 s3={n['s3']:.3f}  含S3 s3={s['s3']:.3f} -> 「{s['flag']}」"


@check("S2 分裂(注入 50ms)")
def _c_split():
    sp = _analyze(bpm=72.0, secs=12, s2_split_ms=50, seed=6)["s2_split"]
    ok = sp["split_ms"] is not None and abs(sp["split_ms"] - 50) < 20
    return ok, f"注入 50ms -> 估计 {sp['split_ms']:.0f}ms(占比 {sp.get('split_frac', 0):.0%})"


# --- 阴性对照:没注入的东西**不应**被检出 -----------------------------------

@check("阴性:S2 分裂假阳性")
def _c_split_neg():
    """正常信号不应报出可观的分裂。

    修复前:5 个随机种子全部报出 18-21ms"分裂"、13%-27% 的周期检出双峰,
    测到的其实是 70Hz S2 的希尔伯特包络纹波(周期 14ms ≈ 当时 15ms 的最小
    峰间距)。这一项就是为了锁住该回归。
    """
    worst, detail = 0.0, []
    for seed in range(5):
        sp = _analyze(bpm=72.0, secs=12, seed=seed)["s2_split"]
        ms = sp.get("split_ms") or 0.0
        frac = sp.get("split_frac") or 0.0
        worst = max(worst, ms)
        detail.append(f"{ms:.0f}ms/{frac:.0%}")
    return worst < 15.0, "5 个种子的正常信号 -> " + " ".join(detail)


@check("阴性:宽带噪声不应判为杂音")
def _c_noise_not_murmur():
    """环境噪声均匀铺满整个心动周期,收缩期与舒张期高频能量近似相等。

    修复前:20 段仅加白噪声(noise=0.25)的正常信号,20 段全部同时报
    "疑似收缩期杂音、疑似舒张期杂音"。现由 asymmetry / noise_suspect 识别。
    """
    caught = 0
    for s in range(8):
        m = _analyze(bpm=72.0, secs=12, noise=0.25, seed=100 + s)["murmur"]
        if m.get("noise_suspect") or m["flag"] == "未见明显杂音":
            caught += 1
    return caught == 8, f"8 段噪声信号中 {caught} 段被正确识别为噪声/无杂音"


@check("阴性:真杂音不应被误判为噪声")
def _c_murmur_not_noise():
    """噪声判据不能把真杂音一起吃掉——真杂音只落在单一时相,不对称度高。"""
    m = _analyze(bpm=72.0, secs=12, murmur=0.5, seed=2)["murmur"]
    ok = (not m.get("noise_suspect")) and "收缩期杂音" in m["flag"]
    return ok, (f"收缩期={m['systolic']:.2f} 舒张期={m['diastolic']:.2f} "
                f"不对称度={m['asymmetry']:.2f} -> 「{m['flag']}」")


# --- 门控:坏信号必须被拦下 --------------------------------------------------

@check("SQI 门控:强噪声")
def _c_sqi():
    ok_r = _analyze(bpm=72.0, secs=12)["sqi"]
    bad = _analyze(bpm=72.0, secs=12, noise=2.0, seed=3)["sqi"]
    return ok_r["ok"] and not bad["ok"], \
        f"干净 SQI={ok_r['sqi']:.2f}(ok={ok_r['ok']})  噪声 SQI={bad['sqi']:.2f}(ok={bad['ok']})"


@check("SQI 门控:50Hz 工频")
def _c_sqi_hum():
    """工频落在 20-200Hz 带内,修复前会把 band_ratio 抬到 1.00、SQI 判 ok,
    而心率已经完全错了(72 -> 109/113 BPM)。"""
    clean = _analyze(bpm=72.0, secs=12, seed=5)["sqi"]
    out = []
    caught = 0
    for amp in (0.5, 2.0, 5.0):
        r = _analyze(bpm=72.0, secs=12, seed=5, hum_hz=50.0, hum=amp)
        q = r["sqi"]
        caught += (not q["ok"])
        out.append(f"amp={amp}: hum={q['hum']:.2f} SQI={q['sqi']:.2f} ok={q['ok']}")
    return clean["ok"] and caught == 3, \
        f"干净 ok={clean['ok']} | " + " | ".join(out)


@check("SQI 门控:高采样率输入的削波")
def _c_sqi_clip():
    """削波必须在重采样前判定:抗混叠滤波会把削波平顶磨圆。

    修复前:同一段硬削波信号以 2000Hz 输入 clip=0.027(拒绝),以 44kHz
    输入(声卡的实际情况)clip=0.000(放行)。
    """
    from scipy import signal as _sig
    x, _ = synthesize(bpm=72.0, secs=12, seed=6)
    x = x / np.max(np.abs(x))
    # 必须**先升采样再削波**:削波发生在声卡的 44kHz 端。反过来做(先削波
    # 再升采样)会被插值把平顶磨圆,那样测的就不是削波信号了。
    lo = _an.analyze(np.clip(x * 3.0, -1.0, 1.0), FS)["sqi"]
    hi = _an.analyze(np.clip(_sig.resample_poly(x, 22, 1) * 3.0, -1.0, 1.0),
                     44000.0)["sqi"]
    ok = (not lo["ok"]) and (not hi["ok"])
    return ok, (f"2kHz 输入 clip={lo['clip']:.3f}(ok={lo['ok']})  "
                f"44kHz 输入 clip={hi['clip']:.3f}(ok={hi['ok']})")


@check("阴性:轻度工频不应被拒")
def _c_hum_neg():
    """门控本身也要有阴性对照,否则只会往"越拒越多"的方向漂。

    工频幅度 0.05-0.2 时 hum 指标已达 0.17-0.75,但心率估计仍完全正确
    (72.0/72.2,置信度 0.85/0.58)——此时拒收是过度拒绝,会把好录音挡在
    门外、逼用户去用 --force,反而架空门控。
    """
    out, bad = [], []
    for amp in (0.05, 0.1, 0.2):
        r = _analyze(bpm=72.0, secs=12, seed=5, hum_hz=50.0, hum=amp)
        ok_bpm = r["bpm"] is not None and abs(r["bpm"] - 72.0) < 3
        out.append(f"amp={amp}: hum={r['sqi']['hum']:.2f} "
                   f"bpm={r['bpm']:.1f} reliable={r['reliable']}")
        if ok_bpm and not r["reliable"]:
            bad.append(f"amp={amp} 心率仍准确却被拒")
    return not bad, " | ".join(out) + ("  ← " + "; ".join(bad) if bad else "")


@check("阴性:响亮真杂音不应被门控挡掉")
def _c_loud_murmur_passes():
    """杂音能量落在 20-200Hz 带外,会压低 band_ratio。若拿它硬性拒绝,
    最该被检出的病例反而先被挡在门外。"""
    blocked = []
    for amp in (0.3, 0.5, 0.8, 1.2):
        for phase in ("systolic", "diastolic"):
            r = _analyze(bpm=72.0, secs=12, murmur=amp, noise=0.06,
                         murmur_phase=phase, seed=7)
            if not r["reliable"]:
                blocked.append(f"amp={amp}/{phase}: {r['unreliable_reason']}")
    return not blocked, ("8 个真杂音用例全部通过门控" if not blocked
                         else "; ".join(blocked))


@check("退化输入不崩溃")
def _c_degenerate():
    """过短/全零输入应返回 reliable=False,而不是抛异常或给出一个自信的数。

    修复前 predict.py 对 bpm=None 直接格式化,抛
    TypeError: unsupported format string passed to NoneType.__format__
    """
    clean = synthesize(bpm=72.0, secs=6.0)[0]
    contaminated = clean.copy()
    contaminated[5000:5400] = np.nan          # 4% 非有限值 -> 应判坏数据
    speckled = clean.copy()
    speckled[[100, 500, 900]] = np.inf        # 极少量 -> 置零后继续
    cases = []
    for name, x in (("0.2s", synthesize(bpm=72.0, secs=1.0)[0][:400]),
                    ("全零", np.zeros(4000)),
                    ("空", np.zeros(0)),
                    ("含NaN(4%)", contaminated),
                    ("常数", np.full(8000, 3.7))):
        try:
            r = _an.analyze(x, FS)
            cases.append(f"{name}: reliable={r['reliable']}")
            if r["reliable"]:
                return False, f"{name} 不应判为可采信"
        except Exception as e:                    # noqa: BLE001
            return False, f"{name} 抛出 {type(e).__name__}: {e}"
    # 极少量非有限值应被置零后正常分析,而不是整段作废
    try:
        r = _an.analyze(speckled, FS)
    except Exception as e:                        # noqa: BLE001
        return False, f"含3个inf 抛出 {type(e).__name__}: {e}"
    if r["bpm"] is None or abs(r["bpm"] - 72.0) > 3:
        return False, f"含3个inf 的 6 秒录音不应作废(bpm={r['bpm']})"
    cases.append(f"含3个inf: bpm={r['bpm']:.1f} reliable={r['reliable']}")
    return True, "  ".join(cases)


@check("频域 HRV 门控")
def _c_hrv():
    """LF 下界 0.04Hz 要求单段 ≥25 秒;修复前门限 min_rr=20 在 72BPM 下
    只需 18 秒就判"可用",与 docstring 声称的"≥1-2 分钟"矛盾。"""
    short = _analyze(bpm=72.0, secs=12)["hrv_freq"]
    mid = _analyze(bpm=72.0, secs=40, irregular=0.06, seed=7)["hrv_freq"]
    long = _analyze(bpm=72.0, secs=150, irregular=0.06, seed=7)["hrv_freq"]
    ok = (not short["available"]) and (not mid["available"]) and long["available"]
    return ok, (f"12s 可用={short['available']}  40s 可用={mid['available']}  "
                f"150s 可用={long['available']}"
                + (f" LF/HF={long['lf_hf']:.2f}" if long["available"] else ""))


def main(argv: list[str] | None = None) -> int:
    """跑全部检查,打印结果,返回失败数作为退出码。"""
    from heartsound.cliutil import Parser
    a = Parser("用合成心音验证算法方向正确性(失败返回非零退出码)").parse_args(
        argv if argv is not None else [])
    del a
    failed = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        try:
            ok, detail = fn()
        except Exception as e:                    # noqa: BLE001
            ok, detail = False, f"抛出 {type(e).__name__}: {e}"
        failed += not ok
        print(f"[{i:2d}] {name}: {detail}  " + ("✅" if ok else "❌"))
    total = len(CHECKS)
    print(f"\n{total - failed}/{total} 通过" + ("" if failed else "  ✅ 全部通过"))
    return 1 if failed else 0


if __name__ == "__main__":
    from heartsound.cliutil import run
    run(main)
