"""监视界面:用合成信号直接驱动,不依赖真麦克风。

重点验证两条设计约定,它们比"能显示"更重要:
  1. 信号质量不合格时,`reliable=False` 且必须给出可操作的原因;
  2. 传给前端的负载必须是可 JSON 序列化的纯量/数组——numpy 类型混进去会让
     整个接口 500,而这类错误只在真机上跑才会暴露。
"""

import json

import numpy as np
import pytest

from heartsound import HeartSoundAnalyzer
from heartsound.monitor import _build_payload, _downsample
from heartsound.synth import synthesize

AN = HeartSoundAnalyzer()


def _payload(**kw):
    x, fs = synthesize(secs=8, **kw)
    return _build_payload(AN, AN.analyze(x, fs), int(fs), "dummy", 0)


# --- 序列化契约 -------------------------------------------------------------

def test_payload_is_json_serializable():
    """numpy 标量/数组混进负载会让接口整个 500,且只在真机上才暴露。"""
    s = json.dumps(_payload(bpm=72, murmur=0.5, seed=2), ensure_ascii=False)
    assert len(s) > 100
    back = json.loads(s)
    assert back["status"] == "ok"


def test_payload_has_fields_the_page_reads():
    d = _payload(bpm=72, seed=1)
    for k in ("reliable", "reason", "bpm", "confidence", "sqi",
              "wave", "env", "beats", "s1", "s2", "window_s",
              "rhythm", "murmur_flag", "extra_flag", "split_flag"):
        assert k in d, f"负载缺字段 {k}"


# --- 门控在界面上的表现 -----------------------------------------------------

def test_good_signal_marked_reliable():
    d = _payload(bpm=72, noise=0.05, seed=3)
    assert d["reliable"] is True
    assert d["reason"] is None
    assert d["bpm"] is not None and abs(d["bpm"] - 72) < 3


@pytest.mark.parametrize("kw,keyword", [
    (dict(noise=2.0, seed=4), None),                       # 强噪声
    (dict(hum_hz=50.0, hum=5.0, seed=5), "工频"),          # 工频干扰
])
def test_bad_signal_is_rejected_with_actionable_reason(kw, keyword):
    """不合格时必须给原因——界面靠它决定是否把判定区变灰。"""
    d = _payload(bpm=72, **kw)
    assert d["reliable"] is False
    assert d["reason"], "拒绝时必须说明原因"
    if keyword:
        assert keyword in d["reason"]


def test_segmentation_marks_are_normalized_positions():
    """S1/S2 与心音峰以 0..1 的相对位置给出,前端直接乘画布宽度。"""
    d = _payload(bpm=72, seed=6)
    for key in ("s1", "s2", "beats"):
        for v in d[key]:
            assert 0.0 <= v <= 1.0, f"{key} 超出 0..1: {v}"
    assert len(d["s1"]) >= 3, "干净信号应检出若干 S1"


# --- 绘图降采样 -------------------------------------------------------------

@pytest.mark.parametrize("spike_at", [0, 6789, 12345, 19998, 19999])
def test_downsample_preserves_extremes(spike_at):
    """分段取极值,不能漏掉尖峰——包括**落在末尾**的。

    早期实现按整段截断,末尾不足一段的样本被直接丢弃;对一个以判断信号
    质量为要务的界面,把尖峰藏起来正好掩盖了该看见的问题。
    """
    x = np.zeros(20000)
    x[spike_at] = 5.0
    x[(spike_at + 3) % 20000] = -4.0
    d = _downsample(x, 1000)
    assert d["n"] <= 1000, f"输出 {d['n']} 列,超过上限 1000"
    assert max(d["hi"]) >= 4.9, f"尖峰在 {spike_at} 处被抹掉了"
    assert min(d["lo"]) <= -3.9, f"负向尖峰在 {spike_at} 附近被抹掉了"


def test_downsample_returns_three_layers():
    """每列给 (最小, 最大, RMS) 三个量。

    只画 min/max 包络,心音这类脉冲信号会呈现为"平线 + 孤立尖峰";RMS 层
    把中间填实,而且它表达的是真实能量,不是把振幅拉伸的视觉作弊。
    """
    x, _ = synthesize(bpm=72, secs=8, seed=1)
    d = _downsample(x, 500)
    assert d["n"] == 500
    for k in ("lo", "hi", "rms"):
        assert len(d[k]) == d["n"], f"{k} 长度与列数不符"
    lo, hi, rms = map(np.array, (d["lo"], d["hi"], d["rms"]))
    assert (hi >= lo).all(), "每列必须 max >= min"
    assert (rms >= 0).all(), "RMS 不可为负"
    assert (rms <= np.maximum(np.abs(lo), np.abs(hi)) + 1e-9).all(),         "RMS 不可超过该列的峰值"


def test_downsample_ref_is_robust_percentile():
    """归一化基准用 99.5 分位而非绝对最大值。

    最大值被 S1 那一根尖峰独占,拿它做基准会把其余部分全压扁
    (实测列高中位从 8.9px 掉到 6.5px)。
    """
    x, _ = synthesize(bpm=72, secs=8, seed=1)
    x = np.concatenate([x, [x.max() * 20]])          # 注入一个孤立巨峰
    d = _downsample(x, 500)
    assert d["ref"] < np.abs(x).max() / 5, "基准被单个离群峰带跑了"


def test_downsample_short_input():
    d = _downsample(np.arange(10.0), 1000)
    assert d["n"] == 10 and len(d["hi"]) == 10


# --- 数据链对齐 -------------------------------------------------------------

@pytest.mark.parametrize("secs", [4, 6, 8, 12, 20])
def test_spectrogram_covers_full_time_axis(secs):
    """回归:频谱图抽帧曾直接截断尾部,时间轴对不上。

    早先写成 m[:, :k*n_frames](k 为整除商),8 秒窗只剩 89% 的时间轴、
    12 秒窗只剩 59%,却仍铺满整个画布并标秒刻度——频谱图与包络图上的
    S1/S2 因此错位。这类错误肉眼很难发现,但会让人读错杂音的时相。
    """
    from heartsound.cnn import logmel
    from heartsound.monitor import _spectrogram
    an = HeartSoundAnalyzer()
    x, fs = synthesize(bpm=72, secs=secs, seed=1)
    xr = an.resample(x, fs)
    full = logmel(xr).shape[1]
    sp = _spectrogram(xr, an.fs)
    assert sp["w"] == min(full, 220)

    # 把最后 5% 的时间放大 40 倍,抽样后这段必须仍是全图最亮处——
    # 证明尾部没有被截掉。不要求"最后一列恰好是 255":末尾几帧的 STFT
    # 窗口跨在放大区边界上,那样断言是在过度指定实现细节。
    xr2 = xr.copy()
    xr2[-len(xr2) // 20:] *= 40
    sp2 = _spectrogram(xr2, an.fs)
    d = np.array(sp2["data"], float).reshape(sp2["w"], sp2["h"])
    tail = max(3, sp2["w"] // 20)                 # 放大区对应的列数
    mid = d[sp2["w"] // 3: 2 * sp2["w"] // 3].max()
    assert d[-tail:].max() > mid + 40, (
        f"时间轴末尾被截掉了:末段最大 {d[-tail:].max():.0f} "
        f"未显著高于中段 {mid:.0f}")
    assert d.argmax() // sp2["h"] >= sp2["w"] - tail, "全图最亮处不在放大区"


def test_all_charts_share_one_time_axis():
    """波形、包络、频谱图必须张成同一段时间——它们叠在同一条秒刻度上读。"""
    an = HeartSoundAnalyzer()
    x, fs = synthesize(bpm=72, secs=8, seed=1)
    r = an.analyze(x, fs)
    p = _build_payload(an, r, int(fs), "d", 0)
    dur = len(r["env"]) / an.fs
    assert abs(p["window_s"] - dur) < 0.02
    # S1/S2/心音峰都以 0..1 的相对位置给出,直接乘画布宽度
    for k in ("s1", "s2", "beats"):
        assert all(0.0 <= v <= 1.0 for v in p[k]), f"{k} 越界"
    # 三张图的数据量都非零且各自自洽
    assert p["wave"]["n"] > 0 and p["env"]["n"] > 0 and p["spec"]["w"] > 0


# --- 听诊区 -----------------------------------------------------------------

def test_sites_match_circor_labels():
    """AV/PV/TV/MV 必须与 CirCor 的听诊区标签一致。

    这不是巧合而是设计:录音时标上位置,数据才能直接喂给 calibrate.py。
    对不上,那条标定路径就断了。
    """
    from heartsound.sites import BY_CODE, CIRCOR_CODES, ORDER, SITES
    assert set(CIRCOR_CODES) <= {s["code"] for s in SITES}
    assert set(ORDER) == {s["code"] for s in SITES}, "顺序表与站点表不一致"
    for s in SITES:
        assert s["landmark"] and s["listen_for"], f"{s['code']} 缺定位或听诊说明"
        x, y = s["xy"]
        assert 0 <= x <= 1 and 0 <= y <= 1, f"{s['code']} 示意图坐标越界"
    # 解剖左右:主动脉瓣区在胸骨**右**缘,肺动脉瓣区在**左**缘
    assert BY_CODE["AV"]["xy"][0] < 0.5 < BY_CODE["PV"]["xy"][0]
    # 心尖区最靠下、最靠左
    assert BY_CODE["MV"]["xy"][1] == max(s["xy"][1] for s in SITES)


def test_state_tracks_site_and_best_quality():
    """切换听诊区后,各区的最佳 SQI 要分别记账——这是"挪一挪找位置"的依据。"""
    from heartsound.monitor import State
    st = State()
    assert st.site == "MV"
    st.note_quality(0.5)
    st.note_quality(0.7)
    assert st.set_site("AV") is True
    assert st.set_site("不存在") is False, "非法听诊区必须拒绝"
    st.note_quality(0.9)
    st.note_quality(0.3)                      # 更差的不该覆盖
    d = st.get()
    assert d["site"] == "AV"
    assert d["best_sqi"] == {"MV": 0.7, "AV": 0.9}


def test_state_get_is_a_snapshot():
    """get() 必须返回副本,调用方改它不能影响内部状态。"""
    from heartsound.monitor import State
    st = State()
    st.set({"status": "ok"})
    d = st.get()
    d["status"] = "tampered"
    assert st.get()["status"] == "ok"


def test_diagram_sidedness_is_self_consistent_and_labelled():
    """示意图左右必须与文字定位一致,且视角必须在图上标明。

    这是**会导致贴错位置**的歧义:图按正面视角画(面对受检者),所以受检者
    的左侧出现在图的右边——主动脉瓣区在胸骨右缘,却画在图的左侧。图上不标
    视角,就会有人当成"这是我自己"而把左右完全贴反。
    """
    from heartsound.sites import SITES, VIEW_NOTE
    assert "正面视角" in VIEW_NOTE and "左侧" in VIEW_NOTE
    for s in SITES:
        assert s["side"] in ("L", "R"), f"{s['code']} 缺 side"
        assert 1 <= s["ics"] <= 6, f"{s['code']} 肋间号不合理"
        # 正面视角:受检者右侧 -> 图的左半(x<0.5),反之亦然
        assert (s["side"] == "R") == (s["xy"][0] < 0.5), \
            f"{s['code']} 的 side={s['side']} 与示意图坐标 x={s['xy'][0]} 矛盾"
        # 文字定位里的"左/右缘"要和 side 对上
        if "右缘" in s["landmark"]:
            assert s["side"] == "R", f"{s['code']} 文字写右缘但 side=L"
        if "左缘" in s["landmark"] or "左锁骨" in s["landmark"]:
            assert s["side"] == "L", f"{s['code']} 文字写左侧但 side=R"


def test_sites_ics_increases_downward_on_diagram():
    """肋间号越大位置越靠下——图上的 y 坐标必须单调对应。"""
    from heartsound.sites import SITES
    pairs = sorted((s["ics"], s["xy"][1]) for s in SITES)
    ys = [y for _, y in pairs]
    assert ys == sorted(ys), f"肋间号与图上高度不单调:{pairs}"
