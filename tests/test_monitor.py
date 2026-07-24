"""监视界面:用合成信号直接驱动,不依赖真麦克风。

重点验证两条设计约定,它们比"能显示"更重要:
  1. 信号质量不合格时,`reliable=False` 且必须给出可操作的原因;
  2. 传给前端的负载必须是可 JSON 序列化的纯量/数组——numpy 类型混进去会让
     整个接口 500,而这类错误只在真机上跑才会暴露。
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path

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


def test_site_coords_calibrated_to_the_anatomical_plate():
    """听诊点坐标必须与底图 thorax.png 上真实的肋骨位置对得上。

    坐标是在 Gray's Anatomy 版画上量出胸骨角(第 2 肋)、肋附着点间距、
    胸骨左右缘与锁骨中线之后标定的,不是摆着好看。这里检验的是标定关系,
    而非具体数值——数值可以微调,关系不能破。
    """
    from heartsound.sites import BY_CODE

    xs = {c: BY_CODE[c]["xy"][0] for c in BY_CODE}
    ys = {c: BY_CODE[c]["xy"][1] for c in BY_CODE}

    # 胸骨中线约在 0.50:AV 在其右(图左),其余在其左(图右)
    assert xs["AV"] < 0.5 < xs["PV"]
    # AV / PV 对称贴在胸骨两缘,离中线的距离应接近
    assert abs((0.5 - xs["AV"]) - (xs["PV"] - 0.5)) < 0.03

    # 左胸骨缘的三个点(PV/ERB/TV)x 必须一致
    assert xs["PV"] == xs["ERB"] == xs["TV"], "左胸骨缘三点未对齐"
    # 心尖区在锁骨中线,明显比胸骨缘更外侧
    assert xs["MV"] > xs["TV"] + 0.08, "心尖区不够外侧"

    # 同一肋间等高;肋间号每 +1,y 增量应大致相等(肋骨等距)
    assert ys["AV"] == ys["PV"], "同为第 2 肋间却不等高"
    steps = [ys["ERB"] - ys["PV"], ys["TV"] - ys["ERB"], ys["MV"] - ys["TV"]]
    assert all(st > 0 for st in steps), "肋间号增大时位置未下移"
    assert max(steps) - min(steps) < 0.02, f"肋间距不均匀:{steps}"


def test_thorax_image_ships_with_the_package():
    """底图必须随包发布,否则装完界面是空的。"""
    import os
    d = os.path.join(os.path.dirname(__file__), "..", "heartsound")
    png = os.path.join(d, "thorax.png")
    assert os.path.exists(png), "缺 heartsound/thorax.png"
    assert os.path.getsize(png) < 200_000, "底图过大,会拖慢页面加载"
    with open(png, "rb") as fh:
        assert fh.read(8) == bytes([137, 80, 78, 71, 13, 10, 26, 10]), \
            "不是有效的 PNG"
    toml = open(os.path.join(d, "..", "pyproject.toml"), encoding="utf-8").read()
    assert "thorax.png" in toml, "pyproject.toml 未把底图列入 package-data"


def test_diagram_labels_orientation_and_credits_source():
    """图上必须标出左右,并署明底图来源。"""
    import os
    ui = open(os.path.join(os.path.dirname(__file__), "..", "heartsound",
                           "ui.html"), encoding="utf-8").read()
    assert "受检者右" in ui and "受检者左" in ui, "示意图未标出左右"
    assert "thorax.png" in ui, "界面未引用解剖底图"
    assert "Gray" in ui, "未标注底图出处(公有领域也应署名)"


def _ui_css():
    from heartsound import monitor
    s = (Path(monitor.__file__).parent / "ui.html").read_text(encoding="utf-8")
    return s[s.index("<style>") + 7:s.index("</style>")]


def test_ui_css_is_structurally_valid():
    """浏览器遇到非法 CSS 只会静默丢弃整条规则,没有任何报错。

    起因是真出过一次:`:root[data-theme="dark"] .torso image, @media (...){...}`
    ——@media 混进了选择器列表。规则被整条丢掉,暗色主题下白底版画就直接
    糊在深色卡片上。测试跑得再多也发现不了,因为它根本不影响 Python。
    """
    css = _ui_css()
    assert css.count("{") == css.count("}"), "花括号不配平"

    depth = 0
    for i, ch in enumerate(css):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif css.startswith("@media", i):
            assert depth == 0, f"@media 嵌在规则内部(偏移 {i})"
        assert depth >= 0, f"多余的右花括号(偏移 {i})"

    assert not re.search(r"[^{}\n]*,\s*\n?\s*@media", css), \
        "@media 出现在选择器列表里——整条规则会被丢弃"
    for m in re.finditer(r"([^{}]+)\{\s*\}", css):
        raise AssertionError(f"空规则(多半是删改的残留): {m.group(1).strip()[:60]}")


def test_ui_defines_plate_filter_for_every_theme():
    """底图是白底黑线,暗色主题必须反相,否则是一块刺眼白板。

    四套主题定义(:root、prefers-color-scheme、data-theme=light/dark)必须
    都给 --plate-filter 赋值:漏掉任何一套,那套主题下就会继承错的值。
    """
    css = _ui_css()
    assert len(re.findall(r"--plate-filter\s*:", css)) == 4, \
        "四套主题都要定义 --plate-filter"
    assert re.search(r"\.torso image\{[^}]*filter:var\(--plate-filter", css), \
        "底图未套用 --plate-filter"
    dark = re.findall(r"--plate-filter:\s*([^;}]+)", css)
    assert sum("invert" in v for v in dark) == 2, "两套暗色主题都要反相"


def test_demo_mode_makes_no_network_requests():
    """演示页没有后端,而 fetch 用的是相对路径——不守卫就会打到宿主站点上。

    必须按块作用域判断,不能只在 fetch 之前找 `__DEMO__` 字样:文件里别处
    也出现这个词,那样写出来的断言恒真(第一版就是这么漏的)。
    """
    from heartsound import monitor
    s = (Path(monitor.__file__).parent / "ui.html").read_text(encoding="utf-8")
    js = s[s.index("</style>"):]

    def scan(tokens):
        """扫一遍 js,对每个 token 记录它所处的块头栈。"""
        stack, hits, pos = [], [], 0
        for m in re.finditer(r"[{}]|" + tokens, js):
            tok = m.group(0)
            if tok == "{":
                stack.append(js[pos:m.start()])      # 这一层块的"头部"
            elif tok == "}":
                if stack:
                    stack.pop()
            else:
                hits.append((m.start(), tok, list(stack)))
            pos = m.end()
        return hits

    def is_guarded(stack):
        return any("!window.__DEMO__" in h for h in stack)

    def line(at):
        return js.count("\n", 0, at) + 1

    fetches = scan(r"fetch\(")
    assert fetches, "没找到任何 fetch,测试本身失效了"

    for at, _, stack in fetches:
        if is_guarded(stack):
            continue
        # 没被词法守卫的,只要它所在函数的每个调用点都被守卫,也算安全
        # (tick() 就是这种:fetch 在函数体里,setInterval/首次调用都在守卫内)
        fn = None
        for head in reversed(stack):
            m = re.search(r"function\s+(\w+)\s*\(", head)
            if m:
                fn = m.group(1)
                break
        assert fn, f"第 {line(at)} 行(js 内)的 fetch 既未被守卫,也不在具名函数里"

        sites = [(a, s) for a, tok, s in scan(rf"\b{fn}\(")
                 if not re.search(rf"function\s+{fn}\s*\($", js[:a + len(fn) + 1])]
        sites = [(a, s) for a, s in sites
                 if not js[:a].rstrip().endswith("function")]
        assert sites, f"{fn}() 里有 fetch 却找不到调用点"
        for a, s in sites:
            assert is_guarded(s), (
                f"{fn}() 内有 fetch,但第 {line(a)} 行(js 内)的调用点未被守卫")


def test_ui_javascript_parses():
    """ui.html 的脚本必须能被真正的 JS 解析器解析通过。

    起因是真出过一次:属性说明里写了 `'...Gray's Anatomy...'`——单引号串里
    的撇号提前闭合了字符串。整个 <script> 块解析失败,**页面全白**。

    这类错误 Python 一侧毫无感知:文件读得出、CSS 结构合法、160 项测试全过,
    但打开就是空白。所以必须交给真解析器,不能用正则糊弄。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("未安装 node,无法做 JS 语法检查")

    from heartsound import monitor
    src = (Path(monitor.__file__).parent / "ui.html").read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    assert blocks, "ui.html 里没找到 <script> 块,测试本身失效了"

    for i, code in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8",
                                         delete=False) as fh:
            fh.write(code)
            tmp = fh.name
        try:
            r = subprocess.run([node, "--check", tmp],
                               capture_output=True, text=True)
            assert r.returncode == 0, (
                f"ui.html 第 {i + 1} 个 <script> 块语法错误:\n{r.stderr.strip()}")
        finally:
            os.unlink(tmp)


def test_inline_event_handlers_dont_break_out_of_their_attribute():
    """内联 on* 属性里的引号必须与属性定界符区分开。

    onclick="pickSite('AV')" 是对的;若属性值里再出现同种引号,属性会提前
    结束,后面的代码被当成新属性名——浏览器同样不报错,只是行为诡异。
    """
    from heartsound import monitor
    src = (Path(monitor.__file__).parent / "ui.html").read_text(encoding="utf-8")

    found = 0
    for m in re.finditer(r'\son(\w+)="([^"]*)"', src):
        found += 1
        # 光看值里有没有引号是查不出来的:属性一旦提前闭合,正则就跟着切在
        # 错的地方,值本身反而"干净"。真正的判据是**闭合引号后面是什么**——
        # 合法的属性边界只有空白、`>` 或 `/>`。
        nxt = src[m.end():m.end() + 1]
        assert nxt == "" or nxt.isspace() or nxt in ">/", (
            f"on{m.group(1)} 的属性被提前闭合(其后是 {nxt!r}):"
            f" {src[m.start():m.end() + 20].strip()[:80]}")
    assert found, "没找到内联 on* 属性,测试本身失效了"


# --------------------------------------------------------------------------
# 采集开关
# --------------------------------------------------------------------------

def test_capture_is_off_until_someone_asks_for_it():
    """启动进程不等于打开麦克风。"""
    from heartsound.monitor import State
    st = State()
    assert st.capture is False
    assert st.get()["status"] == "idle"
    assert st.get()["capture"] is False
    assert State(capture=True).capture is True


def test_set_capture_transitions_and_keeps_the_last_frame():
    from heartsound.monitor import State
    st = State()
    st.set_capture(True)
    assert st.capture and st.t0 is not None
    assert st.get()["status"] == "waiting"

    st.set({"status": "ok", "bpm": 72.0, "wave": {"n": 3}})
    st.set_capture(False)
    d = st.get()
    assert d["status"] == "stopped", "停止后状态要变"
    # 停下来往往正是为了回看刚采到的东西,数据不能跟着清掉
    assert d["bpm"] == 72.0 and d["wave"] == {"n": 3}
    assert d["elapsed_s"] == 0.0 and st.t0 is None

    st.set_capture(False)          # 重复停止应当无副作用
    assert st.get()["bpm"] == 72.0


def test_elapsed_counts_only_while_capturing():
    from heartsound.monitor import State
    st = State()
    assert st.get()["elapsed_s"] == 0.0
    st.set_capture(True)
    st.t0 -= 65.0                  # 假装已经采了 65 秒
    assert 64.0 < st.get()["elapsed_s"] < 66.0
    st.set_capture(False)
    assert st.get()["elapsed_s"] == 0.0


class _FakeCap:
    """替身麦克风:记录开/关次数,好断言输入流真的被释放了。"""

    opened = 0
    closed = 0

    def __init__(self, device=None, window_s=6.0):
        self.fs, self.device_name, self.xruns = 8000, "fake", 0
        self._n = 0

    def start(self):
        _FakeCap.opened += 1
        return self

    def stop(self):
        _FakeCap.closed += 1

    def snapshot(self, min_new_s=0.0):
        self._n += 1
        if self._n < 2:
            return None
        rng = np.random.default_rng(0)
        return rng.normal(0, 0.01, 8000 * 4).astype(np.float32)


def test_capture_loop_opens_and_closes_the_device_with_the_switch(monkeypatch):
    """停止必须**关闭输入流**,而不是继续读着不分析。

    差别是实的:关掉之后系统的录音指示灯会灭,设备也让给别的程序。
    """
    from heartsound import monitor
    monkeypatch.setattr(monitor, "MicCapture", _FakeCap)
    _FakeCap.opened = _FakeCap.closed = 0

    st = monitor.State()
    t = threading.Thread(target=monitor.capture_loop, daemon=True,
                         args=(st, None, 4.0, 0.1, 0.30))
    t.start()
    try:
        time.sleep(0.3)
        assert _FakeCap.opened == 0, "没人按开始,不该打开设备"

        st.set_capture(True)
        deadline = time.time() + 5
        while _FakeCap.opened == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert _FakeCap.opened == 1, "按下开始后应打开设备"

        st.set_capture(False)
        deadline = time.time() + 5
        while _FakeCap.closed == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert _FakeCap.closed == 1, "按下停止后应关闭设备,而不是空转"

        st.set_capture(True)       # 再开一次应当是**新**的缓冲
        deadline = time.time() + 5
        while _FakeCap.opened < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert _FakeCap.opened == 2
    finally:
        st.stop = True
        st.set_capture(False)
        t.join(timeout=3)


def test_capture_loop_gives_up_after_a_device_error(monkeypatch):
    """打不开设备时必须落回停止态,不能每 50ms 重试一次。"""
    from heartsound import monitor

    class _Boom:
        def __init__(self, **kw):
            pass

        def start(self):
            _Boom.tries += 1
            raise OSError("no such device")

    _Boom.tries = 0
    monkeypatch.setattr(monitor, "MicCapture", _Boom)
    st = monitor.State()
    t = threading.Thread(target=monitor.capture_loop, daemon=True,
                         args=(st, None, 4.0, 0.1, 0.30))
    t.start()
    try:
        st.set_capture(True)
        time.sleep(0.6)
        assert _Boom.tries == 1, f"只该试一次,实际 {_Boom.tries} 次"
        assert st.capture is False
        d = st.get()
        assert d["status"] == "error" and "no such device" in d["message"]
    finally:
        st.stop = True
        t.join(timeout=3)


def _serve_state(st):
    """把 handler 挂到一个真实的临时端口上,返回 (端口, 关闭函数)。"""
    from heartsound.monitor import make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def close():
        srv.shutdown()
        srv.server_close()
        t.join(timeout=3)

    return srv.server_address[1], close


def _req(port, path, method="GET"):
    """直接用 http.client,不走 urllib。

    urllib 会读 http_proxy 等环境变量,把发往 127.0.0.1 的请求也交给代理,
    在有代理的环境里全部变成 502。
    """
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        c.request(method, path, body=b"" if method == "POST" else None)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def test_capture_switch_is_post_only():
    """开麦克风不能是 GET。

    GET 被当成幂等、可安全重放的操作:浏览器预取、预加载、扫描器,甚至
    一条聊天里贴出的链接被自动展开,都可能**在没人按按钮的情况下打开
    麦克风**。所以开关只接受 POST。
    """
    from heartsound.monitor import State
    st = State()
    port, close = _serve_state(st)
    try:
        code, _ = _req(port, "/api/capture?on=1")          # GET
        assert code == 404, f"GET 竟然被受理了(HTTP {code})"
        assert st.capture is False, "GET 不该改变采集状态"

        code, body = _req(port, "/api/capture?on=1", "POST")
        assert code == 200 and json.loads(body)["capture"] is True
        assert st.capture is True

        code, body = _req(port, "/api/capture?on=0", "POST")
        assert code == 200 and json.loads(body)["capture"] is False
        assert st.capture is False
    finally:
        close()


def test_capture_switch_parses_its_argument_strictly():
    """含糊的输入一律拒绝,绝不猜成"开"。

    第一版写的是 q.get("on", ["1"]) —— 参数缺失、或 `?on=` 空值,都会落到
    默认的 "1",**结果是把麦克风打开**。开麦克风不该有这种默认。
    """
    from heartsound.monitor import State
    st = State()
    port, close = _serve_state(st)
    try:
        for v in ("0", "false", "no", "OFF"):
            st.set_capture(True)
            assert _req(port, f"/api/capture?on={v}", "POST")[0] == 200
            assert st.capture is False, f"on={v!r} 应当理解为关"
        for v in ("1", "true", "yes", "ON"):
            st.set_capture(False)
            assert _req(port, f"/api/capture?on={v}", "POST")[0] == 200
            assert st.capture is True, f"on={v!r} 应当理解为开"

        for path in ("/api/capture", "/api/capture?on=", "/api/capture?on=maybe",
                     "/api/capture?x=1"):
            st.set_capture(False)
            code, body = _req(port, path, "POST")
            assert code == 400, f"{path} 应当被拒绝,实得 {code}"
            assert json.loads(body)["ok"] is False
            assert st.capture is False, f"{path} 竟然打开了麦克风"
    finally:
        close()


def test_state_endpoint_reports_the_switch():
    from heartsound.monitor import State
    st = State()
    port, close = _serve_state(st)
    try:
        d = json.loads(_req(port, "/api/state")[1])
        assert d["capture"] is False and d["status"] == "idle"
        _req(port, "/api/capture?on=1", "POST")
        d = json.loads(_req(port, "/api/state")[1])
        assert d["capture"] is True
    finally:
        close()


def test_unknown_post_paths_are_rejected():
    from heartsound.monitor import State
    port, close = _serve_state(State())
    try:
        assert _req(port, "/api/site?code=AV", "POST")[0] == 404
        assert _req(port, "/", "POST")[0] == 404
    finally:
        close()


def test_ui_has_start_and_stop_buttons_wired_to_setcapture():
    from heartsound import monitor
    src = (Path(monitor.__file__).parent / "ui.html").read_text(encoding="utf-8")
    assert 'id="btnGo"' in src and 'id="btnStop"' in src
    assert 'onclick="setCapture(true)"' in src
    assert 'onclick="setCapture(false)"' in src
    # 停止键初始必须是禁用的:还没开始就摆一个可点的"停止"是骗人的
    go = src[src.index('id="btnGo"'):src.index('id="btnStop"')]
    stop = src[src.index('id="btnStop"'):]
    assert "disabled" not in go, "开始键初始不该禁用"
    assert "disabled" in stop[:stop.index("</button>")], "停止键初始应禁用"
    assert 'method: "POST"' in src, "前端必须用 POST 调开关"
