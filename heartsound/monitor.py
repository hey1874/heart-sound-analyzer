"""
monitor.py — 浏览器里的实时监视界面(上位机)。

为什么要有
----------
终端只有一行数字。贴听诊器的时候你需要**看见**:波形长什么样、包络上 S1/S2
标在哪、能量落在哪个频段、信号质量为什么不合格。看不见就无从判断是位置不对、
贴得不稳、还是链路本身有问题。

设计前提:**不能让不可靠的判定看起来权威**
-------------------------------------------
给这个项目做界面,最大的风险是漂亮的图表让人相信本不该相信的数字。所以:

  · 信号质量放在最显眼处,不合格时整块判定区**变灰并写明原因**,不显示结论;
  · 杂音 / S3 / S4 这些**阈值未经你的设备标定**的项,一律带"未标定"角标;
  · 心率、分段、间期这些真实数据验证过的量才用正常样式显示。

实现取舍
--------
用标准库 http.server + 前端轮询,**零新增依赖**——与"核心只要 numpy+scipy"
的定位一致。分析本来就是每秒一次(hop_s=1.0),轮询完全够用,不需要 WebSocket。
音频采集在后台线程,HTTP 只读最近一次结果的快照。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .analyzer import HeartSoundAnalyzer
from .capture import MicCapture

PAGE = None            # 延迟加载 ui.html


def _page() -> bytes:
    global PAGE
    if PAGE is None:
        import os
        p = os.path.join(os.path.dirname(__file__), "ui.html")
        with open(p, "rb") as fh:
            PAGE = fh.read()
    return PAGE


def _spectrogram(xr: np.ndarray, fs: float, n_frames: int = 220) -> dict:
    """给界面用的对数梅尔频谱图,量化成 0-255 的整数。

    为什么用频谱图而不是原始波形当主角:心音是**脉冲信号**——实测峰峰值
    中位数只有最大值的 5.5%,仅 4.9% 的列超过最大值一半。极值填充带画出来
    必然是"一条近乎平直的线 + 几根孤立尖峰",稀疏破碎。音频编辑器里波形
    好看是因为音乐/语音能量连续,心音不是。

    而频谱图对这个领域信息量大得多:**杂音就是收缩期里的高频能量带,在
    频谱图上直接看得见**。这也正是 CNN 吃的表示(见 cnn.logmel)。

    量化到 0-255 是为了控制负载:64×220 的浮点数组约 56KB,量化后 14KB。
    """
    from .cnn import logmel

    if xr.size < 512:
        return {"w": 0, "h": 0, "data": []}
    m = logmel(xr)                       # (n_mels, frames),已是对数域
    if m.shape[1] > n_frames:
        # 分箱必须**覆盖整条时间轴**。早先写成 m[:, :k*n_frames](k 为整除
        # 商)会把尾部余数直接截掉:8 秒窗只剩 89% 的时间轴、12 秒窗只剩
        # 59%,却仍被铺满整个画布宽度并标上秒刻度——频谱图与包络图的
        # S1/S2 因此对不上。改用覆盖全轴的边界索引。
        edge = np.linspace(0, m.shape[1], n_frames + 1).astype(int)
        m = np.stack([m[:, edge[i]:max(edge[i] + 1, edge[i + 1])].max(axis=1)
                      for i in range(n_frames)], axis=1)
    lo = float(np.percentile(m, 5))      # 5% 分位当底,拉开对比度
    hi = float(m.max())
    if hi - lo < 1e-9:
        return {"w": 0, "h": 0, "data": []}
    q = np.clip((m - lo) / (hi - lo), 0, 1)
    return {"w": int(m.shape[1]), "h": int(m.shape[0]),
            "data": (q * 255).astype(np.uint8).T.ravel().tolist()}


def _downsample(x: np.ndarray, n_cols: int = 1000) -> dict:
    """给绘图用的降采样:每列给出 (最小, 最大, RMS)。

    为什么要三个量而不是两个
    ------------------------
    心音是**脉冲信号**:实测峰峰值中位数只有最大值的 5.5%,只有 4.9% 的列
    超过最大值一半。只画 min/max 包络,画出来就是"一条近乎平直的线 + 几根
    孤立尖峰",稀疏破碎。

    专业音频软件的画法是**双层**:外层浅色画峰值包络,内层深色画 RMS。
    RMS 把中间填实,波形有了体量感,而且它表达的是**真实能量**,不是把
    振幅拉伸的视觉作弊。

    归一化用 99.5 分位而非绝对最大值:最大值被 S1 那一根尖峰独占,拿它做
    基准会把其余部分全压扁(实测列高中位从 8.9px 掉到 6.5px)。

    为什么不隔点抽样:那会漏掉尖峰,画出来比实际**更干净**——对一个以判断
    信号质量为要务的界面,这种失真正好把该看见的问题藏起来。
    """
    x = np.asarray(x, float).ravel()
    if x.size < 4:
        return {"n": 0, "lo": [], "hi": [], "rms": [], "ref": 1.0}
    n_cols = max(1, min(n_cols, x.size))
    k = int(np.ceil(x.size / n_cols))
    pad = (-x.size) % k
    if pad:                                        # 用末值补齐,不丢尾部
        x = np.concatenate([x, np.full(pad, x[-1])])
    seg = x.reshape(-1, k)
    ref = float(np.percentile(np.abs(x), 99.5)) or float(np.max(np.abs(x))) or 1.0
    r3 = lambda a: [round(float(v), 4) for v in a]      # noqa: E731
    return {"n": int(seg.shape[0]), "ref": round(ref, 6),
            "lo": r3(seg.min(axis=1)), "hi": r3(seg.max(axis=1)),
            "rms": r3(np.sqrt((seg ** 2).mean(axis=1)))}


class State:
    """最近一次分析结果的快照,供 HTTP 线程读取。"""

    def __init__(self, capture: bool = False) -> None:
        self.lock = threading.Lock()
        self.data: dict = {"status": "idle"}
        self.stop = False
        # 是否正在采集。默认**关**:启动进程不等于打开麦克风,得有人明确按
        # 下「开始采集」。停止时采集线程会真的关闭输入流,系统的录音指示灯
        # 随之熄灭,设备也交还给别的程序——只是"不分析"做不到这一点。
        self.capture = bool(capture)
        self.t0: float | None = time.monotonic() if capture else None
        # 每次「开始」自增。采集线程只看 capture 是不够的:若在它忙于
        # snapshot/analyze 期间发生 停->开,再回来看到的仍是 True,设备便
        # 不会重开,**停止前的旧音频会留在环形缓冲里进入第一窗**。代次变了
        # 就重开,这个洞才堵上。
        self.gen = 0
        # 当前听诊区。杂音是位置依赖的(多区取最大可把患者级 AUC 从 0.701
        # 提到 0.762),而且 AV/PV/TV/MV 正是 CirCor 的标签——标上位置,
        # 采到的数据才能直接喂给 calibrate.py。
        self.site: str = "MV"
        self.best: dict[str, float] = {}      # 各区见过的最好 SQI

    def set(self, d: dict) -> None:
        with self.lock:
            self.data = d

    def get(self) -> dict:
        with self.lock:
            d = dict(self.data)
            d["site"] = self.site
            d["best_sqi"] = dict(self.best)
            d["capture"] = self.capture
            d["elapsed_s"] = (round(time.monotonic() - self.t0, 1)
                              if self.t0 is not None else 0.0)
            return d

    def set_capture(self, on: bool) -> dict:
        """开/关采集。状态就地翻转,不必等采集线程转到下一圈。

        前端按下按钮后必须立刻看到变化,否则会以为没点上又点一次。
        """
        on = bool(on)
        with self.lock:
            was, self.capture = self.capture, on
            if on and not was:
                self.t0 = time.monotonic()
                self.gen += 1
                self.data = {"status": "waiting"}
            elif not on and was:
                self.t0 = None
                # 保留最后一帧画面,只把状态改掉:停下来正是要回看刚采到的东西
                self.data = dict(self.data, status="stopped")
            return {"ok": True, "capture": self.capture}

    def set_site(self, code: str) -> bool:
        from .sites import BY_CODE
        if code not in BY_CODE:
            return False
        with self.lock:
            self.site = code
        return True

    def note_quality(self, sqi: float | None) -> None:
        """记下当前听诊区见过的最好信号质量,便于"挪一挪找位置"。"""
        if sqi is None:
            return
        with self.lock:
            if sqi > self.best.get(self.site, 0.0):
                self.best[self.site] = round(float(sqi), 3)


def _build_payload(an: HeartSoundAnalyzer, res: dict, fs: int,
                   device: str, xruns: int) -> dict:
    """把 analyze() 的结果压成前端要的样子。

    只挑绘图与展示需要的量,并且**明确区分**「已验证」与「未标定」两类,
    前端据此决定用什么样式显示。
    """
    sqi = res.get("sqi") or {}
    rh = res.get("rhythm") or {}
    sti = res.get("sti") or {}
    mu = res.get("murmur") or {}
    ex = res.get("extra_sounds") or {}
    sp = res.get("s2_split") or {}

    env = res.get("env")
    env_seg = res.get("env_seg")
    beats = res.get("beats")
    wave = res.get("filtered")
    n_env = len(env) if env is not None else 0

    # S1/S2 位置:用与生产路径相同的窄带包络重算,保证画出来的就是实际用的
    s1 = s2 = []
    if env_seg is not None and len(env_seg) and res.get("bpm"):
        a, b = an.segment_hsmm(env_seg, res["bpm"])
        if len(a) < 3:
            a, b = an.segment_s1s2(an.detect_beats(env_seg))
        s1 = [round(float(v) / n_env, 5) for v in a] if n_env else []
        s2 = [round(float(v) / n_env, 5) for v in b] if n_env else []

    return {
        "status": "ok",
        "device": device, "fs": fs, "xruns": xruns,
        # --- 已验证:可以正常显示 ---
        "reliable": bool(res.get("reliable")),
        "reason": res.get("unreliable_reason"),
        "bpm": res.get("bpm"),
        "confidence": round(float(res.get("confidence") or 0), 3),
        "n_beats": res.get("n_beats"),
        "sqi": {k: (round(float(v), 3) if isinstance(v, (int, float)) else v)
                for k, v in sqi.items()},
        "systole_ms": sti.get("systole_ms"),
        "diastole_ms": sti.get("diastole_ms"),
        "sys_dia_ratio": sti.get("sys_dia_ratio"),
        # --- 未标定:前端会打角标 ---
        "rhythm": rh.get("classification"),
        "cv_robust": rh.get("cv_robust"),
        "murmur_flag": mu.get("flag"),
        "murmur_sys": mu.get("systolic"),
        "murmur_dia": mu.get("diastolic"),
        "murmur_asym": mu.get("asymmetry"),
        "extra_flag": ex.get("flag"),
        "split_flag": sp.get("flag"),
        # --- 绘图数据 ---
        "wave": _downsample(wave, 1000) if wave is not None else
                {"n": 0, "lo": [], "hi": [], "rms": [], "ref": 1.0},
        "spec": _spectrogram(res.get("resampled"), an.fs)
                if res.get("resampled") is not None else {"w": 0, "h": 0, "data": []},
        "env": _downsample(env, 900) if env is not None else
               {"n": 0, "lo": [], "hi": [], "rms": [], "ref": 1.0},
        "beats": ([round(float(v) / n_env, 5) for v in beats]
                  if beats is not None and n_env else []),
        "s1": s1, "s2": s2,
        "window_s": round(n_env / an.fs, 2) if n_env else 0,
    }


def capture_loop(state: State, device, window_s: float, hop_s: float,
                 conf_gate: float) -> None:
    """采集线程:按 state.capture 开合输入流。

    停止时**关闭** MicCapture,而不是继续读着但不分析——两点差别是实的:
    录音指示灯会灭,设备也交还给别的程序。重新开始时环形缓冲是新的,不会
    把停止之前的旧音频算进第一窗。
    """
    an = HeartSoundAnalyzer(conf_thr=conf_gate)
    cap = None
    gen = -1
    try:
        while not state.stop:
            try:
                if not state.capture:
                    if cap is not None:
                        cap.stop()
                        cap = None
                    threading.Event().wait(0.05)
                    continue

                if cap is None or gen != state.gen:
                    if cap is not None:      # 期间发生过 停->开,重开拿新缓冲
                        cap.stop()
                        cap = None
                    # 先记代次:开设备期间若又被翻,下一圈还会重开,不会漏掉
                    gen = state.gen
                    try:
                        cap = MicCapture(device=device,
                                         window_s=window_s).start()
                    except Exception as e:              # noqa: BLE001
                        # 先落回停止态,免得每 50ms 重试一次;顺序反了会被
                        # set_capture 覆盖掉错误信息。
                        state.set_capture(False)
                        state.set({"status": "error",
                                   "message": f"打不开输入设备:"
                                              f"{type(e).__name__}: {e}"})
                        continue
                    state.set({"status": "waiting", "device": cap.device_name,
                               "fs": cap.fs})

                snap = cap.snapshot(min_new_s=hop_s)
                if snap is None:
                    # 缓冲未满时报进度:否则按下开始后的头几秒界面毫无反应,
                    # 看着就像没点上。已经出过结果就不要再退回"缓冲中"——
                    # 那样界面会来回跳。
                    if state.get().get("status") in ("waiting", "buffering"):
                        state.set({"status": "buffering",
                                   "device": cap.device_name, "fs": cap.fs,
                                   "fill": round(cap.fill_ratio, 3)})
                    threading.Event().wait(0.1)
                    continue

                try:
                    res = an.analyze(snap, cap.fs)
                except Exception as e:                  # noqa: BLE001
                    state.set({"status": "error",
                               "message": f"分析出错:{type(e).__name__}: {e}"})
                    continue

                # 分析期间被叫停(或停了又开)——这一帧属于上一轮,丢掉
                if not state.capture or gen != state.gen:
                    continue
                state.note_quality((res.get("sqi") or {}).get("sqi"))
                state.set(_build_payload(an, res, cap.fs, cap.device_name,
                                         cap.xruns))

            except Exception as e:                      # noqa: BLE001
                # 兜底。原先没有:循环里冒出任何意外异常,线程就**悄悄死掉**,
                # 而界面会一直停在"缓冲中"——最坏的一种失败,看上去像还在
                # 工作。现在一律落回停止态并把原因显示出来。
                if cap is not None:
                    try:
                        cap.stop()
                    except Exception:                   # noqa: BLE001
                        pass
                    cap = None
                state.set_capture(False)
                state.set({"status": "error",
                           "message": f"采集线程出错:{type(e).__name__}: {e}"})
    finally:
        if cap is not None:
            cap.stop()


DEMO_SCENES = [
    ("正常", dict(bpm=68, secs=8, noise=0.05, seed=11)),
    ("收缩期杂音", dict(bpm=72, secs=8, murmur=0.5, murmur_shape="diamond",
                        noise=0.05, seed=2)),
    ("环境噪声偏大", dict(bpm=74, secs=8, noise=0.28, seed=23)),
    ("50Hz 工频干扰", dict(bpm=72, secs=8, hum_hz=50.0, hum=3.0, seed=5)),
    ("削波/过载", dict(bpm=70, secs=8, noise=0.05, seed=31)),
]


def demo_loop(state: State, hop_s: float, conf_gate: float) -> None:
    """用合成心音驱动界面,不需要麦克风。

    存在的理由不只是"演示":没有硬件时也能确认界面、门控、绘图都正常,
    接线之前就能排除软件侧的问题。轮播几种典型状态,好让被拒绝的那几种
    (工频、削波、噪声大)也能被看到——它们恰恰是最需要界面说清楚的情况。
    """
    from .synth import synthesize

    an = HeartSoundAnalyzer(conf_thr=conf_gate)
    i = 0
    while not state.stop:
        if not state.capture:
            time.sleep(0.05)
            continue
        name, kw = DEMO_SCENES[i % len(DEMO_SCENES)]
        x, fs = synthesize(**kw)
        if name.startswith("削波"):
            x = np.clip(x / (np.max(np.abs(x)) or 1) * 3.0, -1.0, 1.0)
        payload = _build_payload(an, an.analyze(x, fs), int(fs),
                                 f"演示模式 · {name}", 0)
        if not state.capture:
            continue
        state.set(payload)
        i += 1
        # 每种状态停留几拍,便于看清;Ctrl+C 能及时退出
        for _ in range(int(max(1, 4 / max(hop_s, 0.1)))):
            if state.stop or not state.capture:
                break
            time.sleep(hop_s)


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                               # noqa: N802
            if self.path.startswith("/api/sites"):
                from .sites import FIND_TIP, ORDER, SITES, VIEW_NOTE
                body = json.dumps({"sites": SITES, "order": ORDER,
                                   "tip": FIND_TIP, "view_note": VIEW_NOTE},
                                  ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            elif self.path.startswith("/api/site?"):
                from urllib.parse import parse_qs, urlparse
                code = parse_qs(urlparse(self.path).query).get("code", [""])[0]
                okc = state.set_site(code)
                body = json.dumps({"ok": okc, "site": state.site}).encode()
                self.send_response(200 if okc else 400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            elif self.path.startswith("/api/state"):
                body = json.dumps(state.get(), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
            elif self.path.startswith("/thorax.png"):
                import os
                with open(os.path.join(os.path.dirname(__file__),
                                       "thorax.png"), "rb") as fh:
                    body = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "max-age=86400")
            elif self.path in ("/", "/index.html"):
                body = _page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):                              # noqa: N802
            """采集开关只接受 POST。

            用 GET 会有实际风险:浏览器的预取/预加载、以及任何"访问一下这个
            URL"的行为都可能**在没人按按钮的情况下打开麦克风**。改变状态的
            操作不该是幂等 GET。
            """
            if not self.path.startswith("/api/capture"):
                self.send_response(404)
                self.end_headers()
                return
            from urllib.parse import parse_qs, urlparse
            # 严格解析,不给默认值。原先写成 q.get("on", ["1"]) —— 参数缺失
            # 或写成 `?on=` 都会落到默认的 "1",**结果是把麦克风打开**。
            # 开麦克风这种操作,含糊的输入应当拒绝,而不是猜一个"开"。
            q = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            raw = (q.get("on") or [""])[0].strip().lower()
            truthy, falsy = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}
            if raw not in truthy | falsy:
                body = json.dumps({"ok": False,
                                   "error": f"on 参数无效:{raw!r}"}).encode()
                self.send_response(400)
                self.send_header("Content-Type",
                                 "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(state.set_capture(raw in truthy)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):                      # 静音访问日志
            pass

    return Handler


def serve(device=None, port: int = 8765, window_s: float = 6.0,
          hop_s: float = 1.0, conf_gate: float = 0.30,
          open_browser: bool = True, demo: bool = False,
          auto_start: bool = False) -> int:
    # 演示模式没有麦克风可保护,直接开跑;真采集默认等人按「开始采集」
    state = State(capture=auto_start or demo)
    if demo:
        t = threading.Thread(target=demo_loop, daemon=True,
                             args=(state, hop_s, conf_gate))
    else:
        t = threading.Thread(target=capture_loop, daemon=True,
                             args=(state, device, window_s, hop_s, conf_gate))
    t.start()

    # 只绑 127.0.0.1:这是本机监视工具,没有任何鉴权,不应暴露到网络上
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    url = f"http://127.0.0.1:{port}/"
    print(f"🎙  监视界面: {url}")
    if demo:
        print("   演示模式:用合成心音驱动,不需要麦克风。轮播 "
              f"{len(DEMO_SCENES)} 种典型状态。")
    else:
        if state.capture:
            print("   已自动开始采集(--auto-start)。")
        else:
            print("   麦克风尚未打开——在页面上按「开始采集」才会打开。")
        print("   把听诊器贴稳(胸骨左缘第 4 肋间 或 心尖),Ctrl+C 退出。")
    print("   仅监听本机回环地址,未做鉴权,请勿暴露到公网。")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        state.stop = True
        srv.server_close()
    return 0
