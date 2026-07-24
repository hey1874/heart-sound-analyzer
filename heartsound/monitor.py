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


def _downsample(x: np.ndarray, n: int) -> list[float]:
    """给绘图用的降采样:分段取 (最小, 最大),保住波形的真实起伏。

    为什么不隔点抽样:那会漏掉尖峰,画出来的波形比实际**更干净**——对一个
    以"判断信号质量"为要务的界面来说,这种失真正好把该看见的问题藏起来。

    末尾补齐后再分段:直接截断会丢掉最后不足一段的样本,尖峰若落在末尾就
    看不见了。输出点数保证不超过 n。
    """
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return []
    if x.size <= n:
        return [round(float(v), 5) for v in x]
    half = max(1, n // 2)
    k = int(np.ceil(x.size / half))                # 每段样本数
    pad = (-x.size) % k
    if pad:                                        # 用末值补齐,不丢尾部
        x = np.concatenate([x, np.full(pad, x[-1])])
    seg = x.reshape(-1, k)
    out = np.empty(seg.shape[0] * 2)
    out[0::2] = seg.min(axis=1)
    out[1::2] = seg.max(axis=1)
    return [round(float(v), 5) for v in out]


class State:
    """最近一次分析结果的快照,供 HTTP 线程读取。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict = {"status": "starting"}
        self.stop = False

    def set(self, d: dict) -> None:
        with self.lock:
            self.data = d

    def get(self) -> dict:
        with self.lock:
            return self.data


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
        "wave": _downsample(wave, 1200) if wave is not None else [],
        "env": _downsample(env, 1200) if env is not None else [],
        "beats": ([round(float(v) / n_env, 5) for v in beats]
                  if beats is not None and n_env else []),
        "s1": s1, "s2": s2,
        "window_s": round(n_env / an.fs, 2) if n_env else 0,
    }


def capture_loop(state: State, device, window_s: float, hop_s: float,
                 conf_gate: float) -> None:
    an = HeartSoundAnalyzer(conf_thr=conf_gate)
    try:
        cap = MicCapture(device=device, window_s=window_s).start()
    except Exception as e:                              # noqa: BLE001
        state.set({"status": "error",
                   "message": f"打不开输入设备:{type(e).__name__}: {e}"})
        return
    state.set({"status": "waiting", "device": cap.device_name, "fs": cap.fs})
    try:
        while not state.stop:
            snap = cap.snapshot(min_new_s=hop_s)
            if snap is None:
                threading.Event().wait(0.1)
                continue
            try:
                res = an.analyze(snap, cap.fs)
            except Exception as e:                      # noqa: BLE001
                state.set({"status": "error",
                           "message": f"分析出错:{type(e).__name__}: {e}"})
                continue
            state.set(_build_payload(an, res, cap.fs, cap.device_name,
                                     cap.xruns))
    finally:
        cap.stop()


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                               # noqa: N802
            if self.path.startswith("/api/state"):
                body = json.dumps(state.get(), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
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

        def log_message(self, *a):                      # 静音访问日志
            pass

    return Handler


def serve(device=None, port: int = 8765, window_s: float = 6.0,
          hop_s: float = 1.0, conf_gate: float = 0.30,
          open_browser: bool = True) -> int:
    state = State()
    t = threading.Thread(target=capture_loop, daemon=True,
                         args=(state, device, window_s, hop_s, conf_gate))
    t.start()

    # 只绑 127.0.0.1:这是本机监视工具,没有任何鉴权,不应暴露到网络上
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    url = f"http://127.0.0.1:{port}/"
    print(f"🎙  监视界面: {url}")
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
