"""
realtime.py — 从麦克风实时采集并周期性分析。

由 heartbeat.py 拆分而来,逐字保留。
"""

from __future__ import annotations

import threading

import numpy as np

from .analyzer import HeartSoundAnalyzer


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
