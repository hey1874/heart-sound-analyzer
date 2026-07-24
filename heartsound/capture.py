"""
capture.py — 麦克风采集:索引式环形缓冲。

从 RealtimeHeartRate 抽出,供终端输出与 Web 界面共用一份实现。

实时音频的两条纪律,这里都遵守:
  1. **回调里不做分配**。写指针推进 + 只拷贝新到的那一块;此前用
     np.roll 整体搬移,每次回调都要新分配并拷贝整个窗口(44.1kHz/6s
     实测 313µs,占 512 样本回调预算的 2.7%)。
  2. **回调里不做 I/O**。溢出/欠载只累计计数,交由消费方报告。
"""

from __future__ import annotations

import threading

import numpy as np


class MicCapture:
    """把最近 window_s 秒的音频维持在一个环形缓冲里。

    用法::

        with MicCapture(device=2, window_s=6.0) as cap:
            while True:
                snap = cap.snapshot(min_new_s=1.0)   # 攒够新数据才返回
                if snap is not None:
                    ...
    """

    def __init__(self, device=None, samplerate: int | None = None,
                 window_s: float = 6.0) -> None:
        self.device = device
        self.samplerate = samplerate
        self.window_s = window_s
        self.fs: int = 0
        self.device_name: str = ""
        self.xruns = 0
        self._lock = threading.Lock()
        self._stream = None
        self._ring = None
        self._widx = 0
        self._filled = 0
        self._since = 0

    def __enter__(self) -> "MicCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> "MicCapture":
        import sounddevice as sd            # 延迟导入,核心算法不依赖它

        info = sd.query_devices(self.device, "input")
        self.device_name = info["name"]
        self.fs = int(self.samplerate or info["default_samplerate"])
        win_n = int(self.window_s * self.fs)
        self._ring = np.zeros(win_n, dtype=np.float32)
        self._widx = self._filled = self._since = 0

        def callback(indata, frames, time_info, status):
            if status:
                self.xruns += 1            # 回调内不做 I/O
            block = indata[:, 0]           # 取单声道
            if len(block) >= win_n:        # 块比窗口还长时只保留最新的一窗
                block = block[-win_n:]
            n = len(block)
            with self._lock:
                end = self._widx + n
                if end <= win_n:
                    self._ring[self._widx:end] = block
                else:                      # 跨越缓冲末端,分两段写
                    k = win_n - self._widx
                    self._ring[self._widx:] = block[:k]
                    self._ring[:end - win_n] = block[k:]
                self._widx = end % win_n
                self._filled = min(win_n, self._filled + n)
                self._since += n

        self._stream = sd.InputStream(
            device=self.device, channels=1, samplerate=self.fs,
            dtype="float32", callback=callback)
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def fill_ratio(self) -> float:
        """环形缓冲已填充的比例(0~1)。

        界面按下「开始采集」后要等满一个窗(默认 6 秒)才有第一帧结果。
        不把进度显示出来,这几秒看着就像没点上。
        """
        if self._ring is None or len(self._ring) == 0:
            return 0.0
        with self._lock:
            return min(1.0, self._filled / len(self._ring))

    def snapshot(self, min_new_s: float = 0.0) -> np.ndarray | None:
        """取按时间顺序排好的最近一窗。

        缓冲未填满、或自上次取用以来新数据不足 min_new_s 秒时返回 None。
        分配发生在调用方线程,不在音频回调里。
        """
        if self._ring is None:
            return None
        need = int(min_new_s * self.fs)
        win_n = len(self._ring)
        with self._lock:
            if self._filled < win_n or self._since < need:
                return None
            snap = np.concatenate((self._ring[self._widx:],
                                   self._ring[:self._widx]))
            self._since = 0
        return snap
