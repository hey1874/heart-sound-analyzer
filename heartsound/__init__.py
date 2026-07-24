"""
heartsound — 心音分析工具包。

公开 API
--------
    from heartsound import HeartSoundAnalyzer, RealtimeHeartRate, synthesize

    res = HeartSoundAnalyzer().analyze(x, fs)
    if res["reliable"]:
        print(res["bpm"], res["rhythm"], res["murmur"])

本包由原来的单文件 heartbeat.py 拆分而来。`HeartSoundAnalyzer` 曾是一个
27 个方法、952 行的类,现按职责拆到四个 mixin:

    dsp           滤波、包络、心率估计(与临床无关的信号处理)
    quality       削波、工频、SQI、可采信判定
    segmentation  S1/S2 分段与心动周期重建
    features      节律、STI、杂音、频谱轮廓、额外音、S2 分裂、HRV

拆分**只改代码组织**,方法体逐字保留,算法行为与拆分前完全一致
(22 项自测与 102 项 pytest 全部沿用,数值未变)。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from .analyzer import HeartSoundAnalyzer
from .dsp import _movavg
from .realtime import RealtimeHeartRate
from .synth import synthesize

__all__ = ["HeartSoundAnalyzer", "RealtimeHeartRate", "synthesize", "_movavg"]
__version__ = "0.2.0"
