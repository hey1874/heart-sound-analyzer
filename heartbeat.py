"""
heartbeat.py — 向后兼容入口。

核心算法已拆分为 `heartsound` 包(见 heartsound/__init__.py):原来
`HeartSoundAnalyzer` 是一个 27 个方法、952 行的类,现按职责拆到四个 mixin。
**拆分只改代码组织,算法行为完全一致。**

新代码请直接用包:

    from heartsound import HeartSoundAnalyzer, RealtimeHeartRate

本文件保留是为了:
  1. 已有的 `from heartbeat import HeartSoundAnalyzer` 继续可用;
  2. `python heartbeat.py` / `python heartbeat.py --selftest` 继续可用
     (README 与 docs 里的示例都是这个写法)。

⚠️ 仅用于学习/演示,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

from heartsound import HeartSoundAnalyzer, RealtimeHeartRate, synthesize
from heartsound.dsp import _movavg

# 更早的版本以私有名 `_synthesize` 从本模块导入合成器
_synthesize = synthesize

__all__ = ["HeartSoundAnalyzer", "RealtimeHeartRate", "synthesize",
           "_synthesize", "_movavg"]


def main(argv: list[str]) -> int:
    from heartsound.cliutil import Parser
    p = Parser("心音实时采集与分析(向后兼容入口;新代码请用 heartsound 包)")
    p.add_argument("--selftest", action="store_true",
                   help="用合成心音跑 22 项自测,不需要麦克风;失败返回非零退出码")
    p.add_argument("--device", help="输入设备编号或名称(默认系统默认设备)")
    p.add_argument("--window", type=float, default=6.0, help="分析窗长度(秒)")
    p.add_argument("--hop", type=float, default=1.0, help="输出间隔(秒)")
    a = p.parse_args(argv)

    if a.selftest:
        # 自测已移到 selftest.py:原来的内联版本无论成败都返回退出码 0,
        # 无法接入 CI;现在按失败数返回非零。
        from selftest import main as selftest_main
        return selftest_main()

    dev = a.device
    if dev is not None and dev.isdigit():
        dev = int(dev)
    RealtimeHeartRate(device=dev, window_s=a.window, hop_s=a.hop).run()
    return 0


if __name__ == "__main__":
    from heartsound.cliutil import run
    run(main)
