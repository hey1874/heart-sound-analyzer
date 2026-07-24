"""
monitor.py — 浏览器里的实时心音监视界面(上位机)。

    python monitor.py --demo           # 合成数据驱动,不需要麦克风
    python monitor.py                  # 默认输入设备,自动开浏览器
    python monitor.py --device 2 --port 8800
    python monitor.py --no-browser

实现见 heartsound/monitor.py。零新增依赖(标准库 http.server + 前端轮询)。

⚠️ 仅监听本机回环地址且未做鉴权,请勿暴露到公网。
⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

from heartsound.monitor import serve


def main(argv: list[str]) -> int:
    from heartsound.cliutil import Parser
    p = Parser("浏览器里的实时心音监视界面(波形/包络/S1-S2 分段/信号质量)")
    p.add_argument("--device", help="输入设备编号或名称")
    p.add_argument("--port", type=int, default=8765, help="本机监听端口")
    p.add_argument("--window", type=float, default=6.0, help="分析窗长度(秒)")
    p.add_argument("--hop", type=float, default=1.0, help="分析间隔(秒)")
    p.add_argument("--conf-gate", type=float, default=0.30,
                   help="置信度门限,低于此判信号不清")
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--demo", action="store_true",
                   help="用合成心音驱动界面,不需要麦克风(接线前先验界面)")
    a = p.parse_args(argv)

    dev = a.device
    if dev is not None and dev.isdigit():
        dev = int(dev)
    return serve(device=dev, port=a.port, window_s=a.window, hop_s=a.hop,
                 conf_gate=a.conf_gate, open_browser=not a.no_browser,
                 demo=a.demo)


if __name__ == "__main__":
    from heartsound.cliutil import run
    run(main)
