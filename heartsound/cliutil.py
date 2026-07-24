"""
cliutil.py — 命令行工具的公共部分。

为什么需要
----------
拆分前 6 个命令行工具各写一份 `main()`,参数解析全靠手写
`argv[argv.index("--x") + 1]`,全项目 17 处。这个模式在参数后面漏值时会直接
抛 IndexError——**实测 5 个工具都会崩**:

    calibrate.py x --out              IndexError
    external_eval.py x --cnn          IndexError
    external_eval.py x --per-subset   IndexError
    train_cnn.py x --epochs           IndexError
    train_cnn.py x --cache            IndexError

(predict.py 里同样的 bug 早先单独修过一次,但同样的写法又被复制到了后来
新增的工具里。这正是"同一个 bug 修 N 次"的典型——应该消灭模式,而不是逐个打补丁。)

改用 argparse 后这类错误由标准库统一处理:缺值报清晰的用法提示并返回退出码 2,
不再抛栈回溯。
"""

from __future__ import annotations

import argparse
import sys


def setup_console() -> None:
    """把 stdout/stderr 切到 UTF-8。

    Windows 控制台默认 GBK,中文与 emoji 会乱码。stderr 也要切——否则
    异常信息乱码,反而在出错时最难读。拆分前这段样板在 8 个文件里各有一份。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                          # noqa: BLE001
            pass


class Parser(argparse.ArgumentParser):
    """带医疗免责声明的 ArgumentParser。

    每个面向使用者的入口都应该带这句;放在这里保证不会漏。
    """

    DISCLAIMER = "⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。"

    def __init__(self, description: str, **kw):
        kw.setdefault("formatter_class", argparse.RawDescriptionHelpFormatter)
        kw.setdefault("epilog", self.DISCLAIMER)
        super().__init__(description=description, **kw)


def run(main_fn) -> None:
    """统一入口:设好编码,跑 main,把返回值当退出码。"""
    setup_console()
    sys.exit(main_fn(sys.argv[1:]))
