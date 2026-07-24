"""checkmic 的诊断判据:每种故障只应报出对应的那一项。

开发中这套用例抓到两个假阳性:高通滤波和强工频都会压低分段响度的变异,
被误判成 AGC。修法是对工频门控 + 收紧阈值 + 把结论降级为"疑似"——因为
该判据本身区分不开(实测 CV:正常 0.065、高通 0.033、工频 0.020、
完美 AGC 0.000,区间重叠)。
"""

import numpy as np
import pytest
from scipy import signal as sg

from checkmic import diagnose

FS = 44100.0


def _base(n, rng):
    """模拟环境底噪:1/f 特性,低频更强(真实底噪如此)。"""
    w = rng.standard_normal(n)
    b, a = sg.butter(1, [5 / (FS / 2), 0.45], btype="band")
    pink = sg.lfilter(b, a, w)
    pink = pink / np.max(np.abs(pink)) * 0.05
    lb = sg.sosfilt(sg.butter(2, 80, "low", fs=FS, output="sos"), w)
    x = pink + lb / np.max(np.abs(lb)) * 0.15
    return x / (np.max(np.abs(x)) / 0.2)


def _cases():
    n = int(6 * FS)
    rng = np.random.default_rng(0)
    t = np.arange(n) / FS
    base = _base(n, rng)
    seg = int(0.5 * FS)
    agc = base.copy()
    for i in range(0, n - seg, seg):
        s = agc[i:i + seg]
        r = np.sqrt(np.mean(s ** 2))
        if r > 0:
            agc[i:i + seg] = s / r * 0.05
    return {
        "正常链路": (base, None),
        "高通砍低频": (sg.sosfilt(sg.butter(4, 120, "high", fs=FS,
                                            output="sos"), base), "低频响应"),
        "削波": (np.clip(base * 8, -1, 1), "输入电平"),
        "电平过低": (base * 0.0004, "输入电平"),
        "工频干扰": (base + 0.25 * np.sin(2 * np.pi * 50 * t), "工频干扰"),
        "AGC": (agc, "自动增益(AGC)"),
        "无信号": (np.zeros(n), "输入电平"),
    }


CASES = _cases()


@pytest.mark.parametrize("name", list(CASES))
def test_each_fault_flags_only_its_own_item(name):
    x, expect = CASES[name]
    res = diagnose(np.asarray(x, float), FS)
    flagged = {nm for lv, nm, _ in res if lv != "ok"}
    if expect is None:
        assert not flagged, f"正常链路不应报警,却报了 {flagged}"
    else:
        assert expect in flagged, f"{name} 应报出「{expect}」,实际 {flagged}"
        assert flagged == {expect}, f"{name} 只应报「{expect}」,实际 {flagged}"


def test_diagnose_returns_wellformed_result():
    x, _ = CASES["正常链路"]
    for lv, nm, msg in diagnose(np.asarray(x, float), FS):
        assert lv in ("ok", "warn", "bad")
        assert nm and msg
