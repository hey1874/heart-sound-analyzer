"""监视界面:用合成信号直接驱动,不依赖真麦克风。

重点验证两条设计约定,它们比"能显示"更重要:
  1. 信号质量不合格时,`reliable=False` 且必须给出可操作的原因;
  2. 传给前端的负载必须是可 JSON 序列化的纯量/数组——numpy 类型混进去会让
     整个接口 500,而这类错误只在真机上跑才会暴露。
"""

import json

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
    out = _downsample(x, 1200)
    assert len(out) <= 1200, f"输出 {len(out)} 点,超过上限 1200"
    assert max(out) >= 4.9, f"尖峰在 {spike_at} 处被抹掉了"
    assert min(out) <= -3.9, f"负向尖峰在 {spike_at} 附近被抹掉了"


def test_downsample_short_input_passthrough():
    x = np.arange(10.0)
    assert _downsample(x, 1200) == [float(v) for v in x]
