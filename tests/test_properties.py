"""性质检查:不针对具体数值,而是断言"任何输入下都必须成立"的契约。

这类检查的价值在于覆盖到手写用例想不到的组合。开发本套件时它实际发现了 4 个
缺陷:含 NaN/inf 的输入会静默产出全 NaN 结果、`clip_fraction` 在短输入上必然
假阳性、工频门限过严导致好录音被拒、真杂音被 band_ratio 判据挡掉。
"""

import warnings

import numpy as np
import pytest
from scipy import signal as sg

from classifier import FEATURE_NAMES, feature_vector
from heartbeat import HeartSoundAnalyzer
from synth import synthesize

FS = 2000.0
AN = HeartSoundAnalyzer()


def _degenerate_inputs():
    good = synthesize(secs=6, seed=1)[0]
    return {
        "空": np.zeros(0),
        "单点": np.zeros(1),
        "两点": np.array([1.0, -1.0]),
        "全零": np.zeros(4000),
        "全NaN": np.full(4000, np.nan),
        "含NaN": np.r_[good[:7000], np.full(1000, np.nan)],
        "含inf": np.r_[good[:7000], np.full(1000, np.inf)],
        "常数": np.full(8000, 3.7),
        "极小幅度": good * 1e-12,
        "极大幅度": good * 1e9,
        "单脉冲": np.r_[np.zeros(4000), [1.0], np.zeros(4000)],
        "方波": np.sign(np.sin(2 * np.pi * 1.2 * np.arange(24000) / FS)),
        "直流偏置": good + 100.0,
    }


@pytest.mark.parametrize("name,x", list(_degenerate_inputs().items()),
                         ids=list(_degenerate_inputs()))
def test_degenerate_input_contract(name, x):
    """退化输入:不得抛异常、不得发 RuntimeWarning、不得产出非有限数值。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = AN.analyze(x, FS)

    assert {"bpm", "confidence", "reliable", "sqi", "murmur"} <= set(r)
    assert r["bpm"] is None or np.isfinite(r["bpm"])
    assert np.isfinite(r["confidence"])
    if r["sqi"] is not None:
        for k in ("sqi", "hum", "clip", "band_ratio"):
            assert np.isfinite(r["sqi"][k]), f"sqi[{k}] 非有限"
        assert 0.0 <= r["sqi"]["hum"] <= 1.0
        assert 0.0 <= r["sqi"]["clip"] <= 1.0
    v = feature_vector(r)
    assert v.shape == (len(FEATURE_NAMES),)
    assert not np.isinf(v).any(), "特征向量不得含 inf(NaN 表示缺失是允许的)"


@pytest.mark.parametrize("fs_in,up,dn", [
    (2000, 1, 1), (4000, 2, 1), (8000, 4, 1),
    (44100, 441, 20), (48000, 24, 1), (1000, 1, 2),
])
def test_result_invariant_to_input_samplerate(fs_in, up, dn):
    """同一段心音以不同采样率喂入,心率结论必须一致。"""
    x0, _ = synthesize(bpm=72, secs=12, seed=3)
    x = x0 if up == dn == 1 else sg.resample_poly(x0, up, dn)
    r = AN.analyze(x, float(fs_in))
    assert r["bpm"] is not None and abs(r["bpm"] - 72) < 2


@pytest.mark.parametrize("hum", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("noise", [0.03, 0.15])
def test_gate_is_self_consistent(noise, hum):
    """reliable=True 时各分量必须都合格;reliable=False 时必须给出原因。"""
    x, fs = synthesize(bpm=72, secs=12, noise=noise, seed=5,
                       hum_hz=50.0, hum=hum)
    r = AN.analyze(x, fs)
    q = r["sqi"]
    if r["reliable"]:
        assert q["clip"] < 0.02
        assert q["hum"] < AN.hum_thr
        assert r["confidence"] >= AN.conf_thr
        assert q["ok"]
        assert r["unreliable_reason"] is None
    else:
        assert r["unreliable_reason"], "拒绝时必须说明原因"


@pytest.mark.parametrize("bpm", [42, 55, 72, 95, 130, 165, 178])
def test_bpm_no_octave_error(bpm):
    """全量程不得出现倍频/半频错误,也不得锁到舒张间期竞争峰。"""
    x, fs = synthesize(bpm=float(bpm), secs=10, seed=bpm)
    r = AN.analyze(x, fs, screen=False)
    assert r["bpm"] is not None
    assert abs(r["bpm"] - bpm) < 3, f"{bpm} -> {r['bpm']}"


@pytest.mark.parametrize("amp", [0.3, 0.5, 0.8, 1.2, 1.6, 2.0])
@pytest.mark.parametrize("phase", ["systolic", "diastolic"])
def test_true_murmur_passes_the_gate(amp, phase):
    """真杂音不得被信号质量门控挡掉,且时相归属不得判反。

    两条契约各自对应一个曾经的缺陷:
    - band_ratio 硬拒绝把真杂音挡在门外(杂音能量本就落在 20-200Hz 带外);
    - 分段用主带包络时,强杂音污染 HSMM 发射模型导致时相判反。
    """
    x, fs = synthesize(bpm=72, secs=12, murmur=amp, noise=0.06,
                       murmur_phase=phase, seed=7)
    r = AN.analyze(x, fs)
    assert r["reliable"], r["unreliable_reason"]
    assert not r["murmur"]["noise_suspect"], "真杂音被误判为噪声"
    key = "systolic" if phase == "systolic" else "diastolic"
    other = "diastolic" if phase == "systolic" else "systolic"
    assert r["murmur"][key] > r["murmur"][other], "杂音时相归属判反"


@pytest.mark.parametrize("n", [20, 50, 100, 500, 2000])
def test_clip_fraction_no_false_positive_on_short_input(n):
    """clip 是相对度量,天然下限 1/N;短输入不得因此被判成削波。"""
    x, _ = synthesize(secs=6, seed=2)
    assert HeartSoundAnalyzer.clip_fraction(x[:n]) < 0.02
