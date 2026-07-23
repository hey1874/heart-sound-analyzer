"""单元测试:selftest.py 的端到端检查覆盖不到的边角。"""

import numpy as np
import pytest

from classifier import FEATURE_NAMES, feature_vector, threshold_baseline
from heartbeat import HeartSoundAnalyzer, _movavg
from synth import synthesize

FS = 2000.0


# --- 工具函数 ---------------------------------------------------------------

def test_movavg_preserves_length_and_level():
    """边缘用端点延拓,不能像 np.convolve(mode='same') 那样把两端压低。"""
    v = np.ones(50)
    out = _movavg(v, 11)
    assert out.shape == v.shape
    assert np.allclose(out, 1.0, atol=1e-9), "常数信号平滑后仍应为常数"


def test_movavg_degenerate():
    assert _movavg(np.array([1.0]), 5).shape == (1,)
    assert _movavg(np.array([]), 5).shape == (0,)


@pytest.mark.parametrize("n", [0, 1])
def test_movavg_noop(n):
    v = np.arange(10.0)
    assert np.array_equal(_movavg(v, n), v)


# --- 削波检测 ---------------------------------------------------------------

def test_clip_fraction_detects_flat_top():
    x, _ = synthesize(bpm=72, secs=4, seed=1)
    x = x / np.max(np.abs(x))
    assert HeartSoundAnalyzer.clip_fraction(x) < 0.01
    assert HeartSoundAnalyzer.clip_fraction(np.clip(x * 3, -1, 1)) > 0.02


def test_clip_fraction_degenerate():
    assert HeartSoundAnalyzer.clip_fraction(np.array([])) == 0.0
    assert HeartSoundAnalyzer.clip_fraction(np.zeros(100)) == 0.0


def test_clip_must_be_measured_before_resampling():
    """回归:重采样的抗混叠滤波会磨圆削波平顶,门控因此在真实链路上失效。"""
    from scipy import signal
    x, _ = synthesize(bpm=72, secs=8, seed=2)
    x = x / np.max(np.abs(x))
    x44 = np.clip(signal.resample_poly(x, 22, 1) * 3.0, -1.0, 1.0)
    an = HeartSoundAnalyzer()
    assert an.clip_fraction(x44) > 0.02, "原始 44kHz 波形上应测得削波"
    assert an.clip_fraction(an.resample(x44, 44000.0)) < 0.02, \
        "重采样后削波被磨掉——正是不能在此处判定的原因"
    assert an.analyze(x44, 44000.0)["sqi"]["clip"] > 0.02, \
        "analyze 必须用原始波形判削波"


# --- 工频 -------------------------------------------------------------------

def test_hum_ratio_separates_mains_from_heart_sound():
    an = HeartSoundAnalyzer()
    clean, _ = synthesize(bpm=72, secs=12, seed=3)
    hummed, _ = synthesize(bpm=72, secs=12, seed=3, hum_hz=50.0, hum=1.0)
    assert an.analyze(clean, FS)["sqi"]["hum"] < 0.2
    assert an.analyze(hummed, FS)["sqi"]["hum"] > 0.6


def test_hum_does_not_inflate_quality():
    """回归:工频落在带内,曾使 band_ratio 升到 1.00 而 SQI 仍判 ok。"""
    an = HeartSoundAnalyzer()
    x, _ = synthesize(bpm=72, secs=12, seed=3, hum_hz=50.0, hum=1.0)
    q = an.analyze(x, FS)["sqi"]
    assert q["band_ratio"] > 0.9, "带内占比确实被工频抬高了"
    assert not q["ok"], "但综合判定必须拒绝"


# --- 特征向量 ---------------------------------------------------------------

def test_feature_vector_length_matches_names():
    x, _ = synthesize(bpm=72, secs=8, seed=4)
    v = feature_vector(HeartSoundAnalyzer().analyze(x, FS))
    assert v.shape == (len(FEATURE_NAMES),)


def test_feature_vector_handles_empty_result():
    v = feature_vector(HeartSoundAnalyzer().analyze(np.zeros(0), FS))
    assert v.shape == (len(FEATURE_NAMES),)
    assert np.isnan(v).any()


def test_murmur_shape_features_not_missing_for_normals():
    """回归:形状特征曾只在疑似杂音时才算,'是否缺失'成了标签的代理。"""
    an = HeartSoundAnalyzer()
    for seed in range(4):
        x, _ = synthesize(bpm=72, secs=12, seed=seed)
        mu = an.analyze(x, FS)["murmur"]
        assert mu["occupancy"] is not None
        assert mu["peak_pos"] is not None
        assert mu["flatness"] is not None


# --- 退化输入 ---------------------------------------------------------------

@pytest.mark.parametrize("x", [
    np.zeros(0), np.zeros(1), np.zeros(4000),
    np.full(4000, np.nan), np.ones(3),
])
def test_analyze_never_raises_on_degenerate_input(x):
    r = HeartSoundAnalyzer().analyze(x, FS)
    assert r["reliable"] is False
    assert set(["bpm", "sqi", "murmur", "reliable"]) <= set(r)


def test_short_input_reports_none_bpm_not_crash():
    """回归:predict.py 对 bpm=None 直接格式化会抛 TypeError。"""
    from predict import _fmt
    x, _ = synthesize(bpm=72, secs=1.0, seed=5)
    r = HeartSoundAnalyzer().analyze(x[:400], FS)
    assert r["bpm"] is None
    assert _fmt(r["bpm"], ".0f") == "—"


# --- 分类器 -----------------------------------------------------------------

def test_threshold_baseline_runs():
    X = np.array([[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)] * 4)
    y = np.array([0, 1] * 4)
    b = threshold_baseline(X, y)
    assert 0.0 <= b["acc"] <= 1.0


def test_predict_cli_missing_model_arg():
    """回归:`--model` 作为最后一个参数曾直接 IndexError。"""
    from predict import main
    assert main(["x.wav", "--model"]) == 1


# --- 标定 -------------------------------------------------------------------

def test_from_json_applies_operating_point():
    """calibrate.py 产出的阈值文件应能直接构造已标定的分析器。"""
    import os
    fx = os.path.join(os.path.dirname(__file__), "fixture_thresholds.json")
    base = HeartSoundAnalyzer()
    for op in ("youden", "sens80", "sens90"):
        an = HeartSoundAnalyzer.from_json(fx, operating_point=op)
        assert an.murmur_thr != base.murmur_thr, f"{op} 未生效"
        assert an.calibration["murmur"]["points"][op]["thr"] == an.murmur_thr
    # 敏感度要求越高 -> 阈值越低
    assert (HeartSoundAnalyzer.from_json(fx, "sens90").murmur_thr
            < HeartSoundAnalyzer.from_json(fx, "sens80").murmur_thr
            < HeartSoundAnalyzer.from_json(fx, "youden").murmur_thr)


def test_from_json_overrides_win():
    import os
    fx = os.path.join(os.path.dirname(__file__), "fixture_thresholds.json")
    an = HeartSoundAnalyzer.from_json(fx, "youden", murmur_thr=0.99)
    assert an.murmur_thr == 0.99


def test_calibrate_module_imports():
    """calibrate.py 应可被导入(不依赖数据集存在)。"""
    import calibrate
    assert callable(calibrate.reference_bpm)
    assert calibrate.reference_bpm([(0.0, 0.1, 1)]) is None
    segs = [(i * 0.8, i * 0.8 + 0.1, 1) for i in range(8)]
    assert abs(calibrate.reference_bpm(segs) - 75.0) < 0.1
