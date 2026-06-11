"""
predict.py — 加载训练好的模型,对录音/麦克风做"正常 vs 异常"预测。

用法:
  python predict.py recording.wav                  # 分类一个音频文件
  python predict.py recording.wav --model model_physionet.joblib
  python predict.py --selftest                     # 合成杂音信号自检

模型来源(见 classifier.py):
  model_demo.joblib       合成数据训练 —— ⚠️ 仅验证流程,勿信真实结果
  model_physionet.joblib  真实 PhysioNet 数据训练 —— 可用(仍非诊断)

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import os
import sys

import numpy as np

from classifier import HeartSoundClassifier, FEATURE_NAMES, feature_vector
from heartbeat import HeartSoundAnalyzer


def classify(x: np.ndarray, fs: float, model_path: str) -> dict:
    """对一段波形做预测,返回概率与关键特征。"""
    clf = HeartSoundClassifier.load(model_path)
    an = HeartSoundAnalyzer()
    res = an.analyze(x, fs)
    feat = feature_vector(res)
    prob = float(clf.predict_proba([feat])[0])     # 异常概率
    return {"abnormal_prob": prob, "result": res, "features": feat}


def _print_report(out: dict, model_path: str):
    p = out["abnormal_prob"]
    res = out["result"]
    demo = "demo" in os.path.basename(model_path).lower()
    if demo:
        print("⚠️  DEMO 模型(合成数据训练):以下结果仅演示流程,切勿当真。\n")
    label = "异常" if p >= 0.5 else "正常"
    bar = "█" * int(round(p * 20)) + "·" * (20 - int(round(p * 20)))
    print(f"判定: {label}   异常概率 [{bar}] {p:.2f}")
    # 附带可解释的临床线索
    print(f"心率 {res['bpm']:.0f} BPM (置信度 {res['confidence']:.2f}) | "
          f"SQI {res['sqi']['sqi']:.2f}")
    rh = res.get("rhythm") or {}
    if rh.get("classification"):
        print(f"节律: {rh['classification']}")
    mu = res.get("murmur") or {}
    if mu.get("flag") not in (None, "数据不足", "未见明显杂音"):
        extra = f"({mu['shape']}/{mu['timing']})" if mu.get("shape") else ""
        print(f"杂音: {mu['flag']} {extra}")
    ex = res.get("extra_sounds") or {}
    if ex.get("flag") not in (None, "数据不足", "未见明显额外音"):
        print(f"额外音: {ex['flag']}")
    print("\n(本工具非诊断,如有不适请就医)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = sys.argv[1:]
    model = "model_demo.joblib"
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        args = args[:i] + args[i + 2:]

    if not os.path.exists(model):
        print(f"找不到模型 {model};先运行 python classifier.py --train-demo")
        sys.exit(1)

    if "--selftest" in args:
        from heartbeat import _synthesize
        x, fs = _synthesize(bpm=72, secs=12, murmur=0.5, murmur_shape="diamond",
                            seed=42)
        _print_report(classify(x, fs, model), model)
    elif args:
        import soundfile as sf
        sig, fs = sf.read(args[0])
        if sig.ndim > 1:
            sig = sig[:, 0]
        _print_report(classify(sig, fs, model), model)
    else:
        print(__doc__)
