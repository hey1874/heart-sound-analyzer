"""
predict.py — 加载训练好的模型,对录音做"正常 vs 异常"预测。

用法:
  python predict.py recording.wav                  # 分类一个音频文件
  python predict.py recording.wav --model model_physionet.joblib
  python predict.py recording.wav --force          # 信号质量不合格也强行给判定
  python predict.py --selftest                     # 合成杂音信号自检

模型来源(见 classifier.py):
  model_demo.joblib       合成数据训练 —— ⚠️ 仅验证流程,勿信真实结果
  model_physionet.joblib  真实 PhysioNet 数据训练 —— 可用(仍非诊断)

⚠️ 安全提示:.joblib 走 pickle 反序列化,**加载即执行**其中的代码。只加载
   你自己训练的、或来源可信的模型文件。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import os
import sys

import numpy as np

from classifier import FEATURE_NAMES, HeartSoundClassifier, feature_vector
from heartbeat import HeartSoundAnalyzer


def classify(x: np.ndarray, fs: float, model_path: str) -> dict:
    """对一段波形做预测,返回概率与关键特征。

    信号质量不合格时 `abnormal_prob` 仍会算出来,但 `reliable=False`;
    调用方**必须**先看 reliable 再决定要不要采信,见 _print_report。
    """
    clf = HeartSoundClassifier.load(model_path)
    if list(clf.feature_names) != list(FEATURE_NAMES):
        raise SystemExit(
            f"模型 {model_path} 的特征集与当前代码不一致"
            f"({len(clf.feature_names)} vs {len(FEATURE_NAMES)} 个特征)。\n"
            f"该模型由旧版本训练,直接使用会得到无意义的结果。\n"
            f"请重新训练: python classifier.py --train-demo"
        )
    res = HeartSoundAnalyzer().analyze(x, fs)
    feat = feature_vector(res)
    prob = float(clf.predict_proba([feat])[0])     # 异常概率
    return {"abnormal_prob": prob, "result": res, "features": feat,
            "reliable": bool(res["reliable"])}


def _fmt(v, spec: str, dash: str = "—") -> str:
    """None 安全的格式化。

    原实现直接 f"{res['bpm']:.0f}",而 estimate_bpm 在录音短于约 0.33 秒时
    正常返回 None,于是抛
    TypeError: unsupported format string passed to NoneType.__format__
    """
    return dash if v is None else format(v, spec)


def _print_report(out: dict, model_path: str, force: bool = False) -> int:
    """打印报告。返回进程退出码:0=给出了判定,2=因质量不合格拒绝判定。"""
    res = out["result"]
    sqi = res.get("sqi") or {}
    demo = "demo" in os.path.basename(model_path).lower()
    if demo:
        print("⚠️  DEMO 模型(合成数据训练):以下结果仅演示流程,切勿当真。\n")

    # --- 质量门控:低质量录音不给判定 ---------------------------------------
    # 此前 predict.py 完全不做门控(实时路径却做了),于是一段仅含环境噪声的
    # 正常心音也会得到"判定: 异常  异常概率 0.99"。
    if not out["reliable"] and not force:
        print(f"⛔ 信号质量不合格,不予判定:{res.get('unreliable_reason')}")
        print(f"   SQI {_fmt(sqi.get('sqi'), '.2f')}"
              f"(周期性 {_fmt(sqi.get('periodicity'), '.2f')}"
              f" / 带内占比 {_fmt(sqi.get('band_ratio'), '.2f')}"
              f" / 工频 {_fmt(sqi.get('hum'), '.2f')}"
              f" / 削波 {_fmt(sqi.get('clip'), '.3f')})")
        print("   请改善采集后重录;确要看结果请加 --force(结果不可信)。")
        return 2

    p = out["abnormal_prob"]
    if force and not out["reliable"]:
        print(f"⚠️  --force:信号质量不合格({res.get('unreliable_reason')}),"
              "以下判定不可信。\n")
    n = int(round(p * 20))
    print(f"判定: {'异常' if p >= 0.5 else '正常'}   "
          f"异常概率 [{'█' * n + '·' * (20 - n)}] {p:.2f}")
    print(f"心率 {_fmt(res['bpm'], '.0f')} BPM "
          f"(置信度 {res['confidence']:.2f}) | SQI {_fmt(sqi.get('sqi'), '.2f')}")

    rh = res.get("rhythm") or {}
    if rh.get("classification"):
        print(f"节律: {rh['classification']}")
    mu = res.get("murmur") or {}
    if mu.get("flag") not in (None, "数据不足", "未见明显杂音"):
        # 形状/时相只在**收缩期**窗口上计算(见 murmur_shape),因此仅当确实
        # 疑似收缩期杂音时才展示。否则舒张期杂音会被附上一个基于收缩期静默段
        # 算出的、毫无意义的"全收缩期平台型"。
        extra = ""
        if ("收缩期杂音" in mu["flag"] and not mu.get("noise_suspect")
                and mu.get("shape") not in (None, "数据不足", "不定形")):
            extra = f" ({mu['shape']}/{mu['timing']})"
        print(f"杂音: {mu['flag']}{extra}")
    ex = res.get("extra_sounds") or {}
    if ex.get("flag") not in (None, "数据不足", "未见明显额外音"):
        print(f"额外音: {ex['flag']}")
    print("\n(本工具非诊断,如有不适请就医)")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    model = "model_demo.joblib"
    if "--model" in args:
        i = args.index("--model")
        if i + 1 >= len(args):          # 原实现在此直接 IndexError 抛栈回溯
            print("--model 后面需要跟模型文件路径")
            return 1
        model = args[i + 1]
        args = args[:i] + args[i + 2:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]

    if not os.path.exists(model):
        print(f"找不到模型 {model};先运行 python classifier.py --train-demo")
        return 1

    if "--selftest" in args:
        from synth import synthesize
        x, fs = synthesize(bpm=72, secs=12, murmur=0.5, murmur_shape="diamond",
                           seed=42)
        return _print_report(classify(x, fs, model), model, force)
    if args:
        try:
            import soundfile as sf
        except ImportError:
            print("读取音频需要 soundfile:pip install soundfile")
            return 1
        try:
            sig, fs = sf.read(args[0])
        except Exception as e:                      # noqa: BLE001
            print(f"无法读取 {args[0]}: {type(e).__name__}: {e}")
            return 1
        if sig.ndim > 1:
            sig = sig[:, 0]
        return _print_report(classify(sig, fs, model), model, force)
    print(__doc__)
    return 0


if __name__ == "__main__":
    try:   # Windows 控制台默认 GBK;stderr 也要一起切,否则报错信息乱码
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
