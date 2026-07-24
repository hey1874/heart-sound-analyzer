"""
classifier.py — 向后兼容入口 + 训练/评估命令行。

实现已移到 `heartsound.classifier`。新代码请用:

    from heartsound.classifier import HeartSoundClassifier, extract_features

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations



from heartsound.classifier import (FEATURE_NAMES, HeartSoundClassifier,
                                   _synthetic_dataset, extract_features,
                                   feature_vector, load_physionet2016,
                                   threshold_baseline)

__all__ = ["FEATURE_NAMES", "HeartSoundClassifier", "extract_features",
           "feature_vector", "load_physionet2016", "threshold_baseline"]


def main(argv: list[str]) -> int:
    from heartsound.cliutil import Parser
    p = Parser("心音正常/异常分类基线(工程特征 + HistGradientBoosting)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true",
                   help="合成数据交叉验证 + 单阈值基线对照")
    g.add_argument("--train-demo", action="store_true",
                   help="在合成数据上训练并保存 model_demo.joblib")
    g.add_argument("--train-physionet", metavar="目录",
                   help="在 PhysioNet/CinC 2016 格式目录上训练")
    a = p.parse_args(argv)

    if a.demo:
        print("合成数据集(正常 vs 杂音/S3;正常类含 1/3 高噪声样本)...")
        X, y = _synthetic_dataset(n_per_class=40)
        print(f"样本 {len(y)}  特征 {X.shape[1]}  "
              f"(正常 {int((y == 0).sum())} / 异常 {int((y == 1).sum())})")
        clf = HeartSoundClassifier()
        cv = clf.cross_validate(X, y, cv=5)
        print("\n5 折交叉验证:")
        print(f"  准确率   {cv['acc_mean']:.3f}±{cv['acc_std']:.3f}")
        print(f"  平衡准确率 {cv['bacc_mean']:.3f}±{cv['bacc_std']:.3f}")
        print(f"  AUC      {cv['auc_mean']:.3f}±{cv['auc_std']:.3f}")
        print(f"  敏感度   {cv['sens_mean']:.3f}   特异度 {cv['spec_mean']:.3f}")
        base = threshold_baseline(X, y)
        gain = cv["acc_mean"] - base["acc"]
        print(f"\n对照 — 单特征阈值基线({base['feature']} > {base['thr']:.3f}):"
              f" 准确率 {base['acc']:.3f}")
        print(f"      树模型相对增益: {gain:+.3f}"
              + ("  (增益可观,多特征确实有贡献)" if gain > 0.05 else
                 "  ⚠️ 增益很小,说明这个数据集靠单阈值就能分开,指标没有说明力"))
        print("\n驱动决策的 Top-6 特征(留出集上的置换重要性):")
        for name, imp in clf.feature_importance(X, y)[:6]:
            print(f"  {name:20s} {imp:.3f}")
        print("\n⚠️ 这是合成数据上的管线验证。真实数据结果见 docs/VALIDATION.md")
        return 0

    if a.train_demo:
        out = "model_demo.joblib"
        X, y = _synthetic_dataset(n_per_class=60)
        HeartSoundClassifier().fit(X, y).save(out)
        print(f"✅ 已保存 demo 模型 -> {out}  (合成数据训练,仅验证流程)")
        print("   用法: python predict.py <某段录音.wav>")
        return 0

    print(f"读取真实数据 {a.train_physionet} ...")
    X, y, names = load_physionet2016(a.train_physionet)
    n_norm, n_abn = int((y == 0).sum()), int((y == 1).sum())
    print(f"样本 {len(y)} (正常 {n_norm} / 异常 {n_abn}"
          f",比例 {n_norm / max(1, n_abn):.1f}:1)")
    clf = HeartSoundClassifier()
    cv = clf.cross_validate(X, y, cv=5)
    print("5 折交叉验证:")
    print(f"  准确率 {cv['acc_mean']:.3f}  平衡准确率 {cv['bacc_mean']:.3f}  "
          f"AUC {cv['auc_mean']:.3f}")
    print(f"  敏感度 {cv['sens_mean']:.3f}  特异度 {cv['spec_mean']:.3f}")
    base = threshold_baseline(X, y)
    print(f"  对照 — 单特征阈值基线({base['feature']}): 准确率 {base['acc']:.3f}")
    print("  ⚠️ 按录音切分,未按受试者分组;若同一受试者有多条录音,"
          "以上指标是乐观上界(见 load_physionet2016 文档)。")
    clf.fit(X, y).save("model_physionet.joblib")
    print("✅ 已保存 -> model_physionet.joblib")
    return 0


if __name__ == "__main__":
    from heartsound.cliutil import run
    run(main)
