"""
external_eval.py — 在 PhysioNet/CinC 2016 上做**外部验证**。

为什么必须做
------------
本项目在 CirCor 上报出的工程特征线成绩(AUC 0.876)是**看着同一份交叉验证
结果反复挑参数**得来的:分段带 6 选 1、频谱聚合方式 4 选 1、多区融合 4 选 1
…… 每次"取最好"都在往验证集上过拟合一点。这类数字系统性偏乐观,必须在一份
**从未参与过任何决策**的数据上重跑才算数。

CinC 2016 恰好是理想的外部集:
  - 不同人群(成人为主,CirCor 是巴西儿童筛查)
  - 不同设备(多种听诊器,CirCor 是 DigiScope 数字听诊器)
  - 不同标注体系(正常/异常,CirCor 是有无杂音)
  - 6 个来源子库(a-f),可做留一子库交叉验证 = 直接测跨设备泛化

⚠️ 标签不是同一个任务
---------------------
2016 的"异常"包含瓣膜病、冠心病等多种诊断,杂音只是其中一部分成因;而 CirCor
模型学的是"这个听诊区有没有杂音"。因此 **CirCor→2016 的迁移 AUC 必然低于
CirCor 内部 AUC,这不全是模型的问题**。为了把「域偏移」和「任务差异」分开,
本脚本同时给出:

  A. 零样本迁移:CirCor 上训好的模型直接打分 2016     -> 域偏移 + 任务差异
  B. 2016 内部交叉验证(留一子库):同一套管线就地训练 -> 该数据集的可达上限
  C. 随机分层交叉验证对照:与 B 比较                  -> 单看跨子库损失多少

用法:
    python external_eval.py <2016目录> [--cnn cnn_circor.pt] [--circor <CirCor目录>]

目录约定:<2016目录>/REFERENCE-{a..f}.csv + <2016目录>/data/*.wav
标签 -1=正常 / 1=异常,统一映射为 0/1。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

from classifier import HeartSoundClassifier, feature_vector
from heartbeat import HeartSoundAnalyzer


def auc(score: np.ndarray, label: np.ndarray) -> float:
    m = np.isfinite(score)
    score, label = score[m], label[m]
    if len(set(label.tolist())) < 2:
        return float("nan")
    ys = label[np.argsort(-score)]
    tpr = np.cumsum(ys) / max(1, ys.sum())
    fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(tpr, fpr))


def load_2016(directory: str, cnn_path: str | None = None):
    """提取 2016 的工程特征(以及可选的 CNN 分数)。"""
    import soundfile as sf
    an = HeartSoundAnalyzer()

    model = device = None
    if cnn_path and os.path.exists(cnn_path):
        import torch
        from cnn import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, meta = load_model(cnn_path, device)
        print(f"已加载 CNN(训练于 {meta.get('n_recordings')} 条 CirCor 录音)")

    rows = []
    for sub in "abcdef":
        ref = os.path.join(directory, f"REFERENCE-{sub}.csv")
        if not os.path.exists(ref):
            continue
        for r in csv.reader(open(ref)):
            if len(r) < 2:
                continue
            rows.append((sub, r[0].strip(), int(r[1].strip())))
    print(f"标签共 {len(rows)} 条")

    X, y, sub_of, names, rel, cnn_s = [], [], [], [], [], []
    for i, (sub, stem, lab) in enumerate(rows):
        w = os.path.join(directory, "data", stem + ".wav")
        if not os.path.exists(w):
            continue
        try:
            sig, fs = sf.read(w)
        except Exception:
            continue
        if getattr(sig, "ndim", 1) > 1:
            sig = sig[:, 0]
        res = an.analyze(sig, fs)
        X.append(feature_vector(res))
        y.append(1 if lab == 1 else 0)
        sub_of.append(sub)
        names.append(stem)
        rel.append(res["reliable"])
        if model is not None:
            from cnn import score_recording
            cnn_s.append(score_recording(model, sig, fs, device))
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(rows)} ...", flush=True)
    return (np.array(X, float), np.array(y, int), np.array(sub_of),
            np.array(names), np.array(rel, bool),
            np.array(cnn_s, float) if cnn_s else None)


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 1
    directory = argv[0]
    cnn_path = argv[argv.index("--cnn") + 1] if "--cnn" in argv else None
    circor = argv[argv.index("--circor") + 1] if "--circor" in argv else None
    cache = argv[argv.index("--cache") + 1] if "--cache" in argv else "ext2016.npz"

    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y, sub, names, rel = d["X"], d["y"], d["sub"], d["names"], d["rel"]
        cnn_s = d["cnn"] if "cnn" in d and d["cnn"].size else None
        print(f"从缓存读取 {cache}")
    else:
        X, y, sub, names, rel, cnn_s = load_2016(directory, cnn_path)
        np.savez_compressed(cache, X=X, y=y, sub=sub, names=names, rel=rel,
                            cnn=cnn_s if cnn_s is not None else np.array([]))
    print(f"\n可用 {len(y)} 条 / 异常 {int(y.sum())} / 正常 {int((1 - y).sum())}"
          f" / 通过门控 {int(rel.sum())} ({rel.mean():.0%})")
    for s in sorted(set(sub.tolist())):
        m = sub == s
        print(f"  子库 {s}: {m.sum():>5} 条,异常率 {y[m].mean():.0%}")

    print("\n" + "=" * 66)
    print("A. 零样本迁移:CirCor 上训好的模型直接打分 2016")
    print("=" * 66)
    print("   ⚠️ 2016 的'异常'涵盖多种诊断,杂音只是其一;迁移 AUC 必然低于")
    print("      CirCor 内部 AUC,这不全是模型的问题。")
    if cnn_s is not None:
        print(f"   CNN(CirCor 训练)          AUC {auc(cnn_s, y):.3f}")
        print(f"     仅通过门控的录音          AUC {auc(cnn_s[rel], y[rel]):.3f}")
    if circor:
        import pickle
        cc = os.path.join(circor, "..", "fair_feats.pkl")
        if os.path.exists(cc):
            Xc, yc, gc, _, _ = pickle.load(open(cc, "rb"))
            clf = HeartSoundClassifier().fit(np.array(Xc, float),
                                             np.array(yc, int),
                                             groups=np.array(gc))
            p = clf.predict_proba(X)
            print(f"   树模型(CirCor 训练)       AUC {auc(p, y):.3f}")
            print(f"     仅通过门控的录音          AUC {auc(p[rel], y[rel]):.3f}")

    print("\n" + "=" * 66)
    print("B. 2016 内部:留一子库交叉验证(直接测跨设备/跨来源泛化)")
    print("=" * 66)
    oof = np.full(len(y), np.nan)
    for s in sorted(set(sub.tolist())):
        te = sub == s
        tr = ~te
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            print(f"   留出子库 {s}: 类别单一,跳过")
            continue
        clf = HeartSoundClassifier().fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])
        print(f"   留出子库 {s} ({te.sum():>5} 条): AUC {auc(oof[te], y[te]):.3f}")
    print(f"   合并                        AUC {auc(oof, y):.3f}")

    print("\n" + "=" * 66)
    print("C. 对照:随机分层交叉验证(不按子库分组)")
    print("=" * 66)
    from sklearn.model_selection import StratifiedKFold
    oof2 = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = HeartSoundClassifier().fit(X[tr], y[tr])
        oof2[te] = clf.predict_proba(X[te])
    print(f"   随机 5 折                    AUC {auc(oof2, y):.3f}")
    print(f"   与留一子库之差               {auc(oof2, y) - auc(oof, y):+.3f}"
          "  ← 这就是跨来源泛化的代价")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
