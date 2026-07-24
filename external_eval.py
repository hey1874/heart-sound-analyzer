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
                            [--per-subset 250]   # 每个来源子库最多取多少条

抽样说明:默认每子库上限 250 条。全量 3240 条里子库 e 一家占 2141 条,全跑
等于反复测同一个来源;而 AUC 的标准误在约 165 阳性/635 阴性时已 ≈0.02,足以
分辨本验证关心的差异。设 --per-subset 0 可跑全量。

目录约定:<2016目录>/REFERENCE-{a..f}.csv + <2016目录>/data/*.wav
标签 -1=正常 / 1=异常,统一映射为 0/1。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import csv
import os

import numpy as np

from heartsound import HeartSoundAnalyzer
from heartsound.classifier import HeartSoundClassifier, feature_vector
from heartsound.evalutil import auc




def load_2016(directory: str, cnn_path: str | None = None,
              per_subset: int = 250):
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
        got = [(sub, r[0].strip(), int(r[1].strip()))
               for r in csv.reader(open(ref)) if len(r) >= 2]
        if per_subset and len(got) > per_subset:
            # 分层抽样:按原始阳性率保留,不打乱类别比例
            pos = [g for g in got if g[2] == 1]
            neg = [g for g in got if g[2] != 1]
            frac = per_subset / len(got)
            rng = np.random.default_rng(0)
            keep = ([pos[i] for i in rng.permutation(len(pos))[
                        :max(1, round(len(pos) * frac))]]
                    + [neg[i] for i in rng.permutation(len(neg))[
                        :max(1, round(len(neg) * frac))]])
            print(f"  子库 {sub}: {len(got)} -> 抽样 {len(keep)} 条")
            got = keep
        rows.extend(got)
    print(f"合计 {len(rows)} 条")

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


def main(argv: list[str]) -> int:
    from heartsound.cliutil import Parser
    p = Parser("在 PhysioNet/CinC 2016 上做外部验证")
    p.add_argument("directory", help="2016 数据目录(含 REFERENCE-{a..f}.csv 与 data/)")
    p.add_argument("--cnn", help="CirCor 上训好的 CNN 模型(train_cnn.py --save 产出)")
    p.add_argument("--circor", help="CirCor 目录,用于零样本迁移对照")
    p.add_argument("--cache", default="ext2016.npz", help="特征缓存文件")
    p.add_argument("--per-subset", type=int, default=250,
                   help="每个来源子库最多取多少条(0=全量)")
    a = p.parse_args(argv)

    if os.path.exists(a.cache):
        d = np.load(a.cache, allow_pickle=True)
        X, y, sub, names, rel = d["X"], d["y"], d["sub"], d["names"], d["rel"]
        cnn_s = d["cnn"] if "cnn" in d and d["cnn"].size else None
        print(f"从缓存读取 {a.cache}")
    else:
        X, y, sub, names, rel, cnn_s = load_2016(a.directory, a.cnn, a.per_subset)
        np.savez_compressed(a.cache, X=X, y=y, sub=sub, names=names, rel=rel,
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
    if a.circor:
        import pickle
        cc = os.path.join(a.circor, "..", "fair_feats.pkl")
        if os.path.exists(cc):
            Xc, yc, gc, _, _ = pickle.load(open(cc, "rb"))
            clf = HeartSoundClassifier().fit(np.array(Xc, float),
                                             np.array(yc, int),
                                             groups=np.array(gc))
            pr = clf.predict_proba(X)
            print(f"   树模型(CirCor 训练)       AUC {auc(pr, y):.3f}")
            print(f"     仅通过门控的录音          AUC {auc(pr[rel], y[rel]):.3f}")
        else:
            print(f"   (未找到 {cc},跳过树模型迁移对照)")

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
    from heartsound.cliutil import run
    run(main)
