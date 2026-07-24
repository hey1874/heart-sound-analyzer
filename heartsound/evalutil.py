"""
evalutil.py — 评估相关的公共定义。

为什么需要
----------
拆分前,AUC 计算在三处各写一份(`calibrate._roc`、`external_eval.auc`、
`train_cnn.auc`),CirCor 元数据加载和**杂音阳性标签的定义**也在两处重复。

标签定义重复最危险:那是**评估的口径**。两处写法一旦漂移,两边报出的数字
就不再可比,而且不会报错、不会有测试失败——只会让人拿着两个不同口径的
AUC 做对比。所以这里把它固化成唯一一份。
"""

from __future__ import annotations

import csv
import os

import numpy as np

WAVE_S1 = 1        # CirCor .tsv 里的 wave 编号:1=S1 2=收缩 3=S2 4=舒张 0=未标注


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC 曲线下面积。非有限的分数自动剔除;类别单一时返回 nan。"""
    score = np.asarray(score, float)
    label = np.asarray(label)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    if len(set(label.tolist())) < 2:
        return float("nan")
    ys = label[np.argsort(-score)]
    tpr = np.cumsum(ys) / max(1, ys.sum())
    fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(tpr, fpr))


def load_circor_meta(directory: str) -> dict:
    """读 CirCor 的 training_data.csv,返回 {患者ID: 行}。"""
    path = os.path.join(directory, "training_data.csv")
    if not os.path.exists(path):
        path = os.path.join(directory, "..", "training_data.csv")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 training_data.csv(在 {directory} 下)")
    with open(path, encoding="utf-8") as fh:
        return {r["Patient ID"]: r for r in csv.DictReader(fh)}


def murmur_label(meta_row: dict, location: str) -> int:
    """**唯一**的杂音阳性定义,供所有评估脚本共用。

    一条录音算阳性,当且仅当该患者杂音为 Present **且**这条录音所在听诊区
    出现在 `Murmur locations` 里。

    ⚠️ 把患者级标签直接套到该患者的全部听诊区是**错的**——杂音常只在部分
    听诊区可闻,那样做会把大量实际听不到杂音的录音标成阳性,人为压低指标。
    """
    if meta_row["Murmur"] != "Present":
        return 0
    return int(location in (meta_row["Murmur locations"] or "").split("+"))


def is_labeled(meta_row: dict) -> bool:
    """Murmur=Unknown 的患者(标注者自己也拿不准)不参与训练与评估。"""
    return meta_row.get("Murmur") in ("Present", "Absent")


def load_segments(path: str) -> list[tuple[float, float, int]]:
    """读 CirCor 的 .tsv 分段真值:(起, 止, wave)。"""
    out = []
    with open(path) as fh:
        for line in fh:
            a = line.split()
            if len(a) == 3:
                out.append((float(a[0]), float(a[1]), int(a[2])))
    return out


def reference_bpm(segs: list[tuple[float, float, int]]) -> float | None:
    """由官方 S1 标注的中位 RR 算参考心率。"""
    g = np.array([s for s, _, w in segs if w == WAVE_S1])
    if len(g) < 5:
        return None
    rr = np.diff(g)
    rr = rr[(rr > 0.3) & (rr < 1.6)]       # 去掉跨未标注区造成的跳变
    return 60.0 / float(np.median(rr)) if len(rr) >= 4 else None
