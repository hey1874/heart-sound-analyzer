"""
calibrate.py — 用带标注的真实心音数据标定阈值,并验证分段与杂音检测。

为什么需要它
------------
`HeartSoundAnalyzer` 的判定阈值(murmur_thr / conf_thr / sqi_thr ...)原本是
经验值。合成信号上它们看起来完美(正常 0.02 vs 杂音 1.63,相差 80 倍),但
**真实数据上完全是另一回事**——见 docs/VALIDATION.md。阈值必须在标注数据上
选工作点,而不是拍脑袋。

用法
----
    python calibrate.py <CirCor目录> [--out thresholds.json] [--limit N]

目录需为 PhysioNet CirCor DigiScope 格式(开放许可 ODC-BY,无需 credential):
    https://physionet.org/content/circor-heart-sound/
    training_data.csv        患者级标注(含 Murmur / Murmur locations / 分级)
    training_data/*.wav      录音
    training_data/*.tsv      S1/收缩/S2/舒张 逐段真值(1/2/3/4,0=未标注)

产出
----
1. **门控标定**:用 .tsv 标注算出参考心率,给出 conf_thr 的"保留率 vs 心率
   误差"权衡表——门控该拦的是**测不准的录音**,不是"看起来不像我的合成信号"。
2. **分段验证**:HSMM 检出的 S1 与官方标注比敏感度/查准率/定位误差。
3. **杂音阈值标定**:ROC + 若干工作点(Youden 最优、敏感度≥0.80/0.90)。
4. **分类器评估**:按**患者**分组交叉验证,并与不分组对照量化泄漏。
5. 写出 thresholds.json,可用 `HeartSoundAnalyzer.from_json()` 加载。

⚠️ 交叉验证给出的是该数据集、该人群、该设备上的结果。CirCor 采自巴西儿童
   筛查(数字听诊器),换成成人或换成"听诊器+咪头+声卡"链路,阈值需重新标定。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

from classifier import (FEATURE_NAMES, HeartSoundClassifier, feature_vector,
                        threshold_baseline)
from heartbeat import HeartSoundAnalyzer

WAVE_S1 = 1        # .tsv 中的 wave 编号


def load_segments(path: str) -> list[tuple[float, float, int]]:
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


def _eval_segmentation(an, sig, fs, bpm, segs, tol_ms: float = 100.0):
    """单条录音的 S1 分段评估,返回 (tp, fn, fp, 误差列表)。

    落在未标注区(wave 0)的检出不计假阳性——那里没有真值可比。不排除的话
    查准率会被不公平地压到 0.574(实测)。
    """
    gt = [s for s, _, w in segs if w == WAVE_S1]
    ann = [(s, e) for s, e, w in segs if w != 0]
    if len(gt) < 4 or not ann:
        return None
    env = an.shannon_envelope(an._filtfilt(an.sos_seg, an.resample(sig, fs)))
    s1, _ = an.segment_hsmm(env, bpm)
    det = np.sort(s1) / an.fs
    det = np.array([t for t in det if any(a <= t <= b for a, b in ann)])
    tp = fn = 0
    used, errs = set(), []
    for g in gt:
        if len(det) == 0:
            fn += 1
            continue
        k = int(np.argmin(np.abs(det - g)))
        if abs(det[k] - g) <= tol_ms / 1000.0 and k not in used:
            tp += 1
            used.add(k)
            errs.append(abs(det[k] - g))
        else:
            fn += 1
    return tp, fn, max(0, len(det) - len(used)), errs


def collect(directory: str, limit: int | None = None) -> list[dict]:
    """遍历数据集,对每条录音跑分析并带上标签与真值。"""
    import soundfile as sf

    meta_path = os.path.join(directory, "training_data.csv")
    if not os.path.exists(meta_path):
        raise SystemExit(f"找不到 {meta_path};请确认是 CirCor 数据集目录")
    meta = {r["Patient ID"]: r for r in
            csv.DictReader(open(meta_path, encoding="utf-8"))}

    wavs = sorted(glob.glob(os.path.join(directory, "training_data", "*.wav")))
    if not wavs:                                   # 也接受平铺目录
        wavs = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if limit:
        wavs = wavs[:limit]
    print(f"发现 {len(wavs)} 条录音,开始提取特征 ...")

    an = HeartSoundAnalyzer()
    out = []
    for i, w in enumerate(wavs):
        stem = os.path.splitext(os.path.basename(w))[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        pid, loc = parts[0], parts[1]
        m = meta.get(pid)
        if m is None or m["Murmur"] == "Unknown":   # Unknown 不参与标定
            continue
        try:
            sig, fs = sf.read(w)
        except Exception as e:                      # noqa: BLE001
            print(f"  跳过 {stem}: {type(e).__name__}")
            continue
        if getattr(sig, "ndim", 1) > 1:
            sig = sig[:, 0]
        res = an.analyze(sig, fs)

        segs, ref, seg_eval = [], None, None
        tsv = w[:-4] + ".tsv"
        if os.path.exists(tsv):
            segs = load_segments(tsv)
            ref = reference_bpm(segs)
            # 分段评估在这里就地做完。此前把整段波形留在结果里供后续用,
            # 全量 CirCor(3163 条)转 float64 后要占 2GB+ 内存。
            if res["reliable"]:
                seg_eval = _eval_segmentation(an, sig, fs, res["bpm"], segs)

        out.append({
            "seg_eval": seg_eval,
            "stem": stem, "patient": pid, "location": loc,
            # 标签:患者有杂音**且**该听诊区在杂音位置列表中,才算阳性
            "y": int(m["Murmur"] == "Present"
                     and loc in (m["Murmur locations"] or "").split("+")),
            "grade": m["Systolic murmur grading"],
            "res": res, "feat": feature_vector(res),
            "ref_bpm": ref, "has_segs": bool(segs),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(wavs)} ...", flush=True)
    return out


# ---------------------------------------------------------------------------

def report_gate(recs: list[dict]) -> dict:
    """门控标定:保留率 vs 保留样本的心率误差。"""
    print("\n" + "=" * 68)
    print("1. 信号质量门控标定")
    print("=" * 68)
    have = [r for r in recs if r["ref_bpm"] and r["res"]["bpm"]]
    print(f"  {len(have)}/{len(recs)} 条有官方标注可算参考心率")
    if not have:
        return {}

    def err(sub):
        return np.array([abs(r["res"]["bpm"] - r["ref_bpm"]) for r in sub])

    for label, sub in (("判可采信", [r for r in have if r["res"]["reliable"]]),
                       ("被拒收  ", [r for r in have if not r["res"]["reliable"]])):
        if sub:
            e = err(sub)
            print(f"  {label}: n={len(sub):>4}  心率误差中位 {np.median(e):5.1f} BPM, "
                  f">5BPM {np.mean(e > 5):.0%}, >10BPM {np.mean(e > 10):.0%}")

    print("\n  conf_thr 权衡表(门控该拦的是测不准的录音):")
    print(f"  {'门限':>6} {'保留率':>7} {'中位误差':>9} {'>5BPM':>7} {'>10BPM':>8}")
    table = []
    for thr in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        sub = [r for r in have if r["res"]["confidence"] >= thr]
        if not sub:
            continue
        e = err(sub)
        row = {"conf_thr": thr, "keep": len(sub) / len(have),
               "median_err": float(np.median(e)),
               "gross_err": float(np.mean(e > 10))}
        table.append(row)
        print(f"  {thr:>6.2f} {row['keep']:>6.0%} {row['median_err']:>8.1f} "
              f"{np.mean(e > 5):>6.0%} {row['gross_err']:>7.0%}")
    return {"conf_thr_tradeoff": table}


def report_segmentation(recs: list[dict], tol_ms: float = 100.0) -> dict:
    """分段验证:汇总 collect() 阶段就地算好的逐条结果。"""
    print("\n" + "=" * 68)
    print(f"2. HSMM 分段 vs 官方 S1 标注(±{tol_ms:.0f}ms 容差)")
    print("=" * 68)
    tp = fn = fp = n_rec = 0
    errs = []
    for r in recs:
        if not r.get("seg_eval"):
            continue
        a, b, c, e = r["seg_eval"]
        tp += a
        fn += b
        fp += c
        errs += e
        n_rec += 1
    if not (tp + fn):
        print("  无可用标注")
        return {}
    se, ppv = tp / (tp + fn), tp / max(1, tp + fp)
    f1 = 2 * se * ppv / max(1e-9, se + ppv)
    e = np.array(errs)
    print(f"  {n_rec} 条录音,真值 S1 共 {tp + fn} 个")
    print(f"  敏感度 {se:.3f}   查准率 {ppv:.3f}   F1 {f1:.3f}")
    print(f"  定位误差:中位 {np.median(e) * 1000:.0f}ms  "
          f"均值 {e.mean() * 1000:.0f}ms  90分位 {np.percentile(e, 90) * 1000:.0f}ms")
    return {"segmentation": {"sensitivity": se, "ppv": ppv, "f1": f1,
                             "median_err_ms": float(np.median(e) * 1000),
                             "n_recordings": n_rec, "n_s1": tp + fn}}


def _roc(pos: np.ndarray, neg: np.ndarray) -> float:
    v = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ys = y[np.argsort(-v)]
    tpr = np.cumsum(ys) / max(1, ys.sum())
    fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(tpr, fpr))


def report_murmur(recs: list[dict], default_thr: float = 0.15) -> dict:
    """杂音阈值标定:ROC 与若干工作点。"""
    print("\n" + "=" * 68)
    print("3. 杂音指数阈值标定(仅用通过门控的录音)")
    print("=" * 68)
    ok = [r for r in recs
          if r["res"]["reliable"] and (r["res"]["murmur"] or {}).get("systolic")
          is not None and np.isfinite(r["res"]["murmur"]["systolic"])]
    pos = np.array([r["res"]["murmur"]["systolic"] for r in ok if r["y"] == 1])
    neg = np.array([r["res"]["murmur"]["systolic"] for r in ok if r["y"] == 0])
    print(f"  有杂音 {len(pos)} 条 / 无杂音 {len(neg)} 条")
    if not (len(pos) and len(neg)):
        return {}
    print(f"  有杂音 murmur_sys 中位 {np.median(pos):.3f} "
          f"(IQR {np.percentile(pos, 25):.3f}-{np.percentile(pos, 75):.3f})")
    print(f"  无杂音 murmur_sys 中位 {np.median(neg):.3f} "
          f"(IQR {np.percentile(neg, 25):.3f}-{np.percentile(neg, 75):.3f})")
    auc = _roc(pos, neg)
    print(f"  AUC = {auc:.3f}")

    out = {"auc": auc, "n_pos": len(pos), "n_neg": len(neg), "points": {}}
    cands = np.unique(np.concatenate([pos, neg]))

    def at(t):
        return float(np.mean(pos > t)), float(np.mean(neg <= t))

    se, sp = at(default_thr)
    print(f"\n  当前默认阈值 {default_thr}: 敏感度 {se:.3f}  特异度 {sp:.3f}")
    out["points"]["default"] = {"thr": default_thr, "sens": se, "spec": sp}

    j = [(se_ + sp_ - 1, t, se_, sp_) for t in cands for se_, sp_ in [at(t)]]
    best = max(j)
    print(f"  Youden 最优阈值 {best[1]:.3f}: 敏感度 {best[2]:.3f}  "
          f"特异度 {best[3]:.3f}  (J={best[0]:.3f})")
    out["points"]["youden"] = {"thr": float(best[1]), "sens": best[2],
                               "spec": best[3]}

    for target in (0.80, 0.90):
        feas = [(sp_, t, se_) for t in cands for se_, sp_ in [at(t)]
                if se_ >= target]
        if feas:
            sp_, t, se_ = max(feas)
            print(f"  敏感度≥{target:.2f} 的最佳阈值 {t:.3f}: "
                  f"敏感度 {se_:.3f}  特异度 {sp_:.3f}")
            out["points"][f"sens{int(target * 100)}"] = {
                "thr": float(t), "sens": se_, "spec": sp_}

    print("\n  按杂音分级的检出率(默认阈值下):")
    from collections import defaultdict
    byg = defaultdict(list)
    for r in ok:
        if r["y"] == 1:
            byg[r["grade"]].append(r["res"]["murmur"]["systolic"])
    for g in sorted(byg):
        a = np.array(byg[g])
        print(f"    {g:>8}: n={len(a):>3}  中位 {np.median(a):.3f}  "
              f"检出率 {np.mean(a > default_thr):.0%}")
    return {"murmur": out}


def report_ml(recs: list[dict]) -> dict:
    """分类器评估:按患者分组 vs 不分组(量化泄漏)。"""
    print("\n" + "=" * 68)
    print("4. 分类器评估(28 维工程特征 + HistGradientBoosting)")
    print("=" * 68)
    X = np.array([r["feat"] for r in recs], float)
    y = np.array([r["y"] for r in recs], int)
    g = np.array([r["patient"] for r in recs])
    rel = np.array([r["res"]["reliable"] for r in recs], bool)

    out = {}
    for tag, mask in (("全部录音", np.ones(len(y), bool)),
                      ("仅通过门控", rel)):
        Xs, ys, gs = X[mask], y[mask], g[mask]
        if len(set(ys.tolist())) < 2 or len(set(gs.tolist())) < 6:
            continue
        print(f"\n  --- {tag} (n={len(ys)}, 患者 {len(set(gs.tolist()))}) ---")
        clf = HeartSoundClassifier()
        cv = clf.cross_validate(Xs, ys, cv=5, groups=gs)
        cvn = clf.cross_validate(Xs, ys, cv=5)
        print(f"  按患者分组: 平衡准确率 {cv['bacc_mean']:.3f}±{cv['bacc_std']:.3f}  "
              f"AUC {cv['auc_mean']:.3f}±{cv['auc_std']:.3f}")
        print(f"              敏感度 {cv['sens_mean']:.3f}  特异度 {cv['spec_mean']:.3f}")
        print(f"  不分组对照: AUC {cvn['auc_mean']:.3f}  "
              f"← 受试者级泄漏使 AUC 虚高 {cvn['auc_mean'] - cv['auc_mean']:+.3f}")
        b = threshold_baseline(Xs, ys)
        print(f"  单阈值基线({b['feature']}): 准确率 {b['acc']:.3f}")
        print("  Top-5 特征:")
        for nm, imp in clf.feature_importance(Xs, ys)[:5]:
            print(f"    {nm:20s} {imp:.4f}")
        out[tag] = {"grouped_auc": cv["auc_mean"], "grouped_bacc": cv["bacc_mean"],
                    "ungrouped_auc": cvn["auc_mean"],
                    "leak": cvn["auc_mean"] - cv["auc_mean"],
                    "baseline_acc": b["acc"], "n": len(ys)}
    return {"ml": out}


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 1
    directory = argv[0]
    out_path = "thresholds.json"
    limit = None
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])

    recs = collect(directory, limit)
    if not recs:
        print("没有可用录音")
        return 1
    print(f"\n可用 {len(recs)} 条录音 / {len(set(r['patient'] for r in recs))} 名患者"
          f"(阳性 {sum(r['y'] for r in recs)})")

    result = {"dataset": os.path.abspath(directory), "n_recordings": len(recs),
              "n_patients": len(set(r["patient"] for r in recs)),
              "feature_names": list(FEATURE_NAMES)}
    result.update(report_gate(recs))
    result.update(report_segmentation(recs))
    result.update(report_murmur(recs))
    result.update(report_ml(recs))

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n已写出 {out_path}")
    print("用 HeartSoundAnalyzer.from_json() 加载标定后的阈值。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
