"""
classifier.py — 阶段4:心音正常/异常分类基线(ML)。

思路:把 `HeartSoundAnalyzer` 产出的全部 DSP/临床特征(心率、SQI、节律、STI、
杂音能量/形状、S3/S4、S2 分裂)拼成一个特征向量,接 scikit-learn 的
HistGradientBoosting 分类器(原生支持缺失值 NaN,适合特征常缺的场景)。

- 这是**可解释的工程特征 + 树模型**基线;更高上限是频谱图 + CNN(见 ROADMAP)。
- 真实数据用 PhysioNet/CinC 2016(REFERENCE.csv: 文件名,-1正常/1异常)。
- `--demo` 用合成正常 vs 杂音信号端到端验证管线可跑通并能区分。

依赖: numpy, scipy, scikit-learn(读音频另需 soundfile)。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import numpy as np

from heartbeat import HeartSoundAnalyzer
from synth import synthesize

# 固定的特征顺序(树模型可吃 NaN,缺失填 NaN 即可)
#
# 相比初版增加了 band_clean / hum(工频)、murmur_asymmetry(时相不对称度)、
# murmur_flatness(平坦度)。其中 murmur_asymmetry 是区分"真杂音"与"宽带
# 噪声"的关键量:噪声均匀铺满整个心动周期(不对称度≈0),真杂音集中在单一
# 时相(不对称度≈0.95)。
#
# 注意这里只放**连续量**、不放阈值化后的布尔量(如 noise_suspect):那些布尔
# 量由未标定的经验阈值算出,喂给模型等于把手工阈值当成事实。
#
# 末尾 48 维 spec_* 来自 `spectral_profile`,是本特征集里贡献最大的一组。
# CirCor 实测(885 条录音,按患者分组 5 折):
#   原有 29 维                        AUC 0.775-0.792
#   + 频谱轮廓(仅中位数聚合,24 维)    AUC 0.853
#   + 频谱轮廓(中位+90分位,48 维)     AUC 0.879   ← 当前
# 原因有两层:murmur_index 把整个收缩期的频谱压成一个标量,丢掉了频谱形状;
# 而只取各周期中位数又会把"只在部分周期可闻"的杂音冲淡(多示例问题)。
FEATURE_NAMES = [
    "bpm", "confidence",
    "sqi", "periodicity", "band_ratio", "band_clean", "hum", "clip",
    "cv", "cv_robust", "sd1_ms", "sd2_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "ectopic",
    "systole_ms", "diastole_ms", "sys_dia_ratio",
    "murmur_sys", "murmur_dia", "murmur_asymmetry",
    "murmur_occupancy", "murmur_peak_pos", "murmur_flatness",
    "s3", "s4", "s2_split_ms", "s2_split_frac",
] + [f"spec_{k}" for k in ['sys_med_b20_60', 'sys_med_b60_120', 'sys_med_b120_200', 'sys_med_b200_300', 'sys_med_b300_450', 'sys_med_b450_700', 'sys_med_centroid', 'sys_med_spread', 'sys_med_rolloff85', 'sys_med_flatness', 'sys_med_slope', 'sys_med_hi_lo', 'sys_p90_b20_60', 'sys_p90_b60_120', 'sys_p90_b120_200', 'sys_p90_b200_300', 'sys_p90_b300_450', 'sys_p90_b450_700', 'sys_p90_centroid', 'sys_p90_spread', 'sys_p90_rolloff85', 'sys_p90_flatness', 'sys_p90_slope', 'sys_p90_hi_lo', 'dia_med_b20_60', 'dia_med_b60_120', 'dia_med_b120_200', 'dia_med_b200_300', 'dia_med_b300_450', 'dia_med_b450_700', 'dia_med_centroid', 'dia_med_spread', 'dia_med_rolloff85', 'dia_med_flatness', 'dia_med_slope', 'dia_med_hi_lo', 'dia_p90_b20_60', 'dia_p90_b60_120', 'dia_p90_b120_200', 'dia_p90_b200_300', 'dia_p90_b300_450', 'dia_p90_b450_700', 'dia_p90_centroid', 'dia_p90_spread', 'dia_p90_rolloff85', 'dia_p90_flatness', 'dia_p90_slope', 'dia_p90_hi_lo']]


def _g(d: dict | None, *keys, default=np.nan):
    """安全地从可能为 None 的嵌套字典取值。"""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def feature_vector(res: dict) -> np.ndarray:
    """把 analyze() 的结果字典摊平为固定顺序的特征向量(缺失=NaN)。"""
    v = [
        _g(res, "bpm"), _g(res, "confidence"),
        _g(res, "sqi", "sqi"), _g(res, "sqi", "periodicity"),
        _g(res, "sqi", "band_ratio"), _g(res, "sqi", "band_clean"),
        _g(res, "sqi", "hum"), _g(res, "sqi", "clip"),
        _g(res, "rhythm", "cv"), _g(res, "rhythm", "cv_robust"),
        _g(res, "rhythm", "sd1_ms"),
        _g(res, "rhythm", "sd2_ms"), _g(res, "rhythm", "sdnn_ms"),
        _g(res, "rhythm", "rmssd_ms"), _g(res, "rhythm", "pnn50_pct"),
        _g(res, "rhythm", "ectopic"),
        _g(res, "sti", "systole_ms"), _g(res, "sti", "diastole_ms"),
        _g(res, "sti", "sys_dia_ratio"),
        _g(res, "murmur", "systolic"), _g(res, "murmur", "diastolic"),
        _g(res, "murmur", "asymmetry"),
        _g(res, "murmur", "occupancy"), _g(res, "murmur", "peak_pos"),
        _g(res, "murmur", "flatness"),
        _g(res, "extra_sounds", "s3"), _g(res, "extra_sounds", "s4"),
        _g(res, "s2_split", "split_ms"), _g(res, "s2_split", "split_frac"),
    ] + [_g(res, "spectrum", k) for k in ['sys_med_b20_60', 'sys_med_b60_120', 'sys_med_b120_200', 'sys_med_b200_300', 'sys_med_b300_450', 'sys_med_b450_700', 'sys_med_centroid', 'sys_med_spread', 'sys_med_rolloff85', 'sys_med_flatness', 'sys_med_slope', 'sys_med_hi_lo', 'sys_p90_b20_60', 'sys_p90_b60_120', 'sys_p90_b120_200', 'sys_p90_b200_300', 'sys_p90_b300_450', 'sys_p90_b450_700', 'sys_p90_centroid', 'sys_p90_spread', 'sys_p90_rolloff85', 'sys_p90_flatness', 'sys_p90_slope', 'sys_p90_hi_lo', 'dia_med_b20_60', 'dia_med_b60_120', 'dia_med_b120_200', 'dia_med_b200_300', 'dia_med_b300_450', 'dia_med_b450_700', 'dia_med_centroid', 'dia_med_spread', 'dia_med_rolloff85', 'dia_med_flatness', 'dia_med_slope', 'dia_med_hi_lo', 'dia_p90_b20_60', 'dia_p90_b60_120', 'dia_p90_b120_200', 'dia_p90_b200_300', 'dia_p90_b300_450', 'dia_p90_b450_700', 'dia_p90_centroid', 'dia_p90_spread', 'dia_p90_rolloff85', 'dia_p90_flatness', 'dia_p90_slope', 'dia_p90_hi_lo']]
    assert len(v) == len(FEATURE_NAMES), "特征向量与 FEATURE_NAMES 长度不一致"
    return np.array(v, dtype=float)


def extract_features(
    x: np.ndarray, fs: float, analyzer: HeartSoundAnalyzer | None = None
) -> np.ndarray:
    """对一段波形提特征向量。"""
    an = analyzer or HeartSoundAnalyzer()
    return feature_vector(an.analyze(x, fs))


class HeartSoundClassifier:
    """正常/异常二分类器(HistGradientBoosting,原生支持 NaN)。"""

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier
        # class_weight="balanced":PhysioNet/CinC 2016 正常:异常约 4:1,
        # 不加权时"全判正常"就能拿到约 0.8 的准确率,模型会退化到多数类。
        params = dict(max_iter=300, learning_rate=0.08, max_depth=4,
                      l2_regularization=1.0, class_weight="balanced")
        params.update(kwargs)
        self.model = HistGradientBoostingClassifier(**params)
        self.feature_names = list(FEATURE_NAMES)

    def fit(self, X, y):
        self.model.fit(np.asarray(X, float), np.asarray(y))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, float))

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, float))[:, 1]

    def cross_validate(self, X, y, cv: int = 5, groups=None) -> dict:
        """K 折交叉验证,返回多项指标的均值±标准差。

        groups
        ------
        给出每条录音所属的**受试者**时,改用 StratifiedGroupKFold,保证同一
        受试者的多条录音不会被拆到训练集与测试集两边。按录音随机切分会造成
        受试者级泄漏,把指标推高。

        指标
        ----
        不平衡数据下 accuracy 会误导(4:1 时全判正常即得 0.8),因此同时给出
        balanced_accuracy、AUC,以及敏感度(查出异常的比例)与特异度。筛查
        场景真正关心的是敏感度。
        """
        from sklearn.metrics import make_scorer, recall_score
        from sklearn.model_selection import (
            StratifiedGroupKFold, StratifiedKFold, cross_validate as sk_cv)
        X, y = np.asarray(X, float), np.asarray(y)
        if groups is None:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        else:
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True,
                                            random_state=0)
        scoring = {
            "acc": "accuracy",
            "bacc": "balanced_accuracy",
            "auc": "roc_auc",
            "sens": make_scorer(recall_score, pos_label=1),   # 敏感度
            "spec": make_scorer(recall_score, pos_label=0),   # 特异度
        }
        r = sk_cv(self.model, X, y, cv=splitter, groups=groups, scoring=scoring)
        out = {"grouped": groups is not None}
        for k in scoring:
            v = r["test_" + k]
            out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
        return out

    def feature_importance(
        self, X, y, n_repeats: int = 10, test_size: float = 0.3
    ) -> list[tuple[str, float]]:
        """置换重要性,看哪些特征在驱动决策(可解释性)。

        在**留出集**上计算。原实现先 fit 再在同一份训练数据上算置换重要性,
        报的是训练集重要性,会系统性高估(尤其对树模型)。
        """
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        X, y = np.asarray(X, float), np.asarray(y)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=0)
        self.fit(Xtr, ytr)
        r = permutation_importance(self.model, Xte, yte, n_repeats=n_repeats,
                                   random_state=0)
        order = np.argsort(r.importances_mean)[::-1]
        return [(self.feature_names[i], float(r.importances_mean[i]))
                for i in order]

    def save(self, path: str):
        import joblib
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str) -> "HeartSoundClassifier":
        import joblib
        obj = cls.__new__(cls)
        d = joblib.load(path)
        obj.model = d["model"]
        obj.feature_names = d["features"]
        return obj


# ---------------------------------------------------------------------------
# 真实数据:PhysioNet/CinC 2016 加载接口
# ---------------------------------------------------------------------------
def load_physionet2016(
    directory: str, analyzer=None, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """读取 PhysioNet/CinC 2016 格式目录,返回 (特征矩阵 X, 标签 y, 文件名列表)。

    约定:目录下含若干 *.wav 与一个 REFERENCE.csv(每行: 文件名(无扩展),标签),
    标签 -1=正常 / 1=异常,统一映射为 0/1。需要 `pip install soundfile`。

    ⚠️ 关于受试者分组
    -----------------
    返回文件名是为了让调用方自己构造 `groups` 传给 `cross_validate`。
    CinC 2016 的 REFERENCE.csv **不含受试者 ID**,文件名前缀(a-f)只是来源
    子库而非个体;若该子库中同一受试者贡献了多条录音,按录音随机切分就会
    造成受试者级泄漏、把指标推高。除非你能从数据集附带的元数据里还原受试者
    归属,否则应把交叉验证结果理解为**乐观上界**。
    """
    import csv
    import os
    import soundfile as sf

    an = analyzer or HeartSoundAnalyzer()
    ref = os.path.join(directory, "REFERENCE.csv")
    X, y, names = [], [], []
    skipped = 0
    with open(ref, newline="") as fh:
        rows = [r for r in csv.reader(fh) if len(r) >= 2]
    for k, row in enumerate(rows):
        name, label = row[0].strip(), row[1].strip()
        wav = os.path.join(directory, name + ".wav")
        if not os.path.exists(wav):
            skipped += 1
            continue
        try:
            sig, fs = sf.read(wav)
        except Exception as e:                       # noqa: BLE001
            print(f"  跳过 {name}: {type(e).__name__}: {e}")
            skipped += 1
            continue
        if sig.ndim > 1:
            sig = sig[:, 0]
        X.append(feature_vector(an.analyze(sig, fs)))
        y.append(1 if label in ("1", "abnormal") else 0)
        names.append(name)
        if verbose and (k + 1) % 200 == 0:
            print(f"  已处理 {k + 1}/{len(rows)} ...", flush=True)
    if skipped:
        print(f"  跳过 {skipped} 条(文件缺失或无法读取)")
    return np.array(X, float), np.array(y, int), names


# ---------------------------------------------------------------------------
# 合成数据集(无外部数据时验证整条 ML 管线)
# ---------------------------------------------------------------------------
def _synthetic_dataset(n_per_class=40, fs=2000.0, contaminate=True):
    """生成正常 vs 异常(杂音/S3)合成数据并提特征。返回 X, y。

    contaminate=True 时,正常类里混入 1/3 的"**噪声大但健康**"样本。

    这是刻意加的难度。原合成集的两类可以被单个 `murmur_sys` 阈值几乎完美
    分开(实测单阈值卡 0.15 即得 0.938,而 24 特征全模型 0.963),于是
    "5 折交叉验证准确率 0.96 / AUC 0.98"衡量的其实是**合成器把两类分得多开**,
    而不是模型学到了什么。混入高噪声正常样本后,模型必须学会用
    murmur_asymmetry 把"噪声"和"杂音"区分开,指标才有说明力——这也正是
    真实采集中最主要的假阳性来源。
    """
    an = HeartSoundAnalyzer()
    X, y = [], []
    shapes = ["plateau", "diamond", "crescendo", "decrescendo"]
    for i in range(n_per_class):
        bpm = 55 + (i * 37) % 45                  # 55..99,确定性铺开
        # 正常:1/3 为高噪声但健康(最常见的假阳性来源)
        noisy = contaminate and i % 3 == 2
        xn, _ = synthesize(bpm=bpm, secs=12,
                           noise=(0.18 + 0.04 * (i % 4)) if noisy
                           else (0.04 + 0.02 * (i % 3)),
                           seed=1000 + i)
        X.append(feature_vector(an.analyze(xn, fs)))
        y.append(0)
        # 异常:多为杂音,部分叠加 S3;1/4 为舒张期杂音
        xm, _ = synthesize(bpm=bpm, secs=12, noise=0.04,
                           murmur=0.35 + 0.05 * (i % 6),
                           murmur_shape=shapes[i % 4],
                           murmur_phase="diastolic" if i % 4 == 3 else "systolic",
                           s3=0.4 if i % 3 == 0 else 0.0,
                           seed=5000 + i)
        X.append(feature_vector(an.analyze(xm, fs)))
        y.append(1)
    return np.array(X, float), np.array(y, int)


def threshold_baseline(X, y, feature: str = "murmur_sys") -> dict:
    """单特征阈值基线:扫描该特征的所有阈值,取最佳准确率。

    任何 ML 模型都应该先跟它比。若树模型只比单个手调阈值高几个百分点,那么
    报出来的高指标衡量的是**数据集本身有多好分**,而不是模型的贡献。
    """
    i = FEATURE_NAMES.index(feature)
    v, y = np.asarray(X, float)[:, i], np.asarray(y)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"feature": feature, "acc": float("nan"), "thr": float("nan")}
    best = (0.0, float("nan"))
    for thr in np.unique(finite):
        pred = (v > thr).astype(int)          # NaN > thr -> False,即判正常
        acc = float((pred == y).mean())
        if acc > best[0]:
            best = (acc, float(thr))
    return {"feature": feature, "acc": best[0], "thr": best[1]}


if __name__ == "__main__":
    import sys

    try:   # Windows 控制台默认 GBK;stderr 也要一起切,否则报错信息乱码
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if "--demo" in sys.argv:
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

        # 诚实对照:模型到底比"单个手调阈值"强多少
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
        print("\n(真实数据请用 load_physionet2016(目录) 替换合成集)")

    elif "--train-demo" in sys.argv:
        # 方案A:在合成数据上训练并保存"加载即用"的 demo 模型
        out = "model_demo.joblib"
        X, y = _synthetic_dataset(n_per_class=60)
        clf = HeartSoundClassifier().fit(X, y)
        clf.save(out)
        print(f"✅ 已保存 demo 模型 -> {out}  (合成数据训练,仅验证流程)")
        print("   用法: python predict.py <某段录音.wav>")

    elif "--train-physionet" in sys.argv:
        # 方案B:在真实 PhysioNet/CinC 2016 数据上训练
        i = sys.argv.index("--train-physionet")
        data_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        if not data_dir:
            print("用法: python classifier.py --train-physionet <数据目录>")
            sys.exit(1)
        print(f"读取真实数据 {data_dir} ...")
        X, y, names = load_physionet2016(data_dir)
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

    else:
        print("用法:")
        print("  python classifier.py --demo            # 合成数据交叉验证演示")
        print("  python classifier.py --train-demo      # 训练并保存 demo 模型")
        print("  python classifier.py --train-physionet <目录>  # 真实数据训练")
        print("库: from classifier import HeartSoundClassifier, "
              "extract_features, load_physionet2016")
