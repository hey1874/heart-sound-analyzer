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

from heartbeat import HeartSoundAnalyzer, _synthesize

# 固定的特征顺序(树模型可吃 NaN,缺失填 NaN 即可)
FEATURE_NAMES = [
    "bpm", "confidence",
    "sqi", "periodicity", "band_ratio", "clip",
    "cv", "sd1_ms", "sd2_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "ectopic",
    "systole_ms", "diastole_ms", "sys_dia_ratio",
    "murmur_sys", "murmur_dia", "murmur_occupancy", "murmur_peak_pos",
    "s3", "s4", "s2_split_ms", "s2_split_frac",
]


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
        _g(res, "sqi", "band_ratio"), _g(res, "sqi", "clip"),
        _g(res, "rhythm", "cv"), _g(res, "rhythm", "sd1_ms"),
        _g(res, "rhythm", "sd2_ms"), _g(res, "rhythm", "sdnn_ms"),
        _g(res, "rhythm", "rmssd_ms"), _g(res, "rhythm", "pnn50_pct"),
        _g(res, "rhythm", "ectopic"),
        _g(res, "sti", "systole_ms"), _g(res, "sti", "diastole_ms"),
        _g(res, "sti", "sys_dia_ratio"),
        _g(res, "murmur", "systolic"), _g(res, "murmur", "diastolic"),
        _g(res, "murmur", "occupancy"), _g(res, "murmur", "peak_pos"),
        _g(res, "extra_sounds", "s3"), _g(res, "extra_sounds", "s4"),
        _g(res, "s2_split", "split_ms"), _g(res, "s2_split", "split_frac"),
    ]
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
        params = dict(max_iter=300, learning_rate=0.08,
                      max_depth=4, l2_regularization=1.0)
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

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """分层 K 折交叉验证,返回 accuracy 与 roc_auc 的均值±标准差。"""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        X, y = np.asarray(X, float), np.asarray(y)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        acc = cross_val_score(self.model, X, y, cv=skf, scoring="accuracy")
        auc = cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc")
        return {"acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
                "auc_mean": float(auc.mean()), "auc_std": float(auc.std())}

    def feature_importance(self, X, y, n_repeats: int = 10) -> list[tuple[str, float]]:
        """置换重要性,看哪些特征在驱动决策(可解释性)。"""
        from sklearn.inspection import permutation_importance
        self.fit(X, y)
        r = permutation_importance(self.model, np.asarray(X, float),
                                   np.asarray(y), n_repeats=n_repeats,
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
def load_physionet2016(directory: str, analyzer=None) -> tuple[np.ndarray, np.ndarray]:
    """读取 PhysioNet/CinC 2016 格式目录,返回 (特征矩阵 X, 标签 y)。

    约定:目录下含若干 *.wav 与一个 REFERENCE.csv(每行: 文件名(无扩展),标签),
    标签 -1=正常 / 1=异常,统一映射为 0/1。需要 `pip install soundfile`。
    """
    import csv
    import os
    import soundfile as sf

    an = analyzer or HeartSoundAnalyzer()
    ref = os.path.join(directory, "REFERENCE.csv")
    X, y = [], []
    with open(ref, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            name, label = row[0].strip(), row[1].strip()
            wav = os.path.join(directory, name + ".wav")
            if not os.path.exists(wav):
                continue
            sig, fs = sf.read(wav)
            if sig.ndim > 1:
                sig = sig[:, 0]
            X.append(feature_vector(an.analyze(sig, fs)))
            y.append(1 if label in ("1", "abnormal") else 0)
    return np.array(X, float), np.array(y, int)


# ---------------------------------------------------------------------------
# 合成数据集(无外部数据时验证整条 ML 管线)
# ---------------------------------------------------------------------------
def _synthetic_dataset(n_per_class=40, fs=2000.0):
    """生成正常 vs 异常(杂音/S3)合成数据并提特征。返回 X, y。"""
    an = HeartSoundAnalyzer()
    X, y = [], []
    shapes = ["plateau", "diamond", "crescendo", "decrescendo"]
    for i in range(n_per_class):
        bpm = 55 + (i * 37) % 45                  # 55..99,确定性铺开
        # 正常
        xn, _ = _synthesize(bpm=bpm, secs=12, noise=0.04 + 0.02 * (i % 3),
                            seed=1000 + i)
        X.append(feature_vector(an.analyze(xn, fs)))
        y.append(0)
        # 异常:多为杂音,部分叠加 S3
        xm, _ = _synthesize(bpm=bpm, secs=12, noise=0.04,
                            murmur=0.35 + 0.05 * (i % 6),
                            murmur_shape=shapes[i % 4],
                            s3=0.4 if i % 3 == 0 else 0.0,
                            seed=5000 + i)
        X.append(feature_vector(an.analyze(xm, fs)))
        y.append(1)
    return np.array(X, float), np.array(y, int)


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if "--demo" in sys.argv:
        print("合成数据集(正常 vs 杂音/S3),提取 DSP+临床特征 ...")
        X, y = _synthetic_dataset(n_per_class=40)
        print(f"样本 {len(y)}  特征 {X.shape[1]}  "
              f"(正常 {int((y == 0).sum())} / 异常 {int((y == 1).sum())})")

        clf = HeartSoundClassifier()
        cv = clf.cross_validate(X, y, cv=5)
        print(f"5 折交叉验证: 准确率 {cv['acc_mean']:.3f}±{cv['acc_std']:.3f}  "
              f"AUC {cv['auc_mean']:.3f}±{cv['auc_std']:.3f}  "
              + ("✅" if cv["acc_mean"] > 0.85 else "❌ 偏低"))

        print("\n驱动决策的 Top-6 特征(置换重要性):")
        for name, imp in clf.feature_importance(X, y)[:6]:
            print(f"  {name:18s} {imp:.3f}")
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
        X, y = load_physionet2016(data_dir)
        print(f"样本 {len(y)} (正常 {int((y == 0).sum())} / 异常 {int((y == 1).sum())})")
        clf = HeartSoundClassifier()
        cv = clf.cross_validate(X, y, cv=5)
        print(f"5 折交叉验证: 准确率 {cv['acc_mean']:.3f}  AUC {cv['auc_mean']:.3f}")
        clf.fit(X, y).save("model_physionet.joblib")
        print("✅ 已保存 -> model_physionet.joblib")

    else:
        print("用法:")
        print("  python classifier.py --demo            # 合成数据交叉验证演示")
        print("  python classifier.py --train-demo      # 训练并保存 demo 模型")
        print("  python classifier.py --train-physionet <目录>  # 真实数据训练")
        print("库: from classifier import HeartSoundClassifier, "
              "extract_features, load_physionet2016")
