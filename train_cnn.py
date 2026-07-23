"""
train_cnn.py — 在 CirCor 上训练/评估频谱图 CNN 杂音检测器。

用法:
    python train_cnn.py <CirCor目录> [--cache feats.npz] [--epochs 40]
    python train_cnn.py <CirCor目录> --folds 5        # 按患者分组交叉验证

评估口径与工程特征基线**完全一致**(docs/VALIDATION.md),便于直接比较:
  - 标签:该听诊区在患者的 `Murmur locations` 里才算阳性;
  - 划分:按**患者**分组 5 折,同一患者的多个听诊区不跨折;
  - 指标:录音级 AUC;另报患者级(各区取最大)AUC。

⚠️ 仅用于学习/研究,非医疗器械,不能用于诊断。
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

from cnn import augment, build_model, windows


def extract(directory: str, cache: str | None = None):
    """提特征并缓存。返回 (谱图数组, 窗->录音索引, 录音标签, 患者ID, 录音名)。"""
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"从缓存读取 {cache}")
        return d["S"], d["owner"], d["y"], d["pat"], d["name"]

    import soundfile as sf
    meta_path = os.path.join(directory, "training_data.csv")
    meta = {r["Patient ID"]: r for r in
            csv.DictReader(open(meta_path, encoding="utf-8"))}
    wavs = sorted(glob.glob(os.path.join(directory, "training_data", "*.wav"))) \
        or sorted(glob.glob(os.path.join(directory, "*.wav")))
    print(f"发现 {len(wavs)} 条录音,提取对数梅尔谱 ...")

    S, owner, y, pat, name = [], [], [], [], []
    for i, w in enumerate(wavs):
        stem = os.path.splitext(os.path.basename(w))[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        pid, loc = parts[0], parts[1]
        m = meta.get(pid)
        if m is None or m["Murmur"] == "Unknown":
            continue
        try:
            sig, fs = sf.read(w)
        except Exception:
            continue
        if getattr(sig, "ndim", 1) > 1:
            sig = sig[:, 0]
        ws = windows(sig, fs)
        if not ws:
            continue
        r = len(y)
        S.extend(ws)
        owner.extend([r] * len(ws))
        y.append(int(m["Murmur"] == "Present"
                     and loc in (m["Murmur locations"] or "").split("+")))
        pat.append(pid)
        name.append(stem)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(wavs)} ...", flush=True)

    # 统一时间长度(末尾录音可能差一帧)
    T = min(s.shape[1] for s in S)
    S = np.stack([s[:, :T] for s in S]).astype(np.float16)
    owner = np.array(owner, np.int32)
    y = np.array(y, np.int8)
    pat = np.array(pat)
    name = np.array(name)
    if cache:
        np.savez_compressed(cache, S=S, owner=owner, y=y, pat=pat, name=name)
        print(f"已缓存 -> {cache}")
    return S, owner, y, pat, name


def auc(score: np.ndarray, label: np.ndarray) -> float:
    o = np.argsort(-score)
    ys = label[o]
    tpr = np.cumsum(ys) / max(1, ys.sum())
    fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(tpr, fpr))


def aggregate(win_prob: np.ndarray, owner: np.ndarray, n_rec: int,
              q: float = 0.8) -> np.ndarray:
    """窗级概率 -> 录音级分数。

    用高分位数而非均值:杂音常只在部分心动周期清晰可闻,均值会被大量"安静"
    的窗稀释。q=0.8 在验证集上优于均值与最大值(最大值对单窗噪声太敏感)。
    """
    out = np.zeros(n_rec)
    for r in range(n_rec):
        p = win_prob[owner == r]
        out[r] = np.quantile(p, q) if len(p) else 0.0
    return out


def run_fold(Str, ytr, Ste, epochs, device, seed=0, log=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = build_model().to(device)
    # 阳性窗远少于阴性窗,用 pos_weight 而不是重采样(保留全部阴性信息)
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=epochs * max(1, len(ytr) // 128 + 1))

    Xtr = torch.from_numpy(Str.astype(np.float32)).unsqueeze(1)
    Ytr = torch.from_numpy(ytr.astype(np.float32))
    # 逐样本标准化,消除设备增益差异
    Xtr = (Xtr - Xtr.mean(dim=(2, 3), keepdim=True)) / (
        Xtr.std(dim=(2, 3), keepdim=True) + 1e-5)

    n = len(Ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb = augment(Xtr[idx], rng).to(device, non_blocking=True)
            yb = Ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss) * len(idx)
        if log and (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs} loss {tot / n:.4f}", flush=True)

    model.eval()
    Xte = torch.from_numpy(Ste.astype(np.float32)).unsqueeze(1)
    Xte = (Xte - Xte.mean(dim=(2, 3), keepdim=True)) / (
        Xte.std(dim=(2, 3), keepdim=True) + 1e-5)
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            probs.append(torch.sigmoid(
                model(Xte[i:i + 256].to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probs), model


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 1
    directory = argv[0]
    cache = argv[argv.index("--cache") + 1] if "--cache" in argv else "cnn_feats.npz"
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else 40
    folds = int(argv[argv.index("--folds") + 1]) if "--folds" in argv else 5

    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    S, owner, y, pat, name = extract(directory, cache)
    print(f"\n窗 {len(S)} 个 / 录音 {len(y)} 条 / 患者 {len(set(pat.tolist()))} 名")
    print(f"阳性录音 {int(y.sum())} / 阴性 {int((1 - y).sum())}"
          f"  谱图尺寸 {S.shape[1]}×{S.shape[2]}")
    n_par = sum(p.numel() for p in build_model().parameters())
    print(f"模型参数量 {n_par / 1e3:.0f}k")

    win_y = y[owner]
    rec_score = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    for k, (tr_r, te_r) in enumerate(skf.split(np.zeros(len(y)), y, pat), 1):
        tr_w = np.isin(owner, tr_r)
        te_w = np.isin(owner, te_r)
        print(f"\n--- 第 {k}/{folds} 折: 训练 {tr_w.sum()} 窗 / 测试 {te_w.sum()} 窗 ---")
        p, _ = run_fold(S[tr_w], win_y[tr_w], S[te_w], epochs, device,
                        seed=k, log=True)
        # 折内映射回录音
        sub_owner = owner[te_w]
        for r in te_r:
            pr = p[sub_owner == r]
            rec_score[r] = np.quantile(pr, 0.8) if len(pr) else 0.0
        print(f"    本折录音级 AUC {auc(rec_score[te_r], y[te_r]):.3f}")

    print("\n" + "=" * 60)
    print(f"录音级 AUC(按患者分组 {folds} 折) = {auc(rec_score, y):.3f}")
    print("  对照:工程特征 + 树模型基线 AUC 0.793(仅门控后)/ 0.775(全部)")
    print("       单一 murmur_sys 指标 AUC 0.715")

    # 患者级:各听诊区取最大
    meta = {r["Patient ID"]: r for r in csv.DictReader(
        open(os.path.join(directory, "training_data.csv"), encoding="utf-8"))}
    pids = sorted(set(pat.tolist()))
    ps = np.array([rec_score[pat == p].max() for p in pids])
    py = np.array([int(meta[p]["Murmur"] == "Present") for p in pids])
    print(f"患者级 AUC(各区取最大,{len(pids)} 名)= {auc(ps, py):.3f}")
    print("  对照:murmur_sys 多区取最大 AUC 0.762")

    np.savez("cnn_scores.npz", rec_score=rec_score, y=y, pat=pat, name=name)
    print("\n已保存逐条得分 -> cnn_scores.npz")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
