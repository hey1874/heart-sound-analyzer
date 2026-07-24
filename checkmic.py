"""
checkmic.py — 采集链路体检:在录心音之前,先确认这条链路没问题。

为什么需要
----------
「听诊器 → 咪头 → 3.5mm → 声卡」这条链路上有几个坑**不会报错**,只会让你
拿到"能出数但全是错的"结果:

  1. 系统的降噪/增强(macOS「环境降噪」、Windows「麦克风增强」)是为人声
     设计的,通常在 100Hz 以上做高通——**正好砍掉 S1/S2 的主能量**;AGC 还
     会让幅度失去意义。
  2. 不少声卡/耳麦口本身就在 50-100Hz 以下高通,心音基频 20-150Hz 首当其冲。
  3. 增益过大导致削波,或过小导致量化噪声淹没信号。
  4. 工频干扰(50/60Hz)落在心音带内,肉眼看波形看不出来。

本工具录一段环境声(不用贴听诊器),从**频谱形状**反推这些问题。

用法:
    python checkmic.py                 # 用默认输入设备录 5 秒
    python checkmic.py --device 2      # 指定设备
    python checkmic.py --list          # 列出所有输入设备
    python checkmic.py --seconds 10

⚠️ 这是链路体检,不是听诊。结论基于环境底噪的频谱,只能发现"明显被砍掉"
   一类的问题;要精确标定频响需要已知声源(扫频/校准声级计)。
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import signal


def band_profile(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (频率, 功率谱密度),用 Welch。"""
    nper = int(min(len(x), max(1024, fs // 2)))
    return signal.welch(x, fs=fs, nperseg=nper)


def diagnose(x: np.ndarray, fs: float) -> list[tuple[str, str, str]]:
    """返回 [(等级, 项目, 说明)],等级 ∈ {ok, warn, bad}。"""
    out = []
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    # --- 电平 ---
    if peak <= 1e-6:
        out.append(("bad", "输入电平", "几乎没有信号。检查是否插到了正确的口、"
                                      "四段式(TRRS)接法是否正确、系统是否选对输入设备"))
        return out
    dbfs = 20 * np.log10(peak + 1e-12)
    if peak >= 0.99:
        out.append(("bad", "输入电平", f"峰值 {dbfs:.1f} dBFS,已削波。调低输入增益"))
    elif dbfs < -60:
        out.append(("warn", "输入电平", f"峰值 {dbfs:.1f} dBFS,偏低。"
                                       "调高增益,否则量化噪声会淹没心音"))
    else:
        out.append(("ok", "输入电平", f"峰值 {dbfs:.1f} dBFS,动态范围合适"))

    f, p = band_profile(x, fs)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(p[m].mean()) if m.any() else np.nan

    # --- 低频是否被高通砍掉 ---
    # 环境底噪在低频通常更强(1/f)。若 20-60Hz 明显低于 200-400Hz,
    # 说明输入链路做了高通——这正好砍在心音主能量上。
    lo_b, mid_b, hi_b = band(20, 60), band(60, 150), band(200, 400)
    if np.isfinite(lo_b) and np.isfinite(hi_b) and hi_b > 0:
        ratio_db = 10 * np.log10((lo_b + 1e-20) / (hi_b + 1e-20))
        if ratio_db < -6:
            out.append(("bad", "低频响应",
                        f"20-60Hz 比 200-400Hz 低 {-ratio_db:.0f} dB,"
                        "链路存在高通。S1/S2 主能量在 20-150Hz,会被砍掉。\n"
                        "        → 关掉系统的降噪/麦克风增强;换低频平直的声卡"))
        elif ratio_db < 0:
            out.append(("warn", "低频响应",
                        f"20-60Hz 比 200-400Hz 低 {-ratio_db:.0f} dB,低频略有衰减"))
        else:
            out.append(("ok", "低频响应",
                        f"20-60Hz 相对 200-400Hz 高 {ratio_db:.0f} dB,"
                        "符合环境底噪的 1/f 特征,未见明显高通"))
    if np.isfinite(mid_b) and np.isfinite(lo_b) and lo_b > 0:
        pass

    # --- 工频 ---
    from heartbeat import HeartSoundAnalyzer
    an = HeartSoundAnalyzer()
    xr = an.resample(x, fs)
    fq, pq = signal.welch(xr, fs=an.fs, nperseg=min(len(xr), 2048))
    hum = an.hum_ratio(fq, pq)
    if hum > 0.5:
        out.append(("bad", "工频干扰", f"50/60Hz 线谱占带内能量 {hum:.0%}。"
                                      "检查屏蔽与接地,或改用电池供电的接口"))
    elif hum > 0.2:
        out.append(("warn", "工频干扰", f"50/60Hz 占 {hum:.0%},偏高但尚可"))
    else:
        out.append(("ok", "工频干扰", f"50/60Hz 占 {hum:.0%},干净"))

    # --- AGC 迹象:分段 RMS 被压平 ---
    #
    # ⚠️ 这项判据本身不可靠,不要据此下结论。AGC 会把各段响度压平,但**高通
    # 滤波和强工频同样会**(实测:正常底噪 CV 0.065,高通后 0.033,加 50Hz
    # 工频后 0.020,而完美 AGC 是 0.000——三者区间重叠)。所以这里只在工频
    # 干净时才给提示,且措辞降级为"疑似",真正可靠的办法是直接去看系统设置。
    n = int(0.5 * fs)
    if len(x) >= 6 * n:
        seg_rms = np.array([np.sqrt(np.mean(x[i:i + n] ** 2))
                            for i in range(0, len(x) - n, n)])
        seg_rms = seg_rms[seg_rms > 0]
        if len(seg_rms) >= 4:
            cv = float(seg_rms.std() / (seg_rms.mean() + 1e-12))
            if hum > 0.2:
                out.append(("ok", "自动增益(AGC)",
                            "工频偏强,响度变异不可用于判断 AGC,已跳过"))
            elif cv < 0.02:
                out.append(("warn", "自动增益(AGC)",
                            f"各 0.5 秒段响度几乎一致(变异 {cv:.3f}),疑似 AGC。\n"
                            "        ⚠️ 该判据不可靠(高通滤波也会造成同样现象),"
                            "请直接去系统设置确认"))
            else:
                out.append(("ok", "自动增益(AGC)",
                            f"响度随时间正常起伏(变异 {cv:.2f}),未见明显 AGC"))

    # --- 底噪绝对水平 ---
    out.append(("ok" if rms > 1e-5 else "warn", "底噪 RMS",
                f"{20 * np.log10(rms + 1e-12):.1f} dBFS"))
    return out


def main(argv: list[str]) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("需要 sounddevice:pip install sounddevice")
        return 1

    if "--list" in argv:
        print(sd.query_devices())
        return 0

    device = None
    if "--device" in argv:
        device = argv[argv.index("--device") + 1]
        device = int(device) if device.isdigit() else device
    secs = float(argv[argv.index("--seconds") + 1]) if "--seconds" in argv else 5.0

    info = sd.query_devices(device, "input")
    fs = int(info["default_samplerate"])
    print(f"设备: {info['name']}   采样率 {fs} Hz   录音 {secs:.0f} 秒")
    print("保持安静,**不要贴听诊器**——本检查看的是链路本身,不是心音。\n")
    rec = sd.rec(int(secs * fs), samplerate=fs, channels=1, dtype="float32",
                 device=device)
    sd.wait()
    x = rec[:, 0].astype(np.float64)

    icon = {"ok": "✅", "warn": "⚠️ ", "bad": "❌"}
    results = diagnose(x, fs)
    for level, name, msg in results:
        print(f"{icon[level]} {name}: {msg}")

    bad = sum(1 for lv, _, _ in results if lv == "bad")
    warn = sum(1 for lv, _, _ in results if lv == "warn")
    print()
    if bad:
        print(f"❌ {bad} 项不合格。先解决这些再录心音,否则拿到的数据是错的。")
    elif warn:
        print(f"⚠️  {warn} 项需注意,但可以先试录。")
    else:
        print("✅ 链路检查通过,可以开始录心音了:python heartbeat.py")
    print("\n注:本检查基于环境底噪的频谱形状,只能发现明显的高通/工频/AGC/削波。"
          "\n精确标定频响需要已知声源(扫频或校准声级计)。")
    return 2 if bad else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
