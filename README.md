# 心音听诊分析 / Heart Sound Analyzer

通过(接听诊器的)麦克风实时采集心音,估计**心率(BPM)**、评估**节律规则性**、
给出**杂音能量提示**,并在分析前做**信号质量门控(SQI)**与**HSMM 心音分段**。

> ⚠️ **医疗免责**:本项目仅用于学习/研究/技术演示,**不是医疗器械**,
> 输出**不构成任何医学诊断**。如有健康问题请就医。详见 [LICENSE](LICENSE)。

---

## 功能

| 能力 | 说明 |
|---|---|
| **心率 BPM** | 包络自相关估计 + 抛物线插值,带 0~1 置信度 |
| **信号质量 SQI** | 周期性 / 带内能量占比 / 削波三项综合,坏段自动门控 |
| **HSMM 分段** | 时长约束的 S1→收缩→S2→舒张 4 状态分段(Springer 2016 轻量无监督版) |
| **节律分析** | RR 变异系数 CV、Poincaré SD1/SD2、标准 HRV(SDNN/RMSSD/pNN50)、早搏计数 |
| **STI 收缩间期** | 收缩/舒张期时长与比值(≈LVET 近似) |
| **杂音提示** | 能量指数 + 形状(钻石/平台/递增/递减)与时相(早/中/晚/全收缩期) |
| **实时运行** | 环形缓冲 + 周期分析,终端实时滚动输出 |
| **离线自测** | 内置合成心音,无需麦克风即可验证算法正确性 |

## 安装

```bash
pip install -r requirements.txt
```

核心算法仅需 `numpy` + `scipy`;`sounddevice` 只在实时采集时需要。

## 快速开始

**离线自测**(无需硬件,验证算法):

```bash
python heartbeat.py --selftest
```

预期输出(8 项全绿):

```
[1] 心率: 真值 72.0 -> 估计 72.0 BPM (置信度 0.94)  ✅
[2] 规则节律: CV=0.012 -> 规则  ✅
[3] 不规则节律: CV=0.100 -> 轻度不齐...  ✅
[4] 杂音指数: 正常=0.021  含杂音=1.729  ✅
[5] HSMM分段: 检出 15 个 S1, 相位误差 12ms  ✅
[6] SQI门控: 干净 SQI=0.87(ok=True)  噪声 SQI=0.21(ok=False)  ✅
[7] STI: 收缩期≈300ms 舒张期≈530ms 比值=0.57  ✅
[8] 杂音形状: 注入钻石型 -> 判定「递增-递减/钻石型...」(中收缩期)  ✅
```

**实时采集**(需要麦克风+听诊器):

```bash
python heartbeat.py
```

终端会实时滚动:

```
[████████████████░░░░]  72.3 BPM   置信度 0.81   窗内心音 14   节律:规则
```

## 作为库调用

核心算法 `HeartSoundAnalyzer` 与采集完全解耦,可处理任意波形:

```python
import soundfile as sf                    # 读音频文件需额外:pip install soundfile
from heartbeat import HeartSoundAnalyzer

x, fs = sf.read("recording.wav")          # 任意采样率
res = HeartSoundAnalyzer().analyze(x, fs)

print(res["bpm"], res["confidence"])      # 心率 + 置信度
print(res["sqi"])                         # 信号质量
print(res["rhythm"])                      # 节律指标
print(res["murmur"])                      # 杂音提示
```

## 处理管线

```
重采样(→2000Hz) → 带通(20–200Hz) → 香农能量包络 → 自相关估心率
                                                      │
            ┌── SQI 信号质量门控 ←──────────────────────┤
            ├── HSMM 时长约束分段(S1/S2)─→ 节律分析(RR/Poincaré)
            └── 高频带(150–600Hz)能量 ──────────────→ 杂音提示
```

算法细节见 [docs/ALGORITHM.md](docs/ALGORITHM.md)。

## 硬件

本项目针对**听诊器 + 驻极体咪头(如松下 WM-61A)+ 声卡**链路设计。
低频(20–150Hz)是心音主能量区,**声卡的低频高通是最大风险点**。
接线、供电、采样建议见 [docs/HARDWARE.md](docs/HARDWARE.md)。

## 走向"接近临床"的路线

当前是**筛查级原型**。要逼近临床医生的听诊效果,核心是
**准确分段 + 大型标注心音库上的 ML 模型**(PhysioNet CirCor 2022 / 2016)。
完整路线、医学依据与各阶段工作量见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 目录

```
heartbeat.py          核心算法 + 实时采集 + 自测(单文件,无第三方训练依赖)
docs/ALGORITHM.md     各步骤算法与医学依据
docs/HARDWARE.md      听诊器/WM-61A/声卡 接入与避坑
docs/ROADMAP.md       走向准临床的分阶段路线(SQI→HSMM→临床特征→ML)
requirements.txt      依赖
LICENSE               MIT + 医疗免责声明
```

## 许可

[MIT](LICENSE)。**非医疗用途。**
