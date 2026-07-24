"""
sites.py — 标准心脏听诊区。

为什么这不只是"给人看的说明"
----------------------------
杂音是**位置依赖**的:实测多听诊区取最大值能把患者级 AUC 从 0.701 提到
0.762(见 docs/VALIDATION.md 第 7 节),是投入产出比最高的一项改进,而且
不需要任何额外硬件——换个位置再录一次就行。

而这四个代号(AV / PV / TV / MV)**正是 CirCor 数据集的听诊区标签**。所以
录音时标上位置,采集到的数据才能直接喂给 `calibrate.py` 做标定;不标位置,
那条路就断了。

⚠️ 本模块只提供**解剖定位的参考描述**。工具无法判断你是否真的贴在了正确
   位置——它只能告诉你信号好不好。定位要靠解剖标志(胸骨角 = 第 2 肋间)
   自己确认。

⚠️ 示意图坐标按**正面视角**(面对受检者)给出,因此**受检者的左侧出现在图
   的右边**:主动脉瓣区在胸骨右缘,画在图的左侧。这一点必须在图上明确标出
   ——否则有人会当成"这是我自己"而把左右完全贴反。以文字定位为准。

底图
----
`thorax.png` 裁自 Wikimedia Commons 的
"Heart sounds auscultation areas.svg"(Vinne2 作,基于 Gray's Anatomy 1918 图版
Gray112),**公有领域**。本项目取其未加标记的胸廓版画作底图,提亮并裁剪,
听诊点由本项目按解剖标志重新标定。

来源
----
- 心脏听诊区_百度百科 / 《物理诊断学》
- Kenhub, Heart auscultation: Anatomy and technique
- Oliveira J. et al., The CirCor DigiScope Dataset, IEEE JBHI 2022(标签体系)
"""

from __future__ import annotations

# code 与 CirCor 的听诊区标签一致,便于直接对接 calibrate.py
SITES = [
    {
        "code": "AV",
        "name": "主动脉瓣区",
        "en": "Aortic",
        "landmark": "胸骨右缘第 2 肋间",
        "listen_for": "主动脉瓣狭窄的喷射性收缩期杂音(向颈部放射)",
        # 归一化坐标(相对 thorax.png)。**标定到图上真实的肋骨位置**:
        # 在原版画上量得胸骨角(第 2 肋)y=110、相邻肋附着点间距 45px、
        # 胸骨右缘 x=175 / 左缘 x=262 / 锁骨中线 x=315,第 n 肋间取第 n 与
        # n+1 肋之中点,再按裁剪与缩放换算。正面视角:x 向右 = 受检者左侧
        "xy": (0.3872, 0.2814),
        "ics": 2, "side": "R",
    },
    {
        "code": "PV",
        "name": "肺动脉瓣区",
        "en": "Pulmonic",
        "landmark": "胸骨左缘第 2 肋间",
        "listen_for": "肺动脉瓣病变;S2 分裂在此最清楚",
        "xy": (0.6103, 0.2814),
        "ics": 2, "side": "L",
    },
    {
        "code": "ERB",
        "name": "主动脉瓣第二区",
        "en": "Erb",
        "landmark": "胸骨左缘第 3 肋间",
        "listen_for": "主动脉瓣反流的舒张期递减型杂音",
        "xy": (0.6103, 0.41),
        "ics": 3, "side": "L",
        "optional": True,
    },
    {
        "code": "TV",
        "name": "三尖瓣区",
        "en": "Tricuspid",
        "landmark": "胸骨左缘第 4–5 肋间",
        "listen_for": "三尖瓣反流、室间隔缺损;右心声音在此最强",
        "xy": (0.6103, 0.5386),
        "ics": 4, "side": "L",
    },
    {
        "code": "MV",
        "name": "二尖瓣区(心尖)",
        "en": "Mitral",
        "landmark": "第 5 肋间,左锁骨中线内 0.5–1 cm",
        "listen_for": "二尖瓣反流的全收缩期杂音(向腋下放射);S3 在此最清楚",
        "xy": (0.7462, 0.6671),
        "ics": 5, "side": "L",
    },
]

# 常规听诊顺序:二尖瓣 → 肺动脉瓣 → 主动脉瓣 → 主动脉瓣第二区 → 三尖瓣
ORDER = ["MV", "PV", "AV", "ERB", "TV"]

BY_CODE = {s["code"]: s for s in SITES}

# CirCor 只用这四个(不含 Erb),对接标定时以这四个为准
CIRCOR_CODES = ["AV", "PV", "TV", "MV"]

FIND_TIP = (
    "找位置的办法:先摸到**胸骨角**(胸骨上段的横向骨嵴),它平对第 2 肋,"
    "紧邻其下的肋间就是第 2 肋间;由此往下数即可。"
)

# 示意图是正面视角(面对受检者),受检者左侧在图的右边。必须在图上标出。
VIEW_NOTE = "示意图为正面视角(面对受检者):图的右边 = 受检者的左侧"
