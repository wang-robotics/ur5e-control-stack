"""生成可视化实验报告: 图表 (PNG) + Markdown 报告 (docs/实验报告.md)。

用法: python scripts/make_report.py
产物:
    docs/figures/*.png    4 张图
    docs/实验报告.md      带图报告 (含实验数据表与解读)
"""
import base64
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / "docs" / "实验报告.md"


def read_any(path: Path) -> str:
    """读日志文件 (PowerShell 重定向可能写 UTF-16, cmd 写 UTF-8, 自动识别)。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------
# 1. VLA 版本演进 (成功率口径 = 9cm 内停 1 秒)
#     v1~v2+增益: 历史小样本记录 (5~10 局); v3~v4.5: 2026-09-06 统一 50 局协议复测
# --------------------------------------------------------------------------
vla_versions = ["v1\n(1584帧)", "v2\n(5449帧+增强)", "v2+增益\n(×1.5)", "v3\n(加权+裁剪)", "v4\n(真值目标)", "v4.5\n(RGB-D检测)"]
vla_success = [20.0, 0.0, 37.5, 98.0, 100.0, 100.0]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(vla_versions, vla_success, color=["#8ab4f8", "#f28b82", "#fdd663", "#81c995"])
ax.bar_label(bars, labels=[f"{s}%" for s in vla_success])
ax.set_ylabel("成功率 (%)")
ax.set_title("VLA 行为克隆模型版本演进\n(成功率口径: 距球心 9cm 内停 1 秒; v3~v4.5 为 50 局统一复测)")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(FIGDIR / "vla_versions.png", dpi=120)
plt.close(fig)

# --------------------------------------------------------------------------
# 2. PPO vs VLA 对比 (双口径, 诚实标注阈值差异)
# --------------------------------------------------------------------------
# 评估数据来源 (2026-09-06 双口径复测, 存档见 outputs/eval_vla_*.txt):
#   PPO:  2cm 口径 150/150 (50 局 × 3 种子), 最终距离 1.86±0.09cm
#   VLA 2cm 口径: v4 (真值目标) 0/50, 最近 4.56±1.93cm ≈ 示教终点 4.6±2.4cm (回放测量)
#   VLA 9cm 口径: v4 50/50, v4.5 (修复版检测器) 50/50; 9cm 阈值吞掉检测误差, 分不出
#   → BC 上限 = 示教精度; 检测误差 (~5cm) 是第二层噪声
ppo_2cm, ppo_9cm = 100.0, 100.0          # PPO 2cm 达标 → 9cm 必然达标
vla_2cm, vla_9cm = 0.0, 100.0            # v4 真值 @2cm 0/50; 9cm 口径 50/50
vla_precision = 4.5                      # VLA 最近可达距离 (cm, 2cm 口径实测)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(2)
w = 0.36
b1 = ax1.bar(x - w / 2, [ppo_2cm, vla_2cm], w, label="2cm 口径 (PPO 标准)", color="#4285f4")
b2 = ax1.bar(x + w / 2, [ppo_9cm, vla_9cm], w, label="9cm 口径 (VLA 自评标准)", color="#81c995")
ax1.bar_label(b1, labels=["150/150\n(3种子)", "0/50\n(v4真值)"])
ax1.bar_label(b2, labels=["必然达标", "50/50"])
ax1.set_xticks(x, ["PPO 强化学习", "VLA 行为克隆"])
ax1.set_ylabel("成功率 (%)")
ax1.set_title("成功率: 两种口径对比\n(同一任务, 阈值不同不能直接比)")
ax1.set_ylim(0, 115)
ax1.legend(fontsize=9)

labels = ["PPO\n(2cm 口径)", "VLA v4\n(2cm 口径)"]
vals = [1.86, vla_precision]
bars = ax2.bar(labels, vals, color=["#4285f4", "#34a853"], width=0.5)
ax2.bar_label(bars, labels=["1.86±0.09", f"~{vla_precision}±1.9"])
ax2.axhline(2.0, color="#e53935", linestyle="--", linewidth=1.2)
ax2.text(0.02, 2.25, "PPO 达标线 (2cm)", color="#e53935", fontsize=9)
ax2.annotate("v4 真值目标也到不了 2cm\n(最近 4.56cm ≈ 示教终点 4.6cm)\n→ BC 上限 = 示教精度",
             xy=(1, vla_precision), xytext=(0.42, 6.2),
             fontsize=9, color="#37474f",
             arrowprops=dict(arrowstyle="->", color="#607d8b"))
ax2.set_ylabel("最近可达距离 (cm)")
ax2.set_title("2cm 口径下的最近距离")
ax2.set_ylim(0, 9)
fig.tight_layout()
fig.savefig(FIGDIR / "ppo_vs_vla.png", dpi=120)
plt.close(fig)

# --------------------------------------------------------------------------
# 3. VLA 训练曲线 (解析 outputs/train_vla_log.txt)
# --------------------------------------------------------------------------
vla_log = ROOT / "outputs" / "train_vla_log.txt"
vla_epochs, vla_train, vla_val = [], [], []
if vla_log.exists():
    for line in read_any(vla_log).splitlines():
        m = re.match(r"epoch\s+(\d+)/\d+\s+train\s+([\d.]+)\s+val\s+([\d.]+)", line)
        if m:
            vla_epochs.append(int(m.group(1)))
            vla_train.append(float(m.group(2)))
            vla_val.append(float(m.group(3)))
if vla_epochs:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vla_epochs, vla_train, label="训练损失")
    ax.plot(vla_epochs, vla_val, label="验证损失")
    ax.set_xlabel("轮次 (epoch)")
    ax.set_ylabel("MSE 损失")
    ax.set_title("VLA 行为克隆训练曲线 (v3, 运动帧加权)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "vla_training.png", dpi=120)
    plt.close(fig)

# --------------------------------------------------------------------------
# 4. PPO 训练曲线 (解析 outputs/train_1m_log.txt 的 entropy)
# --------------------------------------------------------------------------
ppo_log = ROOT / "outputs" / "train_1m_log.txt"
steps, entropies = [], []
if ppo_log.exists():
    for line in read_any(ppo_log).splitlines():
        m = re.match(r"\|\s+total_timesteps\s+\|\s+(\d+)", line)
        if m:
            steps.append(int(m.group(1)))
        m = re.match(r"\|\s+entropy_loss\s+\|\s+(-?[\d.]+)", line)
        if m:
            entropies.append(float(m.group(1)))
if steps and entropies:
    n = min(len(steps), len(entropies))
    steps, entropies = steps[-n:], entropies[-n:]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, entropies, color="#4285f4")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("熵 (entropy)")
    ax.set_title("PPO 训练: 策略熵下降 = 从乱试到有主见")
    fig.tight_layout()
    fig.savefig(FIGDIR / "ppo_entropy.png", dpi=120)
    plt.close(fig)

# --------------------------------------------------------------------------
# 报告正文
# --------------------------------------------------------------------------
report = f"""# 实验报告: 六轴机械臂 UR5e 控制栈 (MuJoCo)

> 作者: (你的名字)   日期: 2026-09
> 实验环境: MuJoCo 3.9 + Python 3.10, 单机 RTX 3060 Laptop / CPU 训练

## 1. 实验一: 强化学习 (PPO) 训练末端速度控制策略

- 任务: Reach —— 末端到达随机目标点 (2cm 内成功)
- 环境: UR5eReachEnv, 动作 = 6 维末端速度 (OSC 映射), 观测 21 维
- 训练: PPO, 1M 步 × 3 种子, ~30 分钟 CPU / 种子
- **结果: 成功率 150/150 (100.0% ± 0.0%, 2cm 口径, 50 局 × 3 种子), 最终距离 1.86±0.09cm**

![PPO 熵曲线](figures/ppo_entropy.png)

*解读: 熵 (策略随机性) 随训练下降 = 策略从"乱试"收敛到"有主见"。*

## 2. 实验二: VLA 行为克隆 (图像 -> 动作)

- 数据: 键盘遥操作采集 24 条示教 (~3 分钟人工操作, 其中 14 条自动保存、终点 <9cm), 预处理后 4703 帧
- 模型: 轻量 CNN (224x224 图像 -> 6 维末端速度), MSE 行为克隆
- 部署: 模型输出 -> 增益 x1.5 -> OSC -> 仿真闭环

### 版本演进

![VLA 版本演进](figures/vla_versions.png)

| 版本 | 改动 | 成功率 (9cm 口径) |
|---|---|---|
| v1 | 1584 帧, 基础训练 | 1/5 (20%) |
| v2 | 5449 帧 + 数据增强 | 0/10 (停在半路) |
| v2+增益 | 诊断"回归到均值", 输出 x1.5 | 3/8 (37.5%) |
| v3 | 发呆帧裁剪 + 运动帧 3 倍加权 | 49/50 (98%, 复现重训) |
| v4 | 目标条件输入 (图像 + 真值坐标) | 50/50 (100%) |
| **v4.5** | **目标条件 + RGB-D 实时检测 (修复版, 误差 ~5cm)** | **50/50 (100%)** |

> 口径说明: VLA 的"成功"= 距球心 9cm 内停 1 秒 (与遥操作示教的自评标准一致)。
> PPO 的成功口径是 2cm —— 两者阈值不同, 数字不可直接对比 (见第 3 节)。
> v1~v2+增益为历史小样本记录 (5~10 局); v3~v4.5 为 2026-09-06 统一 50 局协议复测
> (v3 权重曾被覆盖, 按原配方复现重训; 原记录 14/20 (70%) 是小样本噪声)。
> v4.5 的检测器在复测中发现两个 bug (min 选取被边缘像素劫持 + 检测失败静默回退真值),
> 已修复为"红像素质心 + 中位深度 + 中性回退", 上表为修复版成绩。

![VLA 训练曲线](figures/vla_training.png)

*关键诊断: 行为克隆的"回归到均值"——数据中发呆帧偏多导致模型输出偏小、
提前停车; 运动帧加权 + 输出增益是有效解药。架构演进: 端到端看图 (v3)
→ 目标条件 (v4) → 感知+策略完整系统 (v4.5)。*

## 3. 两种学习范式对比 (核心结论)

![PPO vs VLA](figures/ppo_vs_vla.png)

| | PPO 强化学习 | VLA 行为克隆 |
|---|---|---|
| 学习方式 | 试错 (奖励驱动) | 模仿 (示范驱动) |
| 训练代价 | 100 万步 × 3 种子 | ~3 分钟人工示教 |
| 成功率 (2cm 口径) | 150/150 (100%, 3 种子) | **0/50 (v4 真值 / v4.5 修复版相同; 同配方重训 5/50)** |
| 成功率 (9cm 口径) | 必然达标 | 50/50 (v4 与 v4.5 修复版相同) |
| 最近可达距离 (2cm 口径) | 1.86±0.09cm | 4.56±1.93cm (v4 真值) |
| 输入 | 21 维状态 | 相机图像 + RGB-D 检测目标 |
| 适用场景 | 奖励好定义的精确控制 | 有示教、需要视觉/泛化的任务 |

**结论: 同一任务可用两种范式解决, 但必须分口径表述。PPO 在 2cm 标准下
150/150 (50 局×3 种子, 最终距离 1.86±0.09cm); VLA 在 9cm 标准下 50/50,
在 2cm 标准下 0/50 —— 且**换真值目标 (v4) 或换修复版检测器 (v4.5) 结果一样**
(两者 50 局的最终距离分布几乎重合)。回放测量显示遥操作示教的终点距离为
4.6±2.4cm, 与 v4 真值目标下的最近距离 4.56±1.93cm 几乎重合: 行为克隆的
精度上限 = 示教精度, 换目标来源救不了它。RGB-D 检测器 (修复版误差 ~5cm,
开局覆盖率 82%) 在 2cm 紧公差下只是第二层噪声。这恰是行为克隆的已知局限:
策略性能受示教质量约束, 而强化学习是模型法 (动力学已知), 可直接优化到
任意阈值。完整系统由 RGB-D 感知模块与目标条件策略网络组成, 具备接入
视觉-语言指令 (VLA 路线) 的扩展能力。**

**附注 (数据清洗对照实验)**: 曾怀疑训练集混入的 10 条未对准示教拖累精度,
做了受控对照 (同配方, 干净 14 条 vs 全部 24 条): 干净模型 @2cm 反而 0/50
(最近 3.92cm), 脏基线重训 5/50 (最近 2.71cm)。结论: 未对准示教的运动片段
仍有价值, "发呆裁剪 + 运动帧加权"已能消化数据不平衡, 无需清洗。
VLA 偶尔进 2cm 的比例 (~10%) 与示教中终点 <2cm 的占比一致 —— 上限由示教分布决定。

## 4. 附: 运动学层实验 (W2)

- 奇异位形脱困: 折叠点是不稳定平衡, 任意扰动最终脱困 (无最小甩力, 只有起飞时间)
- DLS 阻尼扫掠: λ 越大解越软越慢, 甜蜜点 0.05 (0.01 打滑 / 0.5 爬行)
- 详见 docs/IK推导笔记.md
"""

REPORT.write_text(report, encoding="utf-8")

# --------------------------------------------------------------------------
# 自包含 HTML 版 (手工排版, 图片 base64 内嵌)
# --------------------------------------------------------------------------
def b64(fname: str) -> str:
    data = (FIGDIR / fname).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def figure(fname: str, caption: str) -> str:
    return (
        f'<figure><img src="{b64(fname)}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>实验报告: 六轴机械臂 UR5e 控制栈</title>
<style>
  body {{
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    max-width: 880px; margin: 0 auto; padding: 32px 24px 64px;
    color: #263238; line-height: 1.75; background: #fafbfc;
  }}
  header {{ text-align: center; margin-bottom: 36px; }}
  h1 {{ font-size: 28px; margin-bottom: 6px; color: #1a237e; }}
  .meta {{ color: #78909c; font-size: 14px; }}
  h2 {{
    font-size: 20px; color: #1a237e; border-bottom: 2px solid #3f51b5;
    padding-bottom: 6px; margin-top: 40px;
  }}
  section {{ background: #fff; border-radius: 10px; padding: 20px 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 24px; }}
  figure {{ text-align: center; margin: 20px 0; }}
  figure img {{ max-width: 100%; border-radius: 8px; border: 1px solid #e0e0e0; }}
  figcaption {{ color: #607d8b; font-size: 13px; margin-top: 8px; }}
  table {{ border-collapse: collapse; margin: 16px auto; width: 100%; }}
  th {{ background: #3f51b5; color: #fff; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eceff1; }}
  tr:nth-child(even) td {{ background: #f5f7ff; }}
  .highlight {{
    background: #e8f5e9; border-left: 4px solid #43a047;
    padding: 12px 16px; border-radius: 6px; margin: 16px 0;
  }}
  .badge {{ display:inline-block; background:#43a047; color:#fff;
    border-radius: 4px; padding: 1px 8px; font-size: 13px; margin-left: 6px; }}
  ul {{ padding-left: 22px; }}
</style>
</head>
<body>
<header>
  <h1>实验报告: 六轴机械臂 UR5e 控制栈</h1>
  <div class="meta">作者: (你的名字) &nbsp;|&nbsp; 2026-09 &nbsp;|&nbsp;
    MuJoCo 3.9 + Python 3.10 &nbsp;|&nbsp; 单机 RTX 3060 Laptop (CPU 训练)</div>
</header>

<section>
<h2>1. 强化学习 (PPO): 末端速度控制策略</h2>
<ul>
  <li><b>任务</b>: Reach —— 控制 UR5e 末端到达随机目标点 (2cm 内视为成功)</li>
  <li><b>环境</b>: UR5eReachEnv; 动作 = 6 维末端速度 (经 OSC 映射为关节命令); 观测 = 21 维状态</li>
  <li><b>训练</b>: PPO (Stable-Baselines3), 100 万步 × 3 种子, 约 30 分钟 CPU / 种子</li>
  <li><b>结果</b>: 成功率 <b>150/150 (100.0% ± 0.0%)</b> (2cm 口径, 50 局 × 3 种子),
      最终距离 1.86±0.09cm <span class="badge">达标</span></li>
</ul>
{figure("ppo_entropy.png", "图 1: PPO 训练中策略熵随步数下降 —— 从随机探索收敛到稳定策略")}
</section>

<section>
<h2>2. VLA 行为克隆: 图像到动作的模仿学习</h2>
<ul>
  <li><b>数据</b>: 键盘遥操作采集 24 条示教 (约 3 分钟人工操作, 其中 14 条自动保存、终点 &lt;9cm), 预处理后 4703 帧 (图像 + 末端速度动作对)</li>
  <li><b>模型</b>: 轻量 CNN (224×224 图像 → 6 维末端速度), MSE 行为克隆, 数据增强</li>
  <li><b>部署</b>: 模型输出 → 增益 ×1.5 → OSC 接口 → MuJoCo 仿真闭环</li>
</ul>
<h3 style="color:#37474f">版本演进</h3>
{figure("vla_versions.png", "图 2: VLA 各版本闭环成功率 (口径: 距球心 9cm 内停 1 秒)")}
<table>
  <tr><th>版本</th><th>改动</th><th>成功率 (9cm 口径)</th></tr>
  <tr><td>v1</td><td>1584 帧, 基础训练</td><td>1/5 (20%)</td></tr>
  <tr><td>v2</td><td>5449 帧 + 数据增强</td><td>0/10 (停在半路)</td></tr>
  <tr><td>v2+增益</td><td>诊断"回归到均值", 输出 ×1.5</td><td>3/8 (37.5%)</td></tr>
  <tr><td>v3</td><td>发呆帧裁剪 + 运动帧 3 倍加权</td><td>49/50 (98%, 复现重训)</td></tr>
  <tr><td>v4</td><td>目标条件输入 (图像 + 真值坐标)</td><td>50/50 (100%)</td></tr>
  <tr><td><b>v4.5</b></td><td><b>目标条件 + RGB-D 实时检测 (修复版, 误差 ~5cm)</b></td><td><b>50/50 (100%)</b></td></tr>
</table>
<p style="color:#546e7a;font-size:13px">口径说明: VLA 的"成功" = 距球心 <b>9cm</b> 内停 1 秒
(与遥操作示教自评标准一致); PPO 的成功口径是 <b>2cm</b>。阈值不同,
两列数字不可直接对比 (见第 3 节)。v1~v2+增益为历史小样本记录 (5~10 局);
v3~v4.5 为 2026-09-06 统一 50 局协议复测 (v3 权重曾被覆盖, 按原配方复现重训;
原记录 14/20 (70%) 是小样本噪声)。v4.5 的检测器在复测中发现两个 bug
(min 选取被边缘像素劫持 + 检测失败静默回退真值), 已修复为
"红像素质心 + 中位深度 + 中性回退", 上表为修复版成绩。</p>
{figure("vla_training.png", "图 3: VLA 行为克隆训练曲线 (25 轮, 运动帧加权)")}
<div class="highlight"><b>关键诊断</b>: 行为克隆的"回归到均值"——数据中发呆帧偏多,
模型输出偏小、提前停车。解药: 发呆帧裁剪 + 运动帧损失加权 (治本) + 输出增益 (治标)。
<b>架构演进</b>: 端到端看图 (v3) → 目标条件 (v4) → 感知+策略完整系统 (v4.5):
RGB-D 检测器实时估计目标 (彩色找红球 + 中位深度 + 反投影, 修复版误差 ~5cm,
开局覆盖率 82%, 失败时中性回退), 9cm 口径闭环成功率 50/50
(2cm 口径下 v4 真值目标同样 0/50, 差距分析见第 3 节)。</div>
</section>

<section>
<h2>3. 两种学习范式对比 (核心结论)</h2>
{figure("ppo_vs_vla.png", "图 4: 同一 Reach 任务的两种学习范式 —— 双口径成功率 + 典型停靠精度")}
<table>
  <tr><th></th><th>PPO 强化学习</th><th>VLA 行为克隆</th></tr>
  <tr><td>学习方式</td><td>试错 (奖励驱动)</td><td>模仿 (示范驱动)</td></tr>
  <tr><td>训练代价</td><td>100 万步 × 3 种子</td><td>约 3 分钟人工示教</td></tr>
  <tr><td>成功率 (2cm 口径)</td><td>150/150 (100%, 3 种子)</td><td><b>0/50 (v4 真值 / v4.5 修复版相同; 同配方重训 5/50)</b></td></tr>
  <tr><td>成功率 (9cm 口径)</td><td>必然达标</td><td>50/50 (v4 与 v4.5 修复版相同)</td></tr>
  <tr><td>最近可达距离 (2cm 口径)</td><td>1.86±0.09cm</td><td>4.56±1.93cm (v4 真值)</td></tr>
  <tr><td>输入</td><td>21 维状态</td><td>相机图像 + RGB-D 检测目标</td></tr>
  <tr><td>适用场景</td><td>奖励好定义的精确控制</td><td>有示教、需要视觉与语言泛化的任务</td></tr>
</table>
<div class="highlight"><b>结论</b>: 同一任务可用两种范式解决, 但必须分口径表述。
PPO 在 2cm 标准下 150/150 (50 局×3 种子, 最终距离 1.86±0.09cm);
VLA 在 9cm 标准下 50/50、在 2cm 标准下 0/50 —— 且<b>换真值目标 (v4) 或
换修复版检测器 (v4.5) 结果一样</b> (两者 50 局的最终距离分布几乎重合)。
回放测量显示遥操作示教的终点距离为 4.6±2.4cm, 与 v4 真值目标下的最近距离
4.56±1.93cm 几乎重合: 行为克隆的精度上限 = 示教精度, 换目标来源救不了它。
RGB-D 检测器 (修复版误差 ~5cm, 开局覆盖率 82%) 在 2cm 紧公差下只是第二层噪声。
这恰是行为克隆的已知局限: 策略性能受示教质量约束, 而强化学习是模型法
(动力学已知), 可直接优化到任意阈值。完整系统由 RGB-D 感知模块与目标条件
策略网络组成, 具备接入视觉-语言指令 (VLA 路线) 的扩展能力 ——
这是本项目"具身智能"方向的核心证据。</div>
<div class="highlight"><b>附注 (数据清洗对照实验)</b>: 曾怀疑训练集混入的 10 条未对准示教
拖累精度, 做了受控对照 (同配方, 干净 14 条 vs 全部 24 条): 干净模型 @2cm 反而
0/50 (最近 3.92cm), 脏基线重训 5/50 (最近 2.71cm)。结论: 未对准示教的运动片段
仍有价值, "发呆裁剪 + 运动帧加权"已能消化数据不平衡, 无需清洗。
VLA 偶尔进 2cm 的比例 (~10%) 与示教中终点 &lt;2cm 的占比一致 —— 上限由示教分布决定。</div>
</section>

<section>
<h2>4. 附: 运动学层实验 (W2)</h2>
<ul>
  <li><b>奇异位形脱困</b>: 对折点是<b>不稳定平衡</b>——任意大于零的扰动最终脱困,
      不存在"最小甩力", 只有"甩力越大、起飞越早"</li>
  <li><b>DLS 阻尼扫掠</b>: λ 越大解越软越慢; 0.01 打滑 / 0.05 甜蜜点 / 0.5 爬行</li>
  <li>推导细节见 <code>docs/IK推导笔记.md</code></li>
</ul>
</section>

</body>
</html>
"""

HTML_PATH = ROOT / "docs" / "实验报告.html"
HTML_PATH.write_text(html, encoding="utf-8")
print("报告已生成:", REPORT)
print("HTML 版:", HTML_PATH)
print("图表:", [f.name for f in sorted(FIGDIR.glob("*.png"))])
