# 诊断脚本说明 (scripts/_debug/)

这些脚本是项目**数据可信度核查**过程中写的分析工具，保留在仓库里是为了让
实验报告里的每个结论**都能被复现**。它们不是运行主流程所需的脚本。

| 脚本 | 用途 | 对应的报告结论 |
|---|---|---|
| `sweep_detector.py` | 扫描 50 个球位，统计 RGB-D 检测器误差 | 旧版检测器平均误差 **20.5 ± 11.1 cm**（min-深度被边缘像素劫持） |
| `calibrate_depth.py` | 深度标定：比较 mean / median / min 三种取法 | 修复版（质心 + 中位深度）误差降到 **~5 cm** |
| `inspect_detector.py` | 单帧检测结果可视化与数值转储 | 定位异常像素（深度 2.84 m vs 球面 3.12 m） |
| `diff_v4_v45.py` | 逐局对比"真值目标"与"检测目标"两套评估结果 | 发现 **50 局中 43 局完全相同** → 检测器未起作用的证据 |
| `replay_demos.py` | 把遥操作示教**通过完整物理链路重放**，测量真实终止距离 | 示教终点 **4.6 ± 2.4 cm**（n=14 自动保存） |
| `trace_divergence.py` | 对比"朴素速度积分"与"物理回放"的轨迹发散 | 证明朴素积分失真（终点 4.4 cm vs 真实 30.1 cm） |
| `compare_replay.py` | 对比物理手位置与 FK(q_des) 位置 | 两者相差约 1 cm → 伺服链忠实，回放可信 |
| `summarize_evals.py` | 汇总 `outputs/eval_*.txt` 的逐局距离 | 报告里的成功率/距离统计 |
| `analyze_eval.py` | 单次评估的分布分析（均值/标准差/最好成绩） | 双口径数据表 |
| `flick_sweep.py` | 奇异位形脱困实验：扫描扰动强度 | 折叠点是**不稳定平衡**（无最小甩力，只有起飞时间） |
| `stiffness_scan.py` | 位置伺服 kv/kp 比值的扫掠 | 伺服带宽 **0.8 Hz → ~20 Hz**（kv/kp 0.2 → 0.05） |
| `inspect_vla_log.py` | 解析 VLA 训练日志 | 六版本演进与验证损失曲线 |

## 复现方式

```powershell
conda activate mujoco_project
cd "D:\deepseek test\1"

# 例: 复现检测器误差扫描 (需要数据集, 见 data/README.md)
python scripts/_debug/sweep_detector.py

# 例: 复现示教终点测量 (需要 datasets/teleop/*.npz)
python scripts/_debug/replay_demos.py
```

> 注意：这些脚本需要**数据集**（约 1 GB，未随仓库分发）。
> 数据可用 `scripts/teleop_collect.py` + `scripts/preprocess_vla.py` 重新采集与生成。
