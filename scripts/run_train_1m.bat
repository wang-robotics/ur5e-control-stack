@echo off
rem 1M 步 RL 训练 (由 Windows 计划任务启动, 关闭网页/终端也不受影响)
cd /d "D:\deepseek test\1"
"C:\Users\wzw\miniconda3\envs\mujoco_project\python.exe" scripts\train_sb3.py --timesteps 1000000 > outputs\train_1m_log.txt 2>&1
