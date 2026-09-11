"""VLA 行为克隆训练: 图像 -> 6维末端速度 (轻量 CNN, CPU 可训)。

数据: datasets/vla_dataset.npz (预处理后的遥操作数据)
输出: outputs/vla_bc.pt

用法:
    python scripts/train_vla.py                # 默认 30 轮
    python scripts/train_vla.py --epochs 60
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def augment(x: torch.Tensor) -> torch.Tensor:
    """训练数据增强: 随机亮度/对比度抖动 + 整批随机小位移 (±4px)。

    x: (B,3,H,W) 已归一化张量。全部向量化 (无逐样本循环), CPU 上快。
    """
    b = x.shape[0]
    bright = 1.0 + 0.15 * (2 * torch.rand(b, 1, 1, 1) - 1)
    contrast = 1.0 + 0.15 * (2 * torch.rand(b, 1, 1, 1) - 1)
    x = x * bright * contrast
    sx = int(torch.randint(-4, 5, (1,)).item())
    sy = int(torch.randint(-4, 5, (1,)).item())
    x = torch.roll(x, shifts=(sx, sy), dims=(2, 3))
    return x


class SmallVLA(nn.Module):
    """目标条件视觉动作模型: (图像, 目标坐标) -> 6 维末端速度。

    图像走 4 层卷积 (224 -> 14x14), 目标坐标 (3维) 走小 MLP,
    特征拼接后经头部输出动作。目标条件 = 网络不需要只靠像素找目标,
    只需学会"朝目标走" (视觉定位模块给坐标 + 策略网络执行)。

    goal_cond=False 时是 v3 的纯图像版 (无目标输入, 仅用于历史版本复现)。
    """

    def __init__(self, goal_cond: bool = True) -> None:
        super().__init__()
        self.goal_cond = goal_cond
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.goal_mlp = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        in_dim = 128 * 14 * 14 + (32 if goal_cond else 0)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

    def forward(self, x: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        f = self.features(x)
        if self.goal_cond:
            g = self.goal_mlp(goal)
            f = f.flatten(1)
            return self.head(torch.cat([f, g], dim=1))
        return self.head(f.flatten(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/vla_dataset.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--move-weight", type=float, default=3.0,
                        help="运动帧的损失权重 (发呆帧为1): 抵消发呆帧偏多导致的回归到均值")
    parser.add_argument("--out", default="outputs/vla_bc.pt")
    parser.add_argument("--no-goal", action="store_true",
                        help="纯图像版 (v3 复现): 不给目标坐标, 仅供历史版本对比")
    args = parser.parse_args()

    d = np.load(args.dataset)
    imgs, acts, goals = d["images"], d["actions"], d["goals"]
    X = ((imgs.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    X = torch.from_numpy(X).permute(0, 3, 1, 2)  # (N,H,W,3) -> (N,3,H,W)
    Y = torch.from_numpy(acts)
    G = torch.from_numpy(goals)

    n = len(X)
    split = int(n * 0.9)
    train_dl = DataLoader(
        TensorDataset(X[:split], G[:split], Y[:split]),
        batch_size=args.batch,
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(X[split:], G[split:], Y[split:]), batch_size=args.batch
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"数据集: {n} 帧 (训练 {split}, 验证 {n - split}), 设备: {device}")

    model = SmallVLA(goal_cond=not args.no_goal).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def forward_model(xb: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
        return model(xb, gb) if not args.no_goal else model(xb)

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, cnt = 0.0, 0
        for xb, gb, yb in train_dl:
            xb, gb, yb = xb.to(device), gb.to(device), yb.to(device)
            xb = augment(xb)  # 数据增强
            # 运动帧加权: 有动作的样本损失 x move_weight (抵消发呆帧偏置)
            w = torch.where(
                (yb.abs() > 1e-6).any(dim=1),
                torch.full_like(yb[:, 0], args.move_weight),
                torch.ones_like(yb[:, 0]),
            ).unsqueeze(1)
            opt.zero_grad()
            loss = (w * (forward_model(xb, gb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        model.eval()
        vt, vc = 0.0, 0
        with torch.no_grad():
            for xb, gb, yb in val_dl:
                xb, gb, yb = xb.to(device), gb.to(device), yb.to(device)
                vt += loss_fn(forward_model(xb, gb), yb).item() * len(xb)
                vc += len(xb)
        val = vt / vc
        tag = ""
        if val < best_val:
            best_val = val
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
            tag = " <- 保存最优"
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  "
            f"train {tot / cnt:.5f}  val {val:.5f}{tag}"
        )

    print(f"训练完成, 用时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"最优验证损失 {best_val:.5f}, 模型已保存")


if __name__ == "__main__":
    main()
