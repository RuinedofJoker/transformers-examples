# CIFAR-10 项目学习指南

## 📋 项目概览

这是一个更具挑战性的图像分类项目。你将学习使用**卷积神经网络（CNN）**来识别彩色图像。

---

## 🎯 第1步：数据加载模块（data_loader.py）

### 任务目标
加载 CIFAR-10 数据集并进行预处理。

### 需要实现的功能

#### 1.1 导入必要的库
```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

#### 1.2 定义数据变换

**训练集变换**（包含数据增强）：
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3通道标准化
])
```

**测试集变换**（不需要数据增强）：
```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

#### 1.3 加载数据集
```python
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)
```

#### 1.4 创建 DataLoader
- 批量大小：128
- 训练集打乱：True
- 测试集打乱：False

### 关键区别

| 特性 | FashionMNIST | CIFAR-10 |
|------|-------------|----------|
| 图像形状 | `(1, 28, 28)` | `(3, 32, 32)` |
| 标准化参数 | `(0.5,), (0.5,)` | `(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)` |
| 数据增强 | 不需要 | 需要（提升性能） |

### 验收标准
- ✅ 能成功加载 CIFAR-10 数据集
- ✅ 训练集和测试集使用不同的变换
- ✅ 打印数据集大小和图像形状

---

## 🎯 第2步：CNN 模型定义（model.py）

### 任务目标
设计一个卷积神经网络来处理彩色图像。

### CNN 架构设计

推荐的简单 CNN 架构：

```
输入：(3, 32, 32)
    ↓
卷积层1：Conv2d(3, 32, 3) + ReLU
    ↓
卷积层2：Conv2d(32, 64, 3) + ReLU
    ↓
最大池化：MaxPool2d(2, 2)
    ↓
卷积层3：Conv2d(64, 128, 3) + ReLU
    ↓
最大池化：MaxPool2d(2, 2)
    ↓
展平：Flatten
    ↓
全连接层1：Linear(?, 256) + ReLU
    ↓
全连接层2：Linear(256, 10)
```

### 需要实现的功能

#### 2.1 定义模型类
```python
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 定义全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
```

#### 2.2 实现前向传播
```python
def forward(self, x):
    # 卷积 + 激活
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = self.pool(x)  # 池化

    x = torch.relu(self.conv3(x))
    x = self.pool(x)  # 池化

    # 展平
    x = x.view(-1, 128 * 8 * 8)

    # 全连接
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
```

### 新概念说明

#### Conv2d（卷积层）
```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding)
```
- `in_channels`: 输入通道数（RGB=3）
- `out_channels`: 输出通道数（特征图数量）
- `kernel_size`: 卷积核大小（通常3或5）
- `padding`: 填充（保持尺寸不变）

#### MaxPool2d（最大池化）
```python
nn.MaxPool2d(kernel_size, stride)
```
- 作用：降低特征图尺寸，减少参数
- `(32, 32)` → `(16, 16)` → `(8, 8)`

### 验收标准
- ✅ 模型可以成功实例化
- ✅ 能够处理 `(batch, 3, 32, 32)` 的输入
- ✅ 输出形状正确：`(batch, 10)`
- ✅ 打印模型结构和参数数量

---

## 🎯 第3步：训练脚本（train.py）

### 任务目标
训练 CNN 模型并保存检查点。

### 需要实现的功能

#### 3.1 设置超参数
```python
learning_rate = 0.001
num_epochs = 20  # CNN 需要更多轮次
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### 3.2 初始化组件
```python
model = CIFAR10Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

#### 3.3 训练循环
与 FashionMNIST 类似，但注意：
- 训练轮数更多（20轮）
- 可以添加学习率调度器

#### 3.4 保存模型
```python
torch.save(model.state_dict(), 'checkpoints/cifar10_model.pth')
```

### 提示
- CNN 训练比全连接网络慢
- 建议每个 epoch 打印一次平均损失
- 可以保存最佳模型（准确率最高的）

### 验收标准
- ✅ 训练循环正常运行
- ✅ 损失逐渐下降
- ✅ 模型成功保存
- ✅ 测试准确率 > 70%

---

## 📊 预期训练过程

```
Epoch 1: loss=1.8, accuracy=35%
Epoch 5: loss=1.2, accuracy=55%
Epoch 10: loss=0.8, accuracy=65%
Epoch 20: loss=0.5, accuracy=72%
```

---

## 💡 学习建议

1. **先完成基础版本**：
   - 使用简单的 CNN 架构
   - 不添加复杂功能
   - 确保能跑通

2. **逐步优化**：
   - 增加卷积层
   - 添加 Dropout
   - 使用学习率调度

3. **对比 FashionMNIST**：
   - 注意输入形状的变化
   - 理解 CNN 与全连接的区别
   - 体会数据增强的作用

---

## 🚀 开始编码

现在你可以开始了！从第1步开始，创建 `data_loader.py` 文件。

准备好了吗？
