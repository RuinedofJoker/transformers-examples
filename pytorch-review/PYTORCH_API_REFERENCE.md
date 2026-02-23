# PyTorch API 速查手册

> 前7章核心知识点精简版

---

## 第1章：张量（Tensors）

### 创建张量

```python
import torch

# 从数据创建
torch.tensor([1, 2, 3])
torch.tensor([[1, 2], [3, 4]])

# 特殊张量
torch.zeros(2, 3)           # 全0
torch.ones(2, 3)            # 全1
torch.full((2, 3), 7)       # 全部填充为7

# 随机张量
torch.rand(2, 3)            # 均匀分布 [0, 1)
torch.randn(2, 3)           # 标准正态分布（均值0，标准差1）
torch.randint(0, 10, (2, 3))    # 整数随机 [0, 10)
torch.randint(low=5, high=15, size=(2, 3))  # 整数随机 [5, 15)

# 指定范围的随机数
torch.rand(2, 3) * 10       # [0, 10)
torch.rand(2, 3) * 5 + 2    # [2, 7)
torch.randn(2, 3) * 2 + 5   # 正态分布（均值5，标准差2）

# 序列张量
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1]

# 从其他张量创建
x.clone()                   # 复制
torch.zeros_like(x)         # 相同形状的全0
torch.ones_like(x)          # 相同形状的全1
```

### 张量属性

```python
x.shape         # 形状
x.dtype         # 数据类型
x.device        # 设备（cpu/cuda）
x.ndim          # 维度数
x.numel()       # 元素总数
```

### 索引和切片

```python
x[0]            # 第一个元素
x[0, 1]         # 第0行第1列
x[:, 0]         # 第0列
x[0, :]         # 第0行
x[1:3]          # 切片
```

### 形状操作

```python
x.view(2, 3)            # 改变形状（共享内存）
x.reshape(2, 3)         # 改变形状（可能复制）
x.squeeze()             # 移除所有大小为1的维度
x.squeeze(dim=0)        # 移除指定维度（如果大小为1）
x.unsqueeze(dim=0)      # 在指定位置插入大小为1的维度
x.transpose(0, 1)       # 转置两个维度
x.permute(2, 0, 1)      # 重排所有维度
x.flatten()             # 展平为1D
```

### 张量连接

```python
torch.cat([x, y], dim=0)    # 在指定维度拼接
torch.stack([x, y], dim=0)  # 创建新维度并堆叠
torch.split(x, 2, dim=0)    # 分割张量
```

### 算术运算

```python
x + y           # 加法
x - y           # 减法
x * y           # 逐元素乘法
x / y           # 除法
x @ y           # 矩阵乘法
x.matmul(y)     # 矩阵乘法
x ** 2          # 幂运算
```

### 聚合操作

```python
x.sum()                 # 求和
x.sum(dim=0)            # 按维度求和
x.sum(dim=0, keepdim=True)  # 保持维度
x.mean()                # 平均值
x.max()                 # 最大值
x.min()                 # 最小值
x.argmax()              # 最大值索引
x.argmin()              # 最小值索引
```

### 与 NumPy 互操作

```python
# Tensor → NumPy
x.numpy()

# NumPy → Tensor
torch.from_numpy(arr)
```

---

## 第2章：数据集与数据加载器

### Dataset

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### 预加载数据集

```python
from torchvision import datasets

# FashionMNIST
train_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# CIFAR10
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

### DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

# 迭代
for images, labels in train_loader:
    # 训练代码
    pass
```

---

## 第3章：变换（Transforms）

```python
from torchvision import transforms

# 单个变换
transform = transforms.ToTensor()

# 组合变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 常用变换
transforms.ToTensor()                   # PIL/NumPy → Tensor
transforms.Normalize(mean, std)         # 标准化
transforms.Resize((224, 224))           # 调整大小
transforms.RandomCrop(32)               # 随机裁剪
transforms.RandomHorizontalFlip()       # 随机水平翻转
transforms.Lambda(lambda x: x * 2)      # 自定义变换
```

---

## 第4章：构建神经网络

### 定义模型

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
```

### 常用层

```python
# 全连接层
nn.Linear(in_features, out_features)

# 激活函数
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
torch.relu(x)

# 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size)

# 池化层
nn.MaxPool2d(kernel_size)
nn.AvgPool2d(kernel_size)

# Dropout
nn.Dropout(p=0.5)

# BatchNorm
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
```

### Sequential 容器

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### 模型操作

```python
model.parameters()      # 获取所有参数
model.train()           # 训练模式
model.eval()            # 评估模式
model.to(device)        # 移动到设备
```

---

## 第5章：自动微分（Autograd）

### 梯度计算

```python
# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 前向传播
y = x ** 2

# 反向传播
y.backward()

# 查看梯度
print(x.grad)  # dy/dx

# 清零梯度
x.grad.zero_()
```

### 梯度控制

```python
# 禁用梯度计算
with torch.no_grad():
    y = model(x)

# 分离计算图
y = x.detach()

# 设置是否需要梯度
x.requires_grad = True
x.requires_grad_(True)
```

---

## 第6章：优化模型参数

### 损失函数

```python
import torch.nn as nn

# 回归任务
criterion = nn.MSELoss()            # 均方误差
criterion = nn.L1Loss()             # 平均绝对误差

# 分类任务
criterion = nn.CrossEntropyLoss()   # 交叉熵（包含Softmax）
criterion = nn.BCELoss()            # 二分类交叉熵

# 使用
loss = criterion(outputs, targets)
```

### 优化器

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam（推荐）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

### 训练循环

```python
# 训练
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

# 评估
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # 计算指标
```

---

## 第7章：保存和加载模型

### 保存和加载参数（推荐）

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 保存和加载检查点

```python
# 保存检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 保存完整模型（不推荐）

```python
# 保存
torch.save(model, 'model.pth')

# 加载（PyTorch 2.6+）
model = torch.load('model.pth', weights_only=False)
```

---

## 常用工具函数

### 设备管理

```python
# 检查CUDA是否可用
torch.cuda.is_available()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移动到设备
model.to(device)
data.to(device)
```

### 随机种子

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 调试技巧

```python
# 查看形状
print(x.shape)

# 查看数据类型
print(x.dtype)

# 查看设备
print(x.device)

# 查看是否需要梯度
print(x.requires_grad)

# 查看模型结构
print(model)

# 查看参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')

# 查看某层参数
for name, param in model.named_parameters():
    print(f'{name}: {param.shape}')
```

---

## 完整训练模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. 准备数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义模型
model = MyModel().to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# 6. 保存模型
torch.save(model.state_dict(), 'model.pth')
```

---

**提示**：这份速查手册涵盖了前7章的核心 API，建议收藏备用！
