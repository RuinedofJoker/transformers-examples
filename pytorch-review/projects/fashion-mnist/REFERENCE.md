# FashionMNIST 项目快速参考

## 类别映射

```python
class_names = [
    'T恤',      # 0: T-shirt/top
    '裤子',     # 1: Trouser
    '套衫',     # 2: Pullover
    '连衣裙',   # 3: Dress
    '外套',     # 4: Coat
    '凉鞋',     # 5: Sandal
    '衬衫',     # 6: Shirt
    '运动鞋',   # 7: Sneaker
    '包',       # 8: Bag
    '短靴'      # 9: Ankle boot
]
```

## 常用代码片段

### 数据加载
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

### 模型定义
```python
import torch.nn as nn

class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 训练循环
```python
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### 评估
```python
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
```

### 保存和加载
```python
# 保存
torch.save(model.state_dict(), 'checkpoints/model.pth')

# 加载
model = FashionMNISTNet()
model.load_state_dict(torch.load('checkpoints/model.pth'))
model.eval()
```

## 调试技巧

### 检查数据形状
```python
images, labels = next(iter(train_loader))
print(f"Images shape: {images.shape}")  # [64, 1, 28, 28]
print(f"Labels shape: {labels.shape}")  # [64]
```

### 检查模型输出
```python
outputs = model(images)
print(f"Outputs shape: {outputs.shape}")  # [64, 10]
```

### 查看参数数量
```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

## 常见问题

### Q: 训练很慢怎么办？
A: 减小 batch_size 或减少 epoch 数量

### Q: 准确率太低怎么办？
A:
- 增加训练轮数
- 调整学习率
- 增加网络层数或神经元数量

### Q: 出现 CUDA out of memory？
A:
- 减小 batch_size
- 使用 CPU 训练（去掉 .to(device)）

### Q: 损失不下降？
A:
- 检查学习率是否太小
- 检查数据是否正确加载
- 检查模型前向传播是否正确
