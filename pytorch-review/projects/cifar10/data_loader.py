import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 训练集变换（包含数据增强）
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪（先填充4像素再裁剪回32×32）
    transforms.RandomHorizontalFlip(),         # 随机水平翻转（50%概率）
    transforms.ToTensor(),                     # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3通道标准化
])

# 测试集变换（不需要数据增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练集
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

# 加载测试集
test_data = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# 创建 DataLoader
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

if __name__ == '__main__':
    # 打印数据集信息
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"训练批次数: {len(train_dataloader)}")
    print(f"测试批次数: {len(test_dataloader)}")

    # 查看单张图像的形状
    image, label = train_data[0]
    print(f"\n单张图像形状: {image.shape}")  # 应该是 (3, 32, 32)
    print(f"图像标签: {label}")

    # 查看一个批次的形状
    images, labels = next(iter(train_dataloader))
    print(f"\n批次图像形状: {images.shape}")  # 应该是 (128, 3, 32, 32)
    print(f"批次标签形状: {labels.shape}")    # 应该是 (128,)

    # 可视化部分图像
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # CIFAR-10 类别名称
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

    # 显示前16张图像
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # 反标准化以便显示（从[-1,1]恢复到[0,1]）
            img = images[i] / 2 + 0.5
            # 转换通道顺序：(C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f'{class_names[labels[i]]}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
