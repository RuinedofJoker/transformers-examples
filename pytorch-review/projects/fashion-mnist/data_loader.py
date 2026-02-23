import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root="./data",  # 数据存储路径
    train=True,  # 训练集
    download=True,  # 如果不存在则下载
    transform=transform  # 转换为张量
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.FashionMNIST(
    root="./data",  # 数据存储路径
    train=False,  # 训练集
    download=True,  # 如果不存在则下载
    transform=transform  # 转换为张量
)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

if __name__ == '__main__':
    # 检查训练集和测试集是否重复
    train_image, _ = train_data[0]
    test_image, _ = test_data[0]

    print(f"训练集第一张图片和测试集第一张图片相同吗？")
    print(torch.equal(train_image, test_image))  # 应该输出 False

    dataloader_iter = iter(train_dataloader)
    images, labels = next(dataloader_iter)
    print(images)
    print(labels)

    print("\n")
    print("=" * 50)
    print("\n")

    image, label = train_data[0]
    print(image)
    print(label)

    print("\n")
    print("=" * 50)
    print("\n")

    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # plt.imshow(image.squeeze(), cmap='gray')
    # plt.title(f'Label: {label}')
    # plt.axis('off')
    # plt.show()

    class_names = ['T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f'{class_names[labels[i]]}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()