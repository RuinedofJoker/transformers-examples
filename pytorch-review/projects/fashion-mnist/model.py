import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1 , 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(
        root="./data",  # 数据存储路径
        train=True,  # 训练集
        download=True,  # 如果不存在则下载
        transform=transform  # 转换为张量
    )

    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    images, _ = next(iter(dataloader))

    model = FashionMNISTNet()
    print(model(images))