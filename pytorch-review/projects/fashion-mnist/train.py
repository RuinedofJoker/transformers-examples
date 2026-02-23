import torch
import torch.nn as nn
import torch.optim as optim
from model import FashionMNISTNet
from data_loader import train_dataloader, test_dataloader, test_data

learning_rate = 0.001
num_epochs = 3
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = FashionMNISTNet().to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def test_model(data):
    class_names = ['T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    with torch.no_grad():
        image, label = data
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f"预测类别: {class_names[predicted.item()]}")
        print(f"真实类别: {class_names[label]}")

for i in range(5):
    test_model(test_data[i])

for epoch in range(num_epochs):
    model.train()
    print(f"epoch {epoch} start")
    print(f"total {len(train_dataloader)} batch")
    i = 1
    for images, labels in train_dataloader:
        print(f"batch {i} start")
        images = images.to(device)
        labels = labels.to(device)
        model_output = model(images)
        model_loss = loss(model_output, labels)
        model_loss.backward()
        print(f"loss {model_loss.item()}")
        optimizer.step()
        optimizer.zero_grad()
        print(f"batch {i} end")
        i += 1
    print(f"epoch {epoch} end")

for i in range(5):
    test_model(test_data[i])

model.eval()
correct = 0
total = 0
with torch.no_grad():
    print(f"eval total {len(test_dataloader)} batch")
    i = 1
    for images, labels in test_dataloader:
        print(f"eval batch {i} start")
        images = images.to(device)
        labels = labels.to(device)
        model_output = model(images)
        model_loss = loss(model_output, labels)
        print(f"loss {model_loss.item()}")
        _, predicted = torch.max(model_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"eval batch {i} end")
        i += 1

print(f"correct {correct} / {total}")

torch.save(model.state_dict(), './checkpoints/model.pth')