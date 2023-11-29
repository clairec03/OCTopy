import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from vit_pytorch import ViT
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_dataset_root = "/mnt/c/Users/Sean/Downloads/OCT_DATASET_DO_NOT_SHARE_WITH_ANYONE_split"
im_size = 256
crop_size = 5

transform_train = transforms.Compose(
    [
        # transforms.RandomCrop(im_size, padding=4),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((im_size + 2 * crop_size, im_size + 2 * crop_size)),
        transforms.Lambda(
            lambda img: img.crop((crop_size, crop_size, img.width - crop_size, img.height - crop_size))
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    # transforms.Normalize(mean=[0.11,0.11,0.11], std=[0.18,0.18,0.18])
        # transforms.Normalize(
        # convert to [0,1]
        transforms.Lambda(lambda img: img / np.max(img.numpy())),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # convert to [0,1]
        transforms.Lambda(lambda img: img / np.max(img.numpy())),
    ]
)
train_dataset = ImageFolder(root=my_dataset_root + '/train', transform=transform_train)
val_dataset = ImageFolder(root=my_dataset_root + '/val', transform=transform_test)
test_dataset = ImageFolder(root=my_dataset_root + '/test', transform=transform_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 4,
    dim = 1024,
    depth = 8,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def test(data_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in tqdm(data_loader, leave=False, total=len(data_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Training loop
num_epochs = 200
train_acc, val_acc, test_acc, train_loss = [], [], [], []
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # track accuracy
    train_loss.append(loss.item())
    train_acc.append(test(train_loader))
    val_acc.append(test(val_loader))
    test_acc.append(test(test_loader))

    # save as one csv
    with open('train_acc.csv', 'w') as f:
        f.write('train_loss,train_acc,val_acc,test_acc\n')
        for train_loss_item, train_acc_item, val_acc_item, test_acc_item in zip(train_loss, train_acc, val_acc, test_acc):
            f.write(f'{train_loss_item},{train_acc_item},{val_acc_item},{test_acc_item}\n')

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}')


