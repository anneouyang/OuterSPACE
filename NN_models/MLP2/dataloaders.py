import torch
from torchvision.datasets import MNIST
from config import data_dir, batch_size
from torchvision import transforms

data_transform = transforms.Compose([transforms.ToTensor()])

dataset = MNIST(data_dir, download=True, transform=data_transform)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [45000, 5000, 10000], generator=torch.Generator().manual_seed(42))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}