from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_data(root_folder="./data",train_ratio=0.8):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root=root_folder,
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

    return train_loader, val_loader