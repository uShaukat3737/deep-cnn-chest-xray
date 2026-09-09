import csv
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}


def build_transforms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf


class ChestXrayDataset(Dataset):
    def __init__(self, manifest_path, transform):
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self.items = [(row["path"], row["label"]) for row in reader]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), LABEL_TO_IDX[label]


def compute_dataset_stats(manifest_path):
    to_tensor = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    ds = ChestXrayDataset(manifest_path, transform=to_tensor)
    pixels = [ds[i][0] for i in range(len(ds))]
    stacked = pixels[0].new_empty((len(pixels), *pixels[0].shape))
    for i, p in enumerate(pixels):
        stacked[i] = p
    mean = stacked.mean(dim=(0, 2, 3)).tolist()
    std = stacked.std(dim=(0, 2, 3)).tolist()
    return mean, std


def build_sample_weights(labels):
    counts = Counter(labels)
    return [1.0 / counts[label] for label in labels]
