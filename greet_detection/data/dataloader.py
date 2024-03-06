import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image


class GreetingDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label

def get_all_paths(dir_path: str) -> list:
    import os
    all_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            all_paths.append(os.path.join(root, file))
    return all_paths

def get_dataloader(dir_paths: list,
                    transform_list : list = None, 
                    batch_size = 32, shuffle = True,
                    even_load: bool = True) -> DataLoader:
    resize_transform = transforms.Resize((224, 224))
    convert_to_tensor = transforms.ToTensor()
    must_transforms = [resize_transform, convert_to_tensor]
    transform = transforms.Compose(must_transforms + [t for t in transform_list if t is not None] if transform_list else must_transforms)
    
    data_paths = []
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory path {dir_path} does not exist.")
        if not os.path.isdir(dir_path):
            raise ValueError(f"Path {dir_path} is not a directory.")
        data_paths.extend([(path, 0 if 'not_greets' in path else 1) for path in get_all_paths(dir_path)])
    
    print("Data paths: ", len(data_paths))
    if even_load:
        min_class = min(len([1 for _, label in data_paths if label == 0]), len([1 for _, label in data_paths if label == 1]))
        data_paths = [path for path in data_paths if path[1] == 0][:min_class] + [path for path in data_paths if path[1] == 1][:min_class]
        print("Number of examples per class: ", min_class)
        print("Even load: ", len(data_paths))

    dataset = GreetingDataset(data_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)

    return dataloader


if __name__ == "__main__":
    train_loader = get_dataloader(["greet_detection/data/greets", "greet_detection/data/not_greets"])
    for i, (img, label) in enumerate(train_loader):
        print(img.shape, label)
        if i == 5:
            break