import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import torchvision.transforms.functional as TF

class ResizeAndPad(nn.Module):
    def __init__(self, target_height, target_width, fill=0):
        super(ResizeAndPad, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.fill = fill

    def forward(self, image):
        # Convert the input image to a PyTorch tensor
        image = TF.to_tensor(image)

        # Resize the image while preserving the aspect ratio
        _, height, width = image.shape
        aspect_ratio = width / height
        new_height = self.target_height
        new_width = int(aspect_ratio * new_height)

        if new_width > self.target_width:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)

        resize = TF.resize(image, (new_height, new_width))

        # Pad the resized image to the target size
        pad_height = self.target_height - new_height
        pad_width = self.target_width - new_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        padded_image = TF.pad(resize, padding, fill=self.fill)

        # Convert the padded image back to a PIL Image
        padded_image = TF.to_pil_image(padded_image)

        return padded_image


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

def get_all_paths(dir_path: str, accepted_extensions: list) -> list:
    import os
    all_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].replace(".", "")
            if file_extension in accepted_extensions:
                all_paths.append(os.path.join(root, file))
    return all_paths

def get_dataloader(dir_paths: list,
                    transform_list : list = None, 
                    batch_size = 32, shuffle = True,
                    even_load: bool = True,
                    example_plot: bool=True,
                    train_percentage: float = 0.8) -> dict:
    target_height = 384
    target_width = 384
    resize_and_pad = ResizeAndPad(target_height, target_width)
    resize_transform = resize_and_pad
    convert_to_tensor = transforms.ToTensor()
    must_transforms = [resize_transform, convert_to_tensor]
    transform = transforms.Compose(must_transforms + [t for t in transform_list if t is not None] if transform_list else must_transforms)
    
    data_paths = []
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory path {dir_path} does not exist.")
        if not os.path.isdir(dir_path):
            raise ValueError(f"Path {dir_path} is not a directory.")
        data_paths.extend([(path, 0 if 'not_greets' in path else 1) for path in get_all_paths(dir_path, ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"])])
    
    print("Data paths: ", len(data_paths))
    if even_load:
        min_class = min(len([1 for _, label in data_paths if label == 0]), len([1 for _, label in data_paths if label == 1]))
        data_paths = [path for path in data_paths if path[1] == 0][:min_class] + [path for path in data_paths if path[1] == 1][:min_class]
        print("Number of examples per class: ", min_class)
        print("Even load: ", len(data_paths))

    # Split data_paths into train and eval based on train_percentage
    train_size = int(train_percentage * len(data_paths))
    train_data_paths = data_paths[:train_size]
    eval_data_paths = data_paths[train_size:]

    dataloaders = {}
    train_dataset = GreetingDataset(train_data_paths, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    dataloaders['train'] = train_dataloader

    eval_dataset = GreetingDataset(eval_data_paths, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    dataloaders['eval'] = eval_dataloader

    # Plot one image from the train dataset object
    if example_plot:
        import matplotlib.pyplot as plt
        import numpy as np
        for i in range(5):
            img, label = train_dataset[i]
            img = img.permute(1, 2, 0)
            plt.imshow(img)
            plt.title(f"Label: {label}")
            plt.show()

    return dataloaders


if __name__ == "__main__":
    train_loader = get_dataloader(["greet_detection/data/greets", "greet_detection/data/not_greets"])
    for i, (img, label) in enumerate(train_loader):
        print(img.shape, label)
        if i == 5:
            break