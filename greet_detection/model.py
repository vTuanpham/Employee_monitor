import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from PIL import Image


class GreetingModel(nn.Module):
    def __init__(self, freeze_resnet=True):
        super(GreetingModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        if freeze_resnet:
            # Freeze ResNet layers
            for param in self.resnet.parameters():
                param.requires_grad = False
        # Remove the fully connected layer of ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Added adaptive average pooling layer
        )

        self.fc_blocks = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Added dropout layer
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Added dropout layer
            nn.Linear(1024, 1)  # Output layer
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_blocks(x)
        return x
    
    @staticmethod
    def get_trainable_params(model):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        return sum([torch.numel(p) for p in trainable_params])



if __name__ == "__main__":
    from greet_detection.data.dataloader import get_dataloader
    is_cuda = torch.cuda.is_available()

    # Load the dataloader
    train_dataloader = get_dataloader(["greet_detection/data/greets", "greet_detection/data/not_greets"],
                                batch_size=32, transform_list=[
                                   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                   transforms.RandomAutocontrast(0.5),
                                   transforms.RandomAdjustSharpness(1.5),
                                   transforms.RandomRotation(10),
                                   transforms.RandomHorizontalFlip(), 
                                ],
                                 even_load=True)
    
    eval_dataloader = get_dataloader(["greet_detection/data/google_crawl/greets", "greet_detection/data/google_crawl/not_greets"],
                                batch_size=64, transform_list=None,
                                even_load=False)

    # Initialize the model, loss function, and optimizer
    model = GreetingModel(freeze_resnet=True).to('cuda' if is_cuda else 'cpu')
    print(model.get_trainable_params(model))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001, last_epoch=-1)


    import matplotlib.pyplot as plt

    # Training loop
    num_epochs = 10
    total_loss = 0
    total_batches = 0
    losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            inputs, labels = inputs.to('cuda' if is_cuda else 'cpu'), labels.to('cuda' if is_cuda else 'cpu')
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            losses.append(loss.item())

        lr_scheduler.step()

        for inputs, labels in eval_dataloader:
            inputs, labels = inputs.to('cuda' if is_cuda else 'cpu'), labels.to('cuda' if is_cuda else 'cpu')
            outputs = model(inputs)
            eval_loss = criterion(outputs.squeeze(), labels.float())
            eval_losses.append(eval_loss.item())
        
        average_eval_loss = sum(eval_losses) / len(eval_losses)
        average_loss = total_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}, Average Eval Loss: {average_eval_loss}')


    # Plot the loss and the eval loss
    plt.plot(losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss by Steps')
    plt.legend()
    plt.savefig('greet_detection/checkpoints/loss_plot.png')

    # Save the trained model
    torch.save(model.state_dict(), 'greet_detection/checkpoints/greeting_model.pth')