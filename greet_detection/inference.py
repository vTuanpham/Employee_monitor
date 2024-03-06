import torch
import sys
sys.path.append('./')
from PIL import Image
from torchvision import transforms
from greet_detection.model import GreetingModel


# Function for inference on a single image
def predict_greeting(image_path, 
                     transform=transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor()
                     ]),
                     model_path='greet_detection/checkpoints/greeting_model.pth'):
    model = GreetingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
    probability = torch.sigmoid(output).item()
    return probability


# Example usage for inference
image_path = 'greet_detection/data/google_crawl/not_greets/000027.jpg'
prediction = predict_greeting(image_path)

# Define a threshold for classification (you may adjust this based on your needs)
threshold = 0.5
result = "Greeting" if prediction > threshold else "Not Greeting"

print(f"Prediction: {prediction}, Result: {result}")
