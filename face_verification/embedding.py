import sys
from pathlib import Path
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from .utils import pad_largest_images, find_closest_embedding

# Configuration Constants
FACE_VERIFICATION_PATH = 'face_verification'
FACES_PATH = 'faces'
EXAMPLE_IMAGE_PATHS = [
    "luffy_.jpg",
    "Ny.jpg",
    "The_Rock.jpg",
    "Pham_nhat_vuong.jpg",
    "Tuan.jpg",
    "Thu.jpg"
]
INPUT_IMAGE_PATH = "pham-nhat-vuong.jpg"


def load_image_paths(base_path, image_names):
    """Load image paths based on the provided base path and image names."""
    return [Path(base_path) / FACES_PATH / name for name in image_names]


def create_embedding(image_paths, face_detection=None, embedding=None):
    """Create embeddings for a list of image paths."""
    if image_paths is None:
        image_paths = load_image_paths(FACE_VERIFICATION_PATH, EXAMPLE_IMAGE_PATHS)
    
    if face_detection is None or embedding is None:
        face_detection, embedding = create_model()

    if isinstance(image_paths, list):
        pil_images = [Image.open(img_path) for img_path in image_paths]
        padded_pil_images = pad_largest_images(pil_images)
    elif isinstance(image_paths, str):
        padded_pil_images = Image.open(image_paths)
    elif isinstance(image_paths, Image.Image):
        padded_pil_images = image_paths

    imgs_cropped = face_detection(padded_pil_images)

    if imgs_cropped is None:
        return None, None

    if isinstance(image_paths, list):
        stacked_tensor = torch.stack(imgs_cropped)
    else:
        stacked_tensor = imgs_cropped.unsqueeze(0)

    img_embedding = embedding(stacked_tensor).detach()
    return img_embedding, image_paths


def create_model():
    """Create MTCNN and InceptionResnetV1 models."""
    mtcnn = MTCNN(keep_all=False, image_size=160, device='cuda' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

def run_face_verification(input_image, embed_index):
    """Run face verification."""
    mtcnn, resnet = create_model()

    if isinstance(input_image, str):
        input_image = Image.open(input_image)

    input_image_embedding, _ = create_embedding(input_image, mtcnn, resnet)
    if input_image_embedding is None:
        return None, None
    closest_index, score = find_closest_embedding(input_image_embedding, embed_index, distance_type='cosine', verbose=False)
    
    return closest_index, score
    # print(f"That is {image_paths[closest_index]} with a score of {score}")


if __name__ == "__main__":
    # Example usage
    image_paths = load_image_paths(FACE_VERIFICATION_PATH, EXAMPLE_IMAGE_PATHS)
    input_image_path = "faces/pham-nhat-vuong.jpg"

    mtcnn, resnet = create_model()

    embed_index = create_embedding(image_paths, mtcnn, resnet)
    input_image_embedding = create_embedding(input_image_path, mtcnn, resnet)
    closest_index, score = find_closest_embedding(input_image_embedding, embed_index, distance_type='cosine', verbose=False)
    print(f"That is {image_paths[closest_index]} with a score of {score}")