import os
import sys
import time
import cv2
from tqdm import tqdm
import numpy as np
sys.path.append('./')
from human_detection import initialize_model, process_video, process_yolo_boxes, initialize_video_capture, release_video


# Configuration Constants
VIDEO_FILE_1 = "greet_detection/videos/How to Bow in Japan_ Japanese Bowing Basics - LIVE JAPAN.mp4"
MODEL_FILE = 'yolov8n.pt'
DEVICE_ID = "0"
WINDOW_NAME = 'Greeting dataset creation' 

def create_greet_dataset(video_path, save_interval=1000, max_frames=10000):
    """
    Create a dataset of images from a video, categorizing them as greets or not greets.

    Args:
        video_path (str): The path to the video file.
        save_interval (int, optional): The interval at which to save the images. Defaults to 1000.
        max_frames (int, optional): The maximum number of frames to process. Defaults to 10000.

    Returns:
        dict: A dictionary containing the number of greets, number of not greets, and total frames processed.
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    model = initialize_model(MODEL_FILE)
    video, _, _ = initialize_video_capture(video_path)
    # Get length of video
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    tqdm.write(f"Video length: {video_length} frames")
    progress_bar = tqdm(total=video_length, position=0, leave=True, desc="Processing video")
    full_results = {}
    greets_list = []
    not_greets_list = []
    greets_num = 0
    not_greets_num = 0
    total_frames = 0
    save_folder = "greet_detection/data/greets"  # Change this to the desired folder path
    default_folder = "greet_detection/data/not_greets"  # Change this to the default folder path

    def save_images(greets_list, not_greets_list):
        for image, path in greets_list:
            cv2.imwrite(path, image)
        
        for image, path in not_greets_list:
            cv2.imwrite(path, image)

    while True:
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break
        ret, frame, results = process_video(video, model, 0, visualize=False, verbose=False, confidence=0.65)
        total_frames += 1
        progress_bar.update(1)
        if total_frames % save_interval == 0:
            print(f"Saving images at frame {total_frames}... total greets: {greets_num}, total not greets: {not_greets_num}")
            save_images(greets_list, not_greets_list)
            greets_list = []
            not_greets_list = []
        if total_frames >= max_frames:
            print(f"Maximum number of frames reached: {max_frames}")
            break
        if not ret:
            print("Video processing complete.")
            break
        for result in results:
            pil_images = process_yolo_boxes(result, frame)
            for pil_image in pil_images:
                # Convert pil image to cv2 image
                pil_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                # Check if a key is pressed
                if cv2.waitKey(1) == ord('g'):
                    # Save the pil_image in a different folder
                    save_path = os.path.join(save_folder, f"image_{total_frames}.png")
                    greets_num += 1
                    cv2.imshow(WINDOW_NAME, pil_image)
                    greets_list.append((pil_image, save_path))
                else:
                    # Save the pil_image in the default folder
                    save_path = os.path.join(default_folder, f"image_{total_frames}.png")
                    not_greets_num += 1
                    cv2.imshow(WINDOW_NAME, pil_image)
                    not_greets_list.append((pil_image, save_path))
    full_results['greets_num'] = greets_num
    full_results['not_greets_num'] = not_greets_num
    full_results['total_frames'] = total_frames
    release_video(video)

    return full_results

def clean_up():
    # Clean up all data
    os.system("rm -rf greet_detection/data/greets/*")
    os.system("rm -rf greet_detection/data/not_greets/*")
    print("Clean up complete.")

def main():
    full_results = create_greet_dataset(VIDEO_FILE_1)
    print(full_results)
    # clean_up()

if __name__ == "__main__":
    main()