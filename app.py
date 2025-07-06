# yolo_training.py
import os
import shutil
import yaml
from IPython.display import Image
import torch

# 1. Setup Environment (run once)
def setup_environment():
    # Clone YOLOv5 repo
    if not os.path.exists('yolov5'):
        !git clone https://github.com/ultralytics/yolov5
    os.chdir('yolov5')
    !pip install -r requirements.txt

# 2. Prepare Dataset
def prepare_dataset():
    dataset_path = '../dataset'
    
    # Create folder structure
    os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/val', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/val', exist_ok=True)

    # Example: Move your images and label files here
    # shutil.move('your_image.jpg', f'{dataset_path}/images/train')
    # shutil.move('your_label.txt', f'{dataset_path}/labels/train')

    # Create dataset.yaml
    data = {
        'train': '../dataset/images/train',
        'val': '../dataset/images/val',
        'nc': 3,  # Number of classes
        'names': ['dog', 'bicycle', 'car']  # Your class names
    }
    
    with open('../dataset/dataset.yaml', 'w') as f:
        yaml.dump(data, f)

# 3. Train Model
def train_model():
    !python train.py --img 640 --batch 16 --epochs 50 --data ../dataset/dataset.yaml \
    --cfg models/yolov5s.yaml --weights yolov5s.pt --name mammo_detection

# 4. Test Detection
def test_detection():
    !python detect.py --weights runs/train/mammo_detection/weights/best.pt \
    --img 640 --conf 0.25 --source ../test_images
    
    # Display result
    display(Image(filename='runs/detect/exp/your_image.jpg', width=600))

if __name__ == '__main__':
    setup_environment()
    prepare_dataset()
    train_model()
    test_detection()