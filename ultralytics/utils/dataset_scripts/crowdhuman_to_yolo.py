import os
import json
from PIL import Image


# Root directory of the CrowdHuman dataset
root_dir = ""   # Path to the CrowdHuman dataset folder, e.g. "/path/to/CrowdHuman/"

# Create the necessary folders for labels
train_labels_path = root_dir + "labels/train"
val_labels_path = root_dir + "labels/val"

os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)


# Function to convert bounding boxes into YOLOv8 format
def convert_to_yolo_format(box, img_width, img_height):
    # box format: [x1, y1, w, h]
    x, y, w, h = box
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(w), img_width - x)
    h = min(int(h), img_height - y)

    cx = (x + w / 2.) / img_width
    cy = (y + h / 2.) / img_height
    nw = float(w) / img_width
    nh = float(h) / img_height

    # Ensure the values are within the range [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    return cx, cy, nw, nh

# Function to process the .odgt file
def process_odgt_file(odgt_file, labels_path, images_path):
    with open(odgt_file, "r") as f:
        for line in f:
            data = json.loads(line)
            image_id = data['ID']
            image_file = os.path.join(images_path, f"{image_id}.jpg")
            # Open the image to get its width and height
            try:
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                print(f"Image {image_file} not found. Skipping.")
                continue

            label_file = os.path.join(labels_path, f"{image_id}.txt")
            with open(label_file, "w") as label_f:
                for obj in data['gtboxes']:
                    if obj['tag'] == "person":
                        box = obj['fbox']  # 'fbox' corresponds to the full bounding box
                        x_center, y_center, width, height = convert_to_yolo_format(box, img_width, img_height)
                        label_f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")  # '0' is the class id for 'person'


# Paths to the annotation files and image folders
train_annotations = root_dir + "annotation_train.odgt"
val_annotations = root_dir + "annotation_val.odgt"
train_images_path = root_dir + "images/train"
val_images_path = root_dir + "images/val"

# Process the annotations for both train and validation sets
process_odgt_file(train_annotations, train_labels_path, train_images_path)
process_odgt_file(val_annotations, val_labels_path, val_images_path)

print("Conversion completed!")
