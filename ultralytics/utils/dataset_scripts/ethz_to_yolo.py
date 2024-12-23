# download link: https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md
# ALL TO TRAIN

import os
import shutil


# Function to create YOLO folder structure
def create_yolo_structure(root):
    os.makedirs(os.path.join(root, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels", "val"), exist_ok=True)


# Function to process and copy files
def process_sequence(sequence_path, sequence_name, yolo_root, split="train"):
    images_src = os.path.join(sequence_path, "images")
    labels_src = os.path.join(sequence_path, "labels")
    if not os.path.exists(images_src) or not os.path.exists(labels_src):
        print(f"Skipping {sequence_name}: 'images' or 'labels' folder not found")
        return

    # Paths for target YOLO folder structure
    images_dst = os.path.join(yolo_root, "images", split)
    labels_dst = os.path.join(yolo_root, "labels", split)

    # Process image files
    for file_name in os.listdir(images_src):
        if file_name.endswith(".png"):
            src_image_path = os.path.join(images_src, file_name)
            dst_image_name = f"{sequence_name}_{file_name}"
            dst_image_path = os.path.join(images_dst, dst_image_name)
            shutil.copy(src_image_path, dst_image_path)

            # Check for corresponding label file
            label_name = file_name.replace(".png", ".txt")
            src_label_path = os.path.join(labels_src, label_name)
            if os.path.exists(src_label_path):
                dst_label_name = f"{sequence_name}_{label_name}"
                dst_label_path = os.path.join(labels_dst, dst_label_name)
                shutil.copy(src_label_path, dst_label_path)
            else:
                print(f"Label file not found: {src_label_path}")
                print(f"Deleting image file: {dst_image_path}")
                os.remove(dst_image_path)


# Main script
def main():
    ethz_root = ""  # Path to the root folder of the ETHZ dataset
    yolo_root = ethz_root

    # Create YOLO folder structure
    create_yolo_structure(yolo_root)

    # List all sequences in ETHZ root
    for sequence_name in os.listdir(ethz_root):
        sequence_path = os.path.join(ethz_root, sequence_name)
        if os.path.isdir(sequence_path) and sequence_name.startswith("eth"):
            process_sequence(sequence_path, sequence_name, yolo_root)


if __name__ == "__main__":
    main()


