# images: https://www.cityscapes-dataset.com/file-handling/?packageID=3
# labels: https://www.cityscapes-dataset.com/file-handling/?packageID=28

import os
import json
import shutil

# Define paths
root_folder = ""  # Path to the CityPersons dataset folder, e.g. "/path/to/CityPersons
gt_folder = os.path.join(root_folder, "gtBbox_cityPersons_trainval", "gtBboxCityPersons")
image_folder = os.path.join(root_folder, "leftImg8bit")
yolo_root = os.path.join(root_folder)


# Create YOLO folder structure
def create_yolo_structure(yolo_root):
    os.makedirs(os.path.join(yolo_root, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, "labels", "val"), exist_ok=True)


# Convert bounding boxes to YOLO format
def convert_bbox_to_yolo(image_width, image_height, bbox):
    x_min, y_min, box_width, box_height = bbox
    x_center = (x_min + box_width / 2) / image_width
    y_center = (y_min + box_height / 2) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return x_center, y_center, width, height


# Process JSON annotations
def process_annotations(split, yolo_root):
    gt_split_folder = os.path.join(gt_folder, split)
    image_split_folder = os.path.join(image_folder, split)
    images_dst = os.path.join(yolo_root, "images", split)
    labels_dst = os.path.join(yolo_root, "labels", split)

    for city in os.listdir(gt_split_folder):
        city_gt_folder = os.path.join(gt_split_folder, city)
        city_image_folder = os.path.join(image_split_folder, city)

        if not os.path.isdir(city_gt_folder):
            continue

        for json_file in os.listdir(city_gt_folder):
            if json_file.endswith(".json"):
                # Read JSON file
                json_path = os.path.join(city_gt_folder, json_file)
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Extract image size
                image_name = json_file.replace("_gtBboxCityPersons.json", "_leftImg8bit.png")
                image_path = os.path.join(city_image_folder, image_name)
                if not os.path.exists(image_path):
                    print(f"Image not found for {json_file}, skipping...")
                    continue

                # Extract image dimensions
                image_width = data.get("imgWidth")
                image_height = data.get("imgHeight")

                # Process bounding boxes
                yolo_annotations = []
                for obj in data["objects"]:
                    if obj["label"] != "ignore":
                        bbox = obj["bbox"]  # [x_min, y_min, width, height]
                        yolo_bbox = convert_bbox_to_yolo(image_width, image_height, bbox)
                        yolo_annotations.append(f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}")

                # Write YOLO annotation file
                yolo_file_name = f"cp_{json_file.replace('_gtBboxCityPersons.json', '.txt')}"
                yolo_label_path = os.path.join(labels_dst, yolo_file_name)
                with open(yolo_label_path, "w") as label_file:
                    label_file.write("\n".join(yolo_annotations))

                # Copy image to YOLO folder
                yolo_image_name = f"cp_{image_name.replace('_leftImg8bit.png', '.png')}"
                yolo_image_path = os.path.join(images_dst, yolo_image_name)
                shutil.copy(image_path, yolo_image_path)


# Main script
def main():
    create_yolo_structure(yolo_root)
    for split in ["train", "val"]:
        process_annotations(split, yolo_root)


if __name__ == "__main__":
    main()
