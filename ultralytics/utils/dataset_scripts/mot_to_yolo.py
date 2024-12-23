import os
import shutil
import pandas as pd

def get_frame_dimensions(seqinfo_path):
    """Extract frame width and height from seqinfo.ini."""
    if not os.path.exists(seqinfo_path):
        raise FileNotFoundError(f"{seqinfo_path} not found.")

    frame_width = frame_height = None
    with open(seqinfo_path, "r") as f:
        for line in f:
            if line.startswith("imwidth"):
                frame_width = int(line.split("=")[-1].strip())
            elif line.startswith("imheight"):
                frame_height = int(line.split("=")[-1].strip())

    if frame_width is None or frame_height is None:
        raise ValueError(f"imWidth or imHeight missing in {seqinfo_path}.")

    return frame_width, frame_height

def convert_mot_to_yolo(mot_folder, yolo_folder, use_track_id=False):
    # YOLO folder structure
    images_train_folder = os.path.join(yolo_folder, "images/train")
    labels_train_folder = os.path.join(yolo_folder, "labels/train")
    images_val_folder = os.path.join(yolo_folder, "images/val")
    labels_val_folder = os.path.join(yolo_folder, "labels/val")

    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(labels_train_folder, exist_ok=True)
    os.makedirs(images_val_folder, exist_ok=True)
    os.makedirs(labels_val_folder, exist_ok=True)

    # Track ID offset
    global_track_id_offset = 0

    # Parse each MOT split (train_half, val_half)
    for split in ["train_half", "val_half"]:
        split_folder = os.path.join(mot_folder, split)
        if not os.path.exists(split_folder):
            continue

        for sequence in os.listdir(split_folder):
            sequence_path = os.path.join(split_folder, sequence)
            img_folder = os.path.join(sequence_path, "img1")
            gt_file = os.path.join(sequence_path, "gt", "gt.txt")
            seqinfo_file = os.path.join(sequence_path, "seqinfo.ini")

            if not os.path.exists(img_folder) or not os.path.exists(gt_file) or not os.path.exists(seqinfo_file):
                continue

            # Get frame dimensions from seqinfo.ini
            frame_width, frame_height = get_frame_dimensions(seqinfo_file)

            # Determine YOLO output folders for this split
            images_output_folder = images_train_folder if split == "train_half" else images_val_folder
            labels_output_folder = labels_train_folder if split == "train_half" else labels_val_folder

            # Load gt.txt using pandas
            df = pd.read_csv(gt_file, header=None)
            df.columns = ["frame_id", "track_id", "x", "y", "w", "h", "conf", "class_id", "vis"]

            # Adjust track IDs to ensure uniqueness
            if use_track_id:
                df["track_id"] += global_track_id_offset

            # Calculate new offset for the next sequence
            if use_track_id:
                global_track_id_offset = df["track_id"].max() + 1

            # Filter valid classes and visibility
            valid_classes = [1, 2, 7]
            df = df[(df["class_id"].isin(valid_classes)) & (df["vis"] >= 0.1)]

            # Clip bounding boxes to image dimensions
            df["x"] = df["x"].clip(0, frame_width)
            df["y"] = df["y"].clip(0, frame_height)
            df["w"] = df["w"].clip(0, frame_width - df["x"])
            df["h"] = df["h"].clip(0, frame_height - df["y"])

            # Convert bounding boxes to YOLO format
            df["x_center"] = (df["x"] + df["w"] / 2) / frame_width
            df["y_center"] = (df["y"] + df["h"] / 2) / frame_height
            df["bbox_width"] = df["w"] / frame_width
            df["bbox_height"] = df["h"] / frame_height

            # Group by frame ID and save annotations
            for frame_id, group in df.groupby("frame_id"):
                frame_annotations = []
                for _, row in group.iterrows():
                    if use_track_id:
                        frame_annotations.append(
                            f"0 {row['x_center']:.6f} {row['y_center']:.6f} {row['bbox_width']:.6f} {row['bbox_height']:.6f} {int(row['track_id'])}"
                        )
                    else:
                        frame_annotations.append(
                            f"0 {row['x_center']:.6f} {row['y_center']:.6f} {row['bbox_width']:.6f} {row['bbox_height']:.6f}"
                        )

                # Generate output filenames based on sequence and frame ID
                output_filename = f"{sequence}_{frame_id:06d}"

                # Save corresponding YOLO label file
                label_file_path = os.path.join(labels_output_folder, f"{output_filename}.txt")
                with open(label_file_path, "w") as lf:
                    lf.write("\n".join(frame_annotations))

            # Copy images and ensure filenames match
            for img_name in os.listdir(img_folder):
                img_path = os.path.join(img_folder, img_name)
                frame_id = int(img_name.split(".")[0].split("_")[-1])
                output_filename = f"{sequence}_{frame_id:06d}"
                output_img_path = os.path.join(images_output_folder, f"{output_filename}.jpg")
                shutil.copy(img_path, output_img_path)


# Example usage
mot_folder = "./../../tracker/evaluation/TrackEval/data/gt/mot_challenge/MOT17"
yolo_folder = ""    # Output folder for YOLO annotations and images
convert_mot_to_yolo(mot_folder, yolo_folder, use_track_id=True)