import os
import shutil
from configparser import ConfigParser

# Define paths
base_dir = './TrackEval/data/gt/mot_challenge/MOT17'
train_dir = os.path.join(base_dir, 'train')
train_half_dir = os.path.join(base_dir, 'train_half')
val_half_dir = os.path.join(base_dir, 'val_half')

# Create train_half and val_half directories if they do not exist
os.makedirs(train_half_dir, exist_ok=True)
os.makedirs(val_half_dir, exist_ok=True)

# Process each sequence in the train directory
for sequence in os.listdir(train_dir):
    # Consider only sequences with 'FRCNN' in the name
    if 'FRCNN' not in sequence:
        continue

    sequence_path = os.path.join(train_dir, sequence)
    img_dir = os.path.join(sequence_path, 'img1')
    gt_path = os.path.join(sequence_path, 'gt/gt.txt')
    seqinfo_path = os.path.join(sequence_path, 'seqinfo.ini')

    # Check if the img1 directory, gt.txt, and seqinfo.ini file exist
    if not os.path.isdir(img_dir) or not os.path.isfile(gt_path) or not os.path.isfile(seqinfo_path):
        continue

    # Get a list of all frames and sort them
    frame_files = sorted(os.listdir(img_dir))
    num_frames = len(frame_files)

    # Determine the index for splitting the frames in half
    mid_idx = num_frames // 2

    # Create destination folders for the train_half and val_half splits
    train_sequence_dir = os.path.join(train_half_dir, sequence)
    val_sequence_dir = os.path.join(val_half_dir, sequence)

    train_img_dir = os.path.join(train_sequence_dir, 'img1')
    val_img_dir = os.path.join(val_sequence_dir, 'img1')

    train_gt_dir = os.path.join(train_sequence_dir, 'gt')
    val_gt_dir = os.path.join(val_sequence_dir, 'gt')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(val_gt_dir, exist_ok=True)

    # Copy the first half of the images to train_half/img1 folder
    for frame_file in frame_files[:mid_idx]:
        src_frame_path = os.path.join(img_dir, frame_file)
        dst_frame_path = os.path.join(train_img_dir, frame_file)
        shutil.copy(src_frame_path, dst_frame_path)

    # Copy and rename the last half of the images for val_half/img1 folder
    for new_index, frame_file in enumerate(frame_files[mid_idx:], start=1):
        src_frame_path = os.path.join(img_dir, frame_file)
        new_frame_file = f"img_{new_index:06d}.jpg"  # Renaming to img_000001.jpg, img_000002.jpg, etc.
        dst_frame_path = os.path.join(val_img_dir, new_frame_file)
        shutil.copy(src_frame_path, dst_frame_path)

    # Process the ground truth annotations for the first half frames
    with open(gt_path, 'r') as gt_file:
        annotations = gt_file.readlines()

    train_annotations = []
    val_annotations = []

    for line in annotations:
        frame_id = int(line.split(',')[0])
        if frame_id <= mid_idx:
            train_annotations.append(line)
        else:
            # Update the frame ID for validation annotations to start from 1
            new_frame_id = frame_id - mid_idx
            updated_line = line.replace(str(frame_id), str(new_frame_id), 1)
            val_annotations.append(updated_line)

    # Save the training annotations in train_half/gt folder
    train_gt_path = os.path.join(train_gt_dir, 'gt.txt')
    with open(train_gt_path, 'w') as train_gt_file:
        train_gt_file.writelines(train_annotations)

    # Save the validation annotations in val_half/gt folder
    val_gt_path = os.path.join(val_gt_dir, 'gt.txt')
    with open(val_gt_path, 'w') as val_gt_file:
        val_gt_file.writelines(val_annotations)

    # Load and modify seqinfo.ini for train_half
    config = ConfigParser()
    config.read(seqinfo_path)
    config.set('Sequence', 'seqLength', str(mid_idx))

    train_seqinfo_path = os.path.join(train_sequence_dir, 'seqinfo.ini')
    with open(train_seqinfo_path, 'w') as configfile:
        config.write(configfile)

    # Modify seqinfo.ini for val_half
    config.set('Sequence', 'seqLength', str(num_frames - mid_idx))

    val_seqinfo_path = os.path.join(val_sequence_dir, 'seqinfo.ini')
    with open(val_seqinfo_path, 'w') as configfile:
        config.write(configfile)

print(
    "Train and validation halves, with updated seqinfo.ini and renamed frames, have been created in 'train_half' and 'val_half' folders.")



