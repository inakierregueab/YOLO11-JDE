import os
import cv2
import yaml
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

from ultralytics import YOLO
from tracker.evaluation.evaluate import trackeval_evaluation
from ultralytics.trackers import BYTETracker, BOTSORT, SMILEtrack, BoostTrack, JDETracker, YOLOJDETracker

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT, "smiletrack": SMILEtrack, "boosttrack": BoostTrack, "jdetracker": JDETracker, "yolojdetracker": YOLOJDETracker}

# TODO: CALLBACKS DO NOT WORK WHEN USING DDP
def mot_eval(validator, period=1):
    is_train = validator.training   # Check if the model is being trained
    if is_train:
        if validator.epoch % period != 0 or validator.epoch == 1:
            return  # Evaluate only every 'period' epochs after the first epoch

    # TODO: DEFINE DATASET, externalize
    dataset_name = 'MOT17/val_half'
    seqmap_file = './tracker/evaluation/TrackEval/data/gt/mot_challenge/seqmaps/MOT17-val_half.txt'

    # Define sequences paths to evaluate
    dataset_root = os.path.join('./tracker/evaluation/TrackEval/data/gt/mot_challenge/', dataset_name)
    seq_names = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

    # Define output folder
    output_folder = os.path.join(str(validator.save_dir), dataset_name, 'data')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize model
    model = YOLO(validator.best if is_train else validator.model_path, task=validator.args.task)

    # Initialize counters
    total_frames = 0
    total_time = 0.0

    # Iterate over sequences
    for seq_name in tqdm(seq_names, desc="Tracking sequences"):
        # Sort images
        imgs = sorted(os.listdir(os.path.join(dataset_root, seq_name, 'img1')))

        # Initialize here to restart the tracker for each sequence
        tracker_name = validator.args.tracker.split('.')[0]
        tracker_cfg = dict_to_namespace(yaml.safe_load(open(f"./ultralytics/cfg/trackers/{tracker_name}.yaml")))
        tracker = TRACKER_MAP[tracker_name](args=tracker_cfg, frame_rate=30)

        sequence_data_list = []
        for idx, img in enumerate(imgs):
            if img.endswith('.jpg') or img.endswith('.png'):
                # Get the image path
                img_path = os.path.join(dataset_root, seq_name, 'img1', img)

                # Read the image using OpenCV
                img_file = cv2.imread(img_path)

                # Warm up the model
                if idx == 0:
                    for _ in range(10):
                        _ = model.predict(
                            source=img_path,
                            verbose=False,
                            save=False,
                            conf=0.1,
                            imgsz=1280,
                            max_det=validator.args.max_det,
                            device=validator.args.device,
                            half=validator.args.half,
                            classes=[0],
                        )[0]

                # Infer on the image
                start_time = time.time()
                result = model.predict(
                    source=img_path,
                    verbose=False,
                    save=False,
                    conf=0.1,   # TODO: change to trackers' min confidence
                    imgsz=1280,
                    max_det=validator.args.max_det,     #TODO: reduce accordingly
                    device=validator.args.device,
                    half=validator.args.half,
                    classes=[0],
                )[0]

                # Process tracker's input
                det = result.boxes.cpu().numpy()

                # Update tracker
                if hasattr(tracker, "with_reid"):
                    embeds = result.embeds.data.cpu().numpy()
                    tracks = tracker.update(det, img_file, embeds) if tracker.with_reid else tracker.update(det, img_file, None)
                else:
                    tracks = tracker.update(det, img_file)

                # Update counters
                frame_time = time.time() - start_time
                total_time += frame_time
                total_frames += 1

                # Process results
                if len(tracks) == 0:
                    continue
                frame_data = np.hstack([
                    (np.ones_like(tracks[:, 0]) * (idx + 1)).reshape(-1, 1),  # Frame number
                    tracks[:, 4].reshape(-1, 1),  # Track ID
                    tracks[:, :4],  # Bbox XYXY
                ])
                sequence_data_list.append(frame_data)

        # Save results to file
        sequence_data = np.vstack(sequence_data_list)

        # Convert Bbox in indices 2:6 from TLBR to TLWH  format
        sequence_data[:, 4] -= sequence_data[:, 2]
        sequence_data[:, 5] -= sequence_data[:, 3]

        # Add confidence, class, visibility and empty columns
        constant_cols = np.ones((sequence_data.shape[0], 4)) * -1
        sequence_data = np.hstack([sequence_data, constant_cols])

        # Save results to file
        txt_path = output_folder + f'/{seq_name}.txt'
        with open(txt_path, 'w') as file:
            np.savetxt(file, sequence_data, fmt='%.6f', delimiter=',')

    # Print results
    print(f"Total frames: {total_frames}")
    print(f"Total time (s): {total_time:.3f}")
    print(f"Mean FPS: {total_frames / total_time:.3f}")

    # Evaluate the sequences
    config = {
        'GT_FOLDER': dataset_root,
        'TRACKERS_FOLDER': '/'.join(output_folder.split('/')[:-1]),
        'TRACKERS_TO_EVAL': [''],
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 4,
        'SKIP_SPLIT_FOL': True,
        'SEQMAP_FILE': seqmap_file,
        'PRINT_CONFIG': False,
        'PRINT_RESULTS': False,
    }

    trackeval_evaluation(config)

    # Read HOTA, MOTA, and IDF1 from summary file
    summary_path = '/'.join(output_folder.split('/')[:-1]) + '/pedestrian_summary.txt'
    summary_df = pd.read_csv(summary_path, sep=' ')
    hota, mota, idf1 = summary_df.loc[0, ['HOTA', 'MOTA', 'IDF1']]
    print(f'HOTA: {hota:.3f}, MOTA: {mota:.3f}, IDF1: {idf1:.3f}')

    # Log metrics
    validator.reid_metrics.set_trackeval_metrics(hota, mota, idf1)


def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})
