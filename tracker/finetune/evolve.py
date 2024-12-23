import os
import random
import string
import time
from types import SimpleNamespace

import cv2
import numpy as np
import optuna
import joblib

from functools import partial

import pandas as pd
import yaml

from ultralytics import YOLO
from tracker.evaluation.evaluate import trackeval_evaluation
from ultralytics.trackers import YOLOJDETracker


def generate_unique_tag():
    """
    Generates a unique tag for the experiment by combining the current Unix timestamp with a random 6-character
    alphanumeric string.
    Returns:
        A unique tag for the experiment
    """
    timestamp = int(time.time())  # Current Unix timestamp
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))  # Random 6-character alphanumeric string
    tag = f"exp_{timestamp}_{random_suffix}"
    return tag


def optuna_fitness_fn(trial):
    """
    Fitness function for the Optuna optimization library. This function evaluates the fitness of a solution by running
    the tracker with the specified parameters and evaluating the results using the TrackEval evaluation script.
    Args:
        trial: The Optuna trial object
    Returns:
        The fitness value of the solution
    """
    # Define sequences paths to evaluate
    dataset_root = os.path.join('./../evaluation/TrackEval/data/gt/mot_challenge/MOT17/train_half')
    seq_names = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

    output_folder = f'./outputs/' + generate_unique_tag() + '/MOT17/train_half/data'
    os.makedirs(output_folder, exist_ok=True)

    # TODO: Initialize model
    model = YOLO('./../../reid_xps/CH-jde-64b-100e_TBHS_m075_1280px_20241129-220651/weights/best.pt',task='jde')

    # Update the config with the solution
    tracker_config = dict_to_namespace(yaml.safe_load(open(f"./../../ultralytics/cfg/trackers/yolojdetracker.yaml")))

    tracker_config.track_high_thresh = trial.suggest_float("track_high_thresh", 0.1, 0.7, step=0.05)
    tracker_config.track_low_thresh = trial.suggest_float("track_low_thresh", 0.1, tracker_config.track_high_thresh, step=0.05)
    tracker_config.new_track_thresh = trial.suggest_float("new_track_thresh", 0.1, 1.0, step=0.05)

    tracker_config.first_match_thresh = trial.suggest_float("first_match_thresh", 0., 1.0, step=0.05)
    tracker_config.second_match_thresh = trial.suggest_float("second_match_thresh", 0., 1.0, step=0.05)
    tracker_config.new_match_thresh = trial.suggest_float("new_match_thresh", 0., 1.0, step=0.05)

    tracker_config.first_fuse = trial.suggest_int("first_fuse", 0, 1)
    tracker_config.second_fuse = trial.suggest_int("second_fuse", 0, 1)
    tracker_config.new_fuse = trial.suggest_int("new_fuse", 0, 1)

    tracker_config.proximity_thresh = trial.suggest_float("proximity_thresh", 0., 1.0, step=0.05)
    tracker_config.appearance_thresh = trial.suggest_float("appearance_thresh", 0., 1.0, step=0.05)
    tracker_config.gate_thresh = trial.suggest_float("gate_thresh", 0., 1.0, step=0.05)
    tracker_config.appearance_weight = trial.suggest_float("appearance_weight", 0., 1.0, step=0.05)
    tracker_config.min_weight = trial.suggest_float("min_weight", 0., 1.0, step=0.05)

    tracker_config.track_buffer = trial.suggest_int("track_buffer", 0, 100, step=10)

    # Iterate over sequences
    for seq_name in seq_names:
        # Sort images
        imgs = sorted(os.listdir(os.path.join(dataset_root, seq_name, 'img1')))

        # Initialize here to restart the tracker for each sequence
        tracker = YOLOJDETracker(args=tracker_config, frame_rate=25)

        sequence_data_list = []
        for idx, img in enumerate(imgs):
            if img.endswith('.jpg') or img.endswith('.png'):
                # Get the image path
                img_path = os.path.join(dataset_root, seq_name, 'img1', img)

                # Read the image using OpenCV
                img_file = cv2.imread(img_path)

                # Infer on the image
                result = model.predict(
                    source=img_path,
                    verbose=False,
                    save=False,
                    conf=0.1,
                    imgsz=1280,
                    max_det=300,
                    device=[3],
                    half=False,
                    classes=[0],
                )[0]

                # Process tracker's input
                det = result.boxes.cpu().numpy()

                # Update tracker
                if hasattr(tracker, "with_reid"):
                    embeds = result.embeds.data.cpu().numpy()
                    tracks = tracker.update(det, img_file, embeds) if tracker.with_reid else tracker.update(det,img_file,None)
                else:
                    tracks = tracker.update(det, img_file)

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
        if len(sequence_data_list) > 0:
            sequence_data = np.vstack(sequence_data_list)
        else:
            sequence_data = np.zeros((0, 6))

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

    # Evaluate the sequences
    seqmap_file = './../evaluation/TrackEval/data/gt/mot_challenge/seqmaps/MOT17-train_half.txt'
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
    return hota

def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})



if __name__ == "__main__":

    resume = False
    if resume:
        study = joblib.load(f"73-3df5eacda8fc_study.pkl")
    else:
        study = optuna.create_study(direction="maximize")

    # Load the configuration file
    config = dict_to_namespace(yaml.safe_load(open(f"./../../ultralytics/cfg/trackers/yolojdetracker.yaml")))

    params_to_optimize = [
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "first_match_thresh",
        "second_match_thresh",
        "new_match_thresh",
        "first_fuse",
        "second_fuse",
        "new_fuse",
        "proximity_thresh",
        "appearance_thresh",
        "gate_thresh",
        "appearance_weight",
        "min_weight",
        "track_buffer"]

    initial_params = {key: getattr(config, key) for key in params_to_optimize}
    # Enqueue trial for good starting point
    study.enqueue_trial(initial_params)

    # We could add a continuous save function to save the study every 10 trials and print the best trial
    study.optimize(
        func=optuna_fitness_fn,
        n_trials=500,
        show_progress_bar=True,
    )

    print("\nStudy Statistics: ")
    print("Best Trial:      ", study.best_trial.number)
    print("Best Value:      ", study.best_value)
    print("Best Parameters: ")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
