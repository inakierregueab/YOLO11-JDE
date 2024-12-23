import sys
import os
from multiprocessing import freeze_support

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import TrackEval's evaluation tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TrackEval')))
from .TrackEval import trackeval  # noqa: E402


def trackeval_evaluation(config):
    freeze_support()

    # Use the provided config directly
    eval_config = {k: v for k, v in config.items() if k in trackeval.Evaluator.get_default_eval_config().keys()}
    dataset_config = {k: v for k, v in config.items() if
                      k in trackeval.datasets.MotChallenge2DBox.get_default_dataset_config().keys()}
    metrics_config = {k: v for k, v in config.items() if
                      k in {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5, 'PRINT_CONFIG': False}}

    # Run the evaluation
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []

    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))

    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == '__main__':
    # Define your configuration here
    gt_folder = './TrackEval/data/gt/mot_challenge/MOT17/val_half'
    seqmap_file = './TrackEval/data/gt/mot_challenge/seqmaps/MOT17-val_half.txt'
    trackers_folder = './outputs/tracks/MOT17/val_half'
    trackers_to_eval = 'testing'

    config = {
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': trackers_folder,
        'TRACKERS_TO_EVAL': [trackers_to_eval],
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 4,
        'SKIP_SPLIT_FOL': True,
        'SEQMAP_FILE': seqmap_file,
    }

    trackeval_evaluation(config)

