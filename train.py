import comet_ml
from ultralytics import YOLO
from datetime import datetime
from functools import partial

"""
import os
# Set number of threads
N_THREADS = '8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS
"""

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml
comet_ml.init()

from tracker.evaluation.mot_callback import mot_eval


# Initialize model and load matching weights
model = YOLO('yolo11s-jde.yaml', task='jde').load('./../models/yolo11s.pt')

epochs = 30
batch = 32

model.add_callback("on_val_end", partial(mot_eval, period=epochs))   # Evaluate every X epochs
model.train(
    project='reid_xps',
    name=f'CH-jde-{batch}b-{epochs}e_TBHS_m075_1280px' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),

    data='crowdhuman.yaml',
    epochs=epochs,
    batch=batch,
    device=[0,1,2,3,4,5,6,7],
    # bbox_erase=0.1,
    imgsz=1280,
    # clr=0.5,
    # Freeze layers up to N-1. 24 trains only Re-ID branch, 23 trains only heads, 11 freezes backbone
    # freeze=23,

    close_mosaic=0,     # Always 0 for JDE
    patience=25,
    tracker='jdetracker.yaml',  # Tracker config file with ReID activated

    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    cache=False,
    amp=False,
)
