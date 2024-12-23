import sys
import os
# Add the root directory of your project to the Python path
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker.evaluation.mot_callback import mot_eval
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml

from datetime import datetime
from functools import partial

# Load a model
model = YOLO("yolo11s-jde.yaml", task="jde").load('./../models/yolo11s-jde-tbhs.pt')
model.add_callback("on_val_end", partial(mot_eval, period=3))

# Train the model
results = model.train(
    data="mot17.yaml",
    device=[5],
    batch=8,
    fraction=0.1,
    epochs=3,
    project='reid_xps',
    name='debug' + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'),
    amp=False,
)

