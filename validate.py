import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tracker.evaluation.mot_callback import mot_eval

model = YOLO('./reid_xps/CH-jde-64b-100e_TBHS_m075_1280px_20241129-220651/weights/best.pt', task="jde")
model.add_callback("on_val_start", mot_eval)

model.val(
    project='reid_xps',
    name=f'MOT20-test',
    data='crowdhuman.yaml',
    imgsz=640,
    device=[5],
    #max_det=150,
    tracker='yolojdetracker.yaml',
    half=False,
    amp=False,
)