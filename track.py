from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

"""
# Load the YOLO11 model
model = YOLO("./../models/yolo11s-jde-tbhs.pt", task="jde")

# Open the video file
video_path = "./../videos/MOT17-13.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Get the total number of frames

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
with tqdm(total=total_frames, desc="Processing Frames", unit=" frames") as pbar:
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(
                frame,
                tracker="smiletrack.yaml",
                persist=True,
                verbose=False
            )

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Update the progress bar
            pbar.update(1)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

"""

# Load the model and run the tracker with a custom configuration file
model = YOLO("./../models/yolo11s-jde-tbhs.pt", task="jde")
results = model.track(
    source="./tracker/evaluation/TrackEval/data/gt/mot_challenge/MOT17/val_half/MOT17-10-FRCNN/img1/",
    tracker="jdetracker.yaml",
    show=True,  # show the video while tracking
    persist=True,   # if input is frames, persist=True will track objects across frames
    # Plus any other YOLO arguments for prediction
    imgsz=(608,1088),
)
