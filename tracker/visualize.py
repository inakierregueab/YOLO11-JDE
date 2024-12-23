import cv2
import os
import pandas as pd


def parse_annotations(annotations_path):
    """
    Parse the MOT-format annotation file.
    :param annotations_path: Path to the annotations text file.
    :return: DataFrame containing annotations.
    """
    columns = ["frame", "id", "x", "y", "width", "height", "conf", "x_world", "y_world", "z_world"]
    annotations = pd.read_csv(annotations_path, header=None, names=columns, index_col=False)
    return annotations


def display_or_save_annotations(sequence_dir, annotations_path, save_video=False, output_video_path="output.avi"):
    """
    Display or save annotations over the MOT sequence frames.
    :param sequence_dir: Directory containing sequence frames (images).
    :param annotations_path: Path to the annotations text file.
    :param save_video: Boolean flag to save the annotated frames as a video.
    :param output_video_path: Path to save the output video if save_video is True.
    """
    # Parse annotations
    annotations = parse_annotations(annotations_path)

    # List all frame files in the sequence directory
    frame_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith(('.jpg', '.png'))])

    # Initialize video writer if saving video
    video_writer = None
    if save_video and frame_files:
        first_frame = cv2.imread(os.path.join(sequence_dir, frame_files[0]))
        if first_frame is not None:
            frame_height, frame_width = first_frame.shape[:2]
            video_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'XVID'),  # Codec for .avi files
                25,  # FPS
                (frame_width, frame_height)
            )

    for frame_file in frame_files:
        frame_number = int(os.path.splitext(frame_file)[0])  # Extract frame number from filename
        frame_path = os.path.join(sequence_dir, frame_file)

        # Read the frame
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Could not read frame {frame_file}")
            continue

        # Get annotations for the current frame
        frame_annotations = annotations[annotations['frame'] == frame_number]

        # Draw bounding boxes and IDs on the frame
        for _, row in frame_annotations.iterrows():
            x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
            obj_id = int(row['id'])

            # Purple if ID is 248, otherwise yellow
            #color = (255, 0, 255) if obj_id == 248 else (255, 255, 0)
            color = (255, 0, 255) if obj_id == 248 else (51, 255, 204)
            #color = (0, 0, 255) if obj_id == 248 else (0, 255, 0)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)
            # Put ID
            cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)

        if save_video and video_writer is not None:
            video_writer.write(frame)
        else:
            # Display the frame
            cv2.imshow("MOT Annotations", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    # Release resources
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


# Define paths
sequence_dir = "./evaluation/TrackEval/data/gt/mot_challenge/MOT17/test/MOT17-03-DPM/img1"
annotations_path = ""   # Path to the annotations text file

# Call the function to display annotations
display_or_save_annotations(
    sequence_dir,
    annotations_path,
    save_video=True,
    output_video_path="MOT17-03.avi"
)

