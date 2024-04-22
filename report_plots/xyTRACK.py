from collections import defaultdict
import cv2
import numpy as np
from math import sqrt
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/Users/kostis/Downloads/last.pt')

# Open the video file
video_path = "/Users/kostis/Desktop/brownian VIDS/new VID/2μm/Br2_2.avi"
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)

# Store the track history
track_history = defaultdict(lambda: [])
no_of_frames = 0


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0], verbose=False, conf=0.5)
        no_of_frames += 1

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

        # Display the annotated frame
        cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Inference", 640, 480)
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        annotated_frame = cv2.resize(annotated_frame, (640, 480))

        # Plot the tracks
        for track_id, positions in track_history.items():
            points = np.array(positions, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 190, 0), thickness=10)

        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# Find the particle with the most tracked frames
max_frames_particle_id = max(track_history, key=lambda x: len(track_history[x]))

# Print the x, y coordinates for each frame tracked for the particle with the most frames
most_tracked_frames = track_history[max_frames_particle_id]
for frame_num, (x, y) in enumerate(most_tracked_frames, start=1):
    print(f"Frame {frame_num}: x={x}, y={y}")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
out.release()
