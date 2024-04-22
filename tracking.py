from collections import defaultdict
import cv2
import numpy as np
from math import sqrt
from ultralytics import YOLO


model = YOLO('./pretrained/2um/weights/best.pt')
video_path = "./videos/vid3.avi"
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
track_history = defaultdict(lambda: [])



width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,' ',height)

def main():
    no_of_frames = 0
    
    while cap.isOpened():
    
        success, frame = cap.read()
        if success:
            
            results = model.track(frame, persist=True,verbose=False,conf=0.6,imgsz=1024)
            no_of_frames+=1
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot(labels=False)

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                #if len(track) > 30:  # retain 90 tracks for 90 frames
                #    track.pop(0)
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 190, 0), thickness=10)

            cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 Inference", 640,480)
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            # save output video
            # annotated_frame = cv2.resize(annotated_frame,(640,480))
            # out.write(cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 190, 0), thickness=10))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
        
    for id,positions in track_history.items():
        x_initial, x_last = positions[0][0], positions[len(positions)-1][0]
        y_initial, y_last = positions[0][1], positions[len(positions)-1][1]
        dx = x_last - x_initial
        dy = y_last - y_initial
        print(dx,dy,len(positions))
        
    print('Total frames: ',no_of_frames)   


if __name__ == '__main__':
    main()