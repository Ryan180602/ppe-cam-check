import math
from playsound import playsound

import threading
import cv2
from ultralytics import YOLO

# Load the models
model1 = YOLO("weights/best2.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']


def run_tracker_in_thread(filename, model, file_index):
    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        model = model1
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == "Hardhat":
                    color = (0, 204, 255)
                elif class_name == "Mask":
                    color = (222, 82, 175)
                elif class_name == "Person":
                    color = (0, 149, 255)
                elif class_name == "Safety Vest":
                    color = (255, 255, 0)
                else:
                    color = (128, 128, 128)
                if conf > 0.2:
                    cv2.rectangle(res_plotted, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(res_plotted, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    if (class_name == "NO-Hardhat") or (class_name == "NO-Mask") or (class_name == "NO-Safety Vest"):
                        label = '{WARNING}' + f'{class_name}{conf}'
                        playsound('static/sound/alarm.wav')
                    cv2.putText(res_plotted, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1,
                                lineType=cv2.LINE_AA)
        cv2.imshow(f"Stream_{file_index}", res_plotted)
        yield res_plotted

    # key = cv2.waitKey(1)
    # if key == ord('q'):
    # break

# Release video sources
    video.release()


# Define the video files for the trackers
    video_file1 = 0
    video_file2 = 1

# Create the tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 0), daemon=True)
    tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model1, 1), daemon=True)

# Start the tracker threads
    tracker_thread1.start()
    tracker_thread2.start()

# Wait for the tracker threads to finish
    tracker_thread1.join()
    tracker_thread2.join()
