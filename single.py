from ultralytics import YOLO
import cv2

model = YOLO("yolov8m-pose.pt")


def calculate_pose(key_points: list) -> str | None:
    if len(key_points) == 0:
        return None
    if key_points[0][1] == 0 or key_points[7][1] == 0 or key_points[8][1] == 0:
        return None
    if key_points[0][1] >= key_points[8][1] and key_points[0][1] >= key_points[7][1]:
        return "down"
    elif key_points[0][1] < key_points[8][1] and key_points[0][1] < key_points[7][1]:
        return "up"
    return None


def check(video_id: int) -> int:
    cap = cv2.VideoCapture(rf"./sources/{video_id}.mp4")
    pose = "up"
    count = 0

    while cap.isOpened():
        _, img = cap.read()
        img = cv2.resize(img, (640, 480))
        predictions = list(model.predict(source=img, stream=True, verbose=False, show=True, conf=0.7))
        if len(predictions) != 0:
            key_points = predictions[0].keypoints.xyn.numpy()[0]
            new_pose = calculate_pose(key_points)
            if pose == "up" and new_pose == "down":
                pose = new_pose
            elif pose == "down" and new_pose == "up":
                pose = new_pose
                count += 1
                print("pushup")

    return count


check(0)
