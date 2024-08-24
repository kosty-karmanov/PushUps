from ultralytics.engine.results import Results
from ultralytics import YOLO
from SORT import *
import math
import cv2


class Person:
    def __init__(self, user_id: int) -> None:
        self.used_id: int = user_id
        self.pose_is_up: bool = True
        self.count: int = 0
        self.on_image: bool = True
        self.box: tuple = (0, 0, 0, 0)
        self.key_points: list = []

    def update(self, new_pose_is_up: bool, key_points: list, box: tuple) -> None:
        self.key_points = key_points
        self.box = box
        if self.pose_is_up and not new_pose_is_up:
            self.pose_is_up = False
        elif not self.pose_is_up and new_pose_is_up:
            self.pose_is_up = True
            self.count += 1
            print(f"{self.used_id} made {self.count} push ups!")


persons: dict[int, Person] = {}
tracker = Sort(max_age=10, min_hits=10, iou_threshold=0.3)
model = YOLO("yolov8m-pose.pt")


def update_persons(key_points: dict) -> None:
    for user_id in key_points:
        if user_id not in persons:
            persons[user_id] = Person(user_id)
    for user_id in persons:
        if user_id in key_points:
            persons[user_id].on_image = True
            pose = calculate_pose(key_points[user_id][0])
            if pose:
                persons[user_id].update(pose == "up", key_points[user_id][1], key_points[user_id][1])
        else:
            persons[user_id].on_image = False


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


def match_results_with_dict(tracker_results: list, temp_dict: dict) -> dict:
    """
    Соединяет результаты SORT и ключевых точек
    """
    ans_dict = {}
    for result in tracker_results:
        x1, y1, x2, y2, user_id = map(int, result)
        min_differ = 1e9
        best_key = 0
        for key in temp_dict.keys():
            current_differ = abs(x1 - key[0]) + abs(y1 - key[1]) + abs(x2 - key[2]) + abs(y2 - key[3])
            if min_differ > current_differ:
                min_differ = current_differ
                best_key = key
        ans_dict[user_id] = [temp_dict[best_key], (x1, y1, x2, y2)]
        temp_dict.pop(best_key)
    return ans_dict


def use_sort(predictions: list[Results]) -> dict[int, list]:
    """
    Использование алгоритма SORT.
    """
    temp_dict = {}
    sort_data = np.empty((0, 5))
    temp_cnt = 0
    prediction = predictions[0]
    for box in prediction.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        temp_dict[(x1, y1, x2, y2)] = prediction[temp_cnt].keypoints.xyn.numpy()[0]
        conf = math.ceil((box.conf[0] * 100)) / 100
        sort_data = np.vstack((sort_data, np.array([x1, y1, x2, y2, conf])))
        temp_cnt += 1

    tracker_results = tracker.update(sort_data)

    return match_results_with_dict(tracker_results, temp_dict)


def draw(image) -> None:
    for person in persons:
        if persons[person].on_image:
            box = persons[person].box
            push_ups = persons[person].count
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            cv2.putText(image, f"Id: {person}", (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (194, 60, 169), 2, 2)
            cv2.putText(image, f"Push-ups: {push_ups}", (box[0], box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (194, 60, 169), 2, 2)


def check(video_id: int) -> None:
    cap = cv2.VideoCapture(rf"./sources/{video_id}.mp4")
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.resize(img, (640, 480))
        predictions = list(model.predict(source=img, stream=True, verbose=False, conf=0.7))
        key_points = use_sort(predictions)
        update_persons(key_points)
        draw(img)
        cv2.imshow("img", img)
        cv2.waitKey(1)


check(0)
