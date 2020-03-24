import cv2
from model import GestureModel
import numpy as np

model = GestureModel("model.json", "weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray_fr, (128, 128))
        pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(fr, pred, (150, 45), font, 1, (255, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
