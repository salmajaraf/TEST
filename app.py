import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

class poseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

detector = poseDetector()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    img = detector.findPose(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    
    _, buffer = cv2.imencode('.jpg', img)
    img_str = buffer.tobytes()
    return jsonify({'processed_frame': img_str.decode('latin1'), 'count': len(lmList)})

if __name__ == "__main__":
    app.run()
