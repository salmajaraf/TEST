from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import time
import PoseModule as pm
from PIL import Image
import io

app = Flask(__name__)
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global count, dir, pTime
    file = request.files['frame']
    img = Image.open(file.stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img)

    result = {'count': count, 'angles': []}
    if lmList:
        # Right Arm
        angle1 = detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle2 = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle1, (50, 150), (0, 100))
        bar = np.interp(angle1, (50, 140), (650, 100))
        per2 = np.interp(angle2, (50, 150), (0, 100))

        result['angles'] = [angle1, angle2]

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if 95 <= per <= 100 and 95 <= per2 <= 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per <= 5 and per2 <= 5:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        result['count'] = count

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
