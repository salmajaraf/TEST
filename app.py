from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import time
import PoseModule as pm
import base64
import os

app = Flask(__name__, template_folder=os.path.abspath(os.path.dirname(__file__)))
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global count, dir, pTime
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        # Right Arm
        angle1 = detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle2 = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle1, (50, 150), (0, 100))
        bar = np.interp(angle1, (50, 140), (650, 100))
        per2 = np.interp(angle2, (50, 150), (0, 100))

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

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (img.shape[1] - 60, img.shape[0] - 20), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (img.shape[1] - 60, img.shape[0] - 20), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (img.shape[1] - 120, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, img.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    _, buffer = cv2.imencode('.jpg', img)
    processed_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processed_frame': processed_frame, 'count': count})

if __name__ == '__main__':
    app.run(debug=True)
