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
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    frame = request.files['frame'].read()
    npimg = np.frombuffer(frame, np.uint8)
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
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    ret, buffer = cv2.imencode('.jpg', img)
    img = buffer.tobytes()

    return jsonify({'processed_frame': img.decode('latin1'), 'count': count})

if __name__ == '__main__':
    app.run(debug=True)
