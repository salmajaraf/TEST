from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import PoseModule as pm

app = Flask(__name__)

detector = pm.poseDetector()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    
    count, dir = 0, 0
    if lmList:
        angle = detector.findAngle(img, 12, 14, 16)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(per, (0, 100), (650, 100))

        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    _, buffer = cv2.imencode('.jpg', img)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processed_frame': encoded_frame, 'count': count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
