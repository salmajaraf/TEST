from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import time
import PoseModule as pm
from io import BytesIO
import base64

app = Flask(__name__)
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global count, dir, pTime
    url = 0
    cap = cv2.VideoCapture(url)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
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

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Use generator to yield frames
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
