from flask import Flask, render_template_string, Response
import cv2
import numpy as np
import time
import PoseModule as pm

app = Flask(__name__)

# URL of the video stream
url = "http://192.168.12.20:4040/video"

# Open the video stream
cap = cv2.VideoCapture(url)

# Initialize pose detector
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

def generate_frames():
    global count, dir, pTime
    while True:
        success, img = cap.read()
        if not success:
            break

        # Get the dimensions of the frame
        (h, w) = img.shape[:2]

        # Process the image
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
            cv2.rectangle(img, (1100, 100), (w-60, h-20), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (w-60, h-20), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (w-120, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, h-40), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        # Yield the frame as part of the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    # Return an HTML page with the video stream
    return "Pose Detection Server is Running"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

