from flask import Flask, render_template_string, Response, request
import cv2
import numpy as np
import time
import PoseModule as pm

app = Flask(__name__)

# Initialize pose detector
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

# To store the latest processed frame
latest_frame = None

def process_frame(frame):
    global count, dir, pTime, latest_frame
    (h, w) = frame.shape[:2]
    img = detector.findPose(frame, False)
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

    # Store the latest processed frame
    latest_frame = img

@app.route('/')
def index():
    return "Pose Detection Server is Running"

@app.route('/video_feed', methods=['POST'])
def video_feed():
    def video_feed():
    global latest_frame
    try:
        # Read the frame from the request
        file = request.files['file']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Process the frame
        process_frame(img)

        app.logger.info("Frame received and processed successfully")
    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")

    return "Frame received and processed"

@app.route('/view_feed')
def view_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
