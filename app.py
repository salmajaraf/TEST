from flask import Flask, request, Response, jsonify
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

@app.route('/')
def index():
    return "Pose Detection Server is Running"

@app.route('/video_feed', methods=['POST'])
def video_feed():
    global count, dir, pTime

    response_data = {"percentage": 0, "feedback": "No data"}
    try:
        # Read the frame from the request
        file = request.files['file']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Log the size of the received image data
        print("Received image data size:", len(np_img), "bytes")

        # Process the frame
        (h, w) = img.shape[:2]
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img)

        feedback = ""
        per = 0

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
                feedback = "Great! Keep it up!"
                if dir == 0:
                    count += 0.5
                    dir = 1
            elif per <= 5 and per2 <= 5:
                color = (0, 255, 0)
                feedback = "Nice! Almost there!"
                if dir == 1:
                    count += 0.5
                    dir = 0
            else:
                feedback = "Keep going!"

        response_data["percentage"] = per
        response_data["feedback"] = feedback

    except Exception as e:
        response_data["feedback"] = f"Error: {e}"

    # Log the response data
    print("Sending response:", response_data)

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
