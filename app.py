import cv2
import numpy as np
import time
import PoseModule as pm

# url = "http://192.168.11.104:4747/video"
url = "curls1.mp4"


# Open the video stream
cap = cv2.VideoCapture(url)

# cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
while True:
    success, img = cap.read()
    # Get the dimensions of the frame
    (h, w) = img.shape[:2]

    # center = (w / 2, h / 2)
    # M = cv2.getRotationMatrix2D(center, 90, 1.0)
    # img = cv2.warpAffine(img, M, (w, h))

    # # img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("img.png")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img)
    # print(lmList)
    if len(lmList) != 0:
        # Right Arm
        angle1 = detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle2 = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle1, (50, 150), (0, 100))
        bar = np.interp(angle1, (50, 140), (650, 100))
        per2 = np.interp(angle2, (50, 150), (0, 100))
        print(angle1, per)
        print(bar)

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if 95 <= per <= 100 and 95 <= per2 <= 100 :
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per <= 5 and per2 <= 5:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (w-60, h-20), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (w-60, h-20), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (w-120, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, h-40), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
