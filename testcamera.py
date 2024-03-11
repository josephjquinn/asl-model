import cv2
from cvzone.HandTrackingModule import HandDetector

# Set parameter to 0,1,2,3 based on your webcam configuration
cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2)
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    cv2.imshow("Result", img)
    cv2.waitKey(1)
