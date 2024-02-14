import os
import pickle

import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)  # Use the default webcam

number_of_classes = int(input("Enter the number of classes (signs): "))
dataset_size = int(input("Enter the number of images to capture for each sign: "))

for j in range(number_of_classes):
    label = input("Enter the label for class {}: ".format(j))
    label_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print('Collecting data for class {}: {}'.format(j, label))

    done = False
    while not done:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            done = True

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, 'Signing for: {}'.format(label), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(label_dir, '{}_{}.jpg'.format(label, counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
