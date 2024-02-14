import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
cap = cv2.VideoCapture(1)

print("Type 'exit' to stop data collection")

while True:
    class_dir = input("Enter directory name for data pool: ")
    class_path = os.path.join(DATA_DIR, class_dir)

    if class_dir == "exit":
        exit()

    if not os.path.exists(class_path):
        os.makedirs(class_path)


    print('Collecting data for dir ' + class_dir)

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_path, '{}.jpg'.format(counter)), frame)

        counter += 1

    print("Data collection successful")

cap.release()
cv2.destroyAllWindows()
