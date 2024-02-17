import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
import time

DATA_DIR = './data'

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {}
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_dir):
        # Get the first filename in the directory
        first_filename = os.listdir(label_dir)[0]
        # Extract class name from the first filename
        class_name = first_filename.split('_')[0]
        labels_dict[int(label)] = class_name

spelled_word = ""  # Initialize spelled word
last_prediction_time = time.time()  # Initialize time of last prediction
last_predicted_letter = None  # Initialize last predicted letter

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction_proba = model.predict_proba([np.asarray(data_aux)])[0]
        predicted_class_index = int(np.argmax(prediction_proba))
        predicted_letter = labels_dict[predicted_class_index]
        confidence = prediction_proba[predicted_class_index]

        # Add recognized letter to the spelled word if the predicted letter remains the same for 3 seconds
        if predicted_letter == last_predicted_letter:
            if time.time() - last_prediction_time >= 1.5:
                spelled_word += predicted_letter
                last_prediction_time = time.time()
        else:
            last_predicted_letter = predicted_letter
            last_prediction_time = time.time()

        # Display spelled word on the screen
        cv2.putText(frame, f"Spelled Word: {spelled_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

        # Add a margin below the bounding box
        margin = 40
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (x1, y1 - margin), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
