import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in [os.listdir(os.path.join(DATA_DIR, dir_))[0]]:
        label = img_path[0]

        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Create a blank image to draw the hand skeleton on
                blank_img = np.zeros_like(img_rgb)
                mp_drawing.draw_landmarks(
                    blank_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                x_min = (
                    min(hand_landmarks.landmark, key=lambda lm: lm.x).x
                    * img_rgb.shape[1]
                )
                y_min = (
                    min(hand_landmarks.landmark, key=lambda lm: lm.y).y
                    * img_rgb.shape[0]
                )
                x_max = (
                    max(hand_landmarks.landmark, key=lambda lm: lm.x).x
                    * img_rgb.shape[1]
                )
                y_max = (
                    max(hand_landmarks.landmark, key=lambda lm: lm.y).y
                    * img_rgb.shape[0]
                )

                x, y, w, h = (
                    int(x_min),
                    int(y_min),
                    int(x_max - x_min),
                    int(y_max - y_min),
                )

                padding = 60

                cropped_img = img_rgb[
                    max(0, y - padding) : min(y + h + padding, img_rgb.shape[0]),
                    max(0, x - padding) : min(x + w + padding, img_rgb.shape[1]),
                ]
                cropped_blank_img = blank_img[
                    max(0, y - padding) : min(y + h + padding, img_rgb.shape[0]),
                    max(0, x - padding) : min(x + w + padding, img_rgb.shape[1]),
                ]

                data.append(cropped_img)
                labels.append((dir_, cropped_blank_img))

                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(cropped_img)
                ax1.set_title("Cropped Image")
                ax2.imshow(cropped_blank_img)
                ax2.set_title("Hand Skeleton")
                plt.subplots_adjust(bottom=0.2)
                fig.text(
                    0.5, 0.1, "Directory: {}".format(label), ha="center", fontsize=10
                )
                plt.show()
                break
