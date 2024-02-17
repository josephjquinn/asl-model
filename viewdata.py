import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Check if it's a directory
        continue

    processed_image = False  # Flag to track if an image has been processed in the directory
    for img_path in os.listdir(dir_path):
        if processed_image:  # If an image has already been processed, break out of the loop
            break

        label = img_path[0]

        img = cv2.imread(os.path.join(dir_path, img_path))
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
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                # Display the hand skeleton with directory name as label
                plt.figure()
                plt.imshow(blank_img)
                plt.title(label)  # Set the title as the directory name
                plt.axis('off')  # Turn off axis
                plt.show()

        processed_image = True  # Set the flag to indicate that an image has been processed in the directory
