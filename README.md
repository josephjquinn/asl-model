# Hand Gesture Recognition using Mediapipe and Random Forest Classifier

This project is aimed at recognizing hand gestures in real-time using computer vision techniques and machine learning algorithms. This project utilizes the Mediapipe library for hand tracking and landmark detection, and a Random Forest classifier for gesture classification.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Hand Gesture Recognition is a project designed to enable real-time interpretation of hand gestures using computer vision and machine learning techniques. By harnessing the capabilities of the Mediapipe library for hand tracking and the Random Forest classifier for gesture classification, this project offers a simple yet effective solution for recognizing and understanding hand gestures.

![Figure_1](https://github.com/josephjquinn/asl-model/assets/81782398/651c56d5-bbc7-49d0-971b-fa75aba3a667)
![Figure_2](https://github.com/josephjquinn/asl-model/assets/81782398/9f25fc88-c3d2-4b69-933c-18239dc2dae2)
![Figure_3](https://github.com/josephjquinn/asl-model/assets/81782398/86428435-dec7-4268-a8ae-bc672fabcc3a)

## Features

- Real-time hand gesture recognition using webcam input.
- Provided pre trained hand gesture dataset model.
- Create your own dataset model based on custom gestures and labels.
- Training of a Random Forest classifier for gesture classification.
- Interactive spelling of words based on recognized gestures.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/josephjquinn/asl-model
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Image Collection

- Run `collect_images.py` to collect hand gesture images for training.
- Follow the prompts to specify the number of classes and images per class.

### Dataset Creation

- Run `form_dataset.py` to create dataset.

### Training

- Run `train_model.py` to train the Random Forest classifier on the collected dataset.

### Real-time Recognition

- Run `realtime_recognition.py` to perform real-time hand gesture recognition using the trained model.
- Run `spelling.py` enables users to spell out words letter by letter through captured hand gestures in real time.

### Visualizaiton

- Run `viewdata.py` to view the extacted hand vecors from each of your classes.
- Run `viewoverlay.py` to view the overlayed hand data from each of your classes.
