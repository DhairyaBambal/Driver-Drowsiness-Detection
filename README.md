# Driver Drowsiness Detection

A real-time computer vision system that detects driver drowsiness using eye movements and Eye Aspect Ratio (EAR).

## Overview

Driver Drowsiness Detection monitors a driver's eyes through a webcam and identifies prolonged eye closure. When drowsiness is detected, the system activates an emergency alarm.

## Features

- Real-time webcam monitoring
- Face and eye landmark detection
- Eye Aspect Ratio (EAR) calculation
- Drowsiness detection using consecutive frames
- Emergency audio alarm
- Drowsiness event counting
- Monitoring time tracking
- Drowsiness percentage
- Streamlit web interface

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Streamlit

## Project Structure

```text
Driver-Drowsiness-Detection/
│
├── app.py
├── main.py
├── classifier.py
├── features.py
├── face_landmarker.task
├── haarcascade_eye.xml
└── haarcascade_frontalface_default.xml

## Running the Project

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy scipy streamlit
```

Run the application:

```bash
streamlit run app.py
```

## Output

The application displays:

- Current EAR
- Drowsiness status
- Number of drowsy events
- Monitoring time
- Drowsiness percentage

## Future Enhancements

- Head pose estimation
- Yawning detection
- Mobile application support
- Machine learning based classification
- Driver fatigue analysis

## Author

Dhairya Dinkar Bambal
