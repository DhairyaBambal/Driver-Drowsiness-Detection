import cv2
import mediapipe as mp
import winsound
import threading
import time

from classifier import DrowsinessDetector
from features import eye_aspect_ratio


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Eye landmark points
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# Face landmark model
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


# Detector and counter
detector = DrowsinessDetector()
drowsy_events = 0
was_drowsy = False


# Alarm control
alarm_running = False


def emergency_alarm():
    global alarm_running

    if alarm_running:
        return

    alarm_running = True

    while alarm_running:

        # Same original 1000 Hz tone
        winsound.Beep(1000, 300)

        if not alarm_running:
            break

        time.sleep(0.08)

        winsound.Beep(1000, 300)

        if not alarm_running:
            break

        time.sleep(0.08)

        winsound.Beep(1000, 300)

        # Short pause before repeating
        time.sleep(0.4)


def start_alarm():
    global alarm_running

    if not alarm_running:
        threading.Thread(
            target=emergency_alarm,
            daemon=True
        ).start()


def stop_alarm():
    global alarm_running
    alarm_running = False


# Start webcam
cap = cv2.VideoCapture(0)


# Fullscreen window
cv2.namedWindow(
    "Driver Drowsiness Detection",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Driver Drowsiness Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# Start MediaPipe
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # Detect face
        result = landmarker.detect(mp_image)


        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape


            # Left eye
            left_eye = [
                (
                    int(landmarks[i].x * w),
                    int(landmarks[i].y * h)
                )
                for i in LEFT_EYE
            ]


            # Right eye
            right_eye = [
                (
                    int(landmarks[i].x * w),
                    int(landmarks[i].y * h)
                )
                for i in RIGHT_EYE
            ]


            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2


            # Check drowsiness
            drowsy = detector.check_drowsiness(ear)


            # Count drowsiness events
            if drowsy and not was_drowsy:
                drowsy_events += 1

            was_drowsy = drowsy


            # Draw eye points
            for point in left_eye + right_eye:

                cv2.circle(
                    frame,
                    point,
                    2,
                    (0, 255, 0),
                    -1
                )


            # Display EAR
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # Display drowsy events
            cv2.putText(
                frame,
                f"Drowsy Events: {drowsy_events}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # Drowsy
            if drowsy:

                start_alarm()

                cv2.putText(
                    frame,
                    "DROWSY!",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )


            # Alert
            else:

                stop_alarm()

                cv2.putText(
                    frame,
                    "ALERT",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )


        # Show webcam
        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )


        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Stop alarm
stop_alarm()

# Close webcam
cap.release()

cv2.destroyAllWindows()