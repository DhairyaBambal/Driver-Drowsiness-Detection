import streamlit as st
import cv2
import mediapipe as mp
import winsound
import threading
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from classifier import DrowsinessDetector
from features import eye_aspect_ratio


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("Driver Drowsiness Detection")
st.caption("Real-time eye-based drowsiness monitoring system")


# --------------------------------------------------
# EYE LANDMARKS
# --------------------------------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


# --------------------------------------------------
# DETECTOR
# --------------------------------------------------

detector = DrowsinessDetector()

drowsy_events = 0
was_drowsy = False

start_time = time.time()
total_frames = 0
drowsy_frames = 0


# --------------------------------------------------
# ALARM
# --------------------------------------------------

alarm_running = False


def emergency_alarm():

    global alarm_running

    if alarm_running:
        return

    alarm_running = True

    while alarm_running:

        winsound.Beep(1000, 300)

        if not alarm_running:
            break

        time.sleep(0.08)

        winsound.Beep(1000, 300)

        if not alarm_running:
            break

        time.sleep(0.08)

        winsound.Beep(1000, 300)

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


# --------------------------------------------------
# FONT
# --------------------------------------------------

FONT_PATH = r"C:\Windows\Fonts\segoeuib.ttf"

try:

    font_status = ImageFont.truetype(
        FONT_PATH,
        42
    )

    font_info = ImageFont.truetype(
        FONT_PATH,
        25
    )

except:

    font_status = ImageFont.load_default()
    font_info = ImageFont.load_default()


# --------------------------------------------------
# TEXT BOX
# --------------------------------------------------

def draw_text_box(
    frame,
    text,
    position,
    font,
    text_color=(255, 255, 255),
    box_color=(20, 20, 20)
):

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(image)

    x, y = position

    bbox = draw.textbbox(
        (x, y),
        text,
        font=font
    )

    padding = 10

    draw.rounded_rectangle(
        (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ),
        radius=10,
        fill=box_color
    )

    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_color
    )

    frame[:] = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


# --------------------------------------------------
# STREAMLIT STATISTICS
# --------------------------------------------------

st.markdown("### Live Statistics")

stat1, stat2, stat3, stat4 = st.columns(4)

time_box = stat1.empty()
event_box = stat2.empty()
ear_box = stat3.empty()
percentage_box = stat4.empty()


# --------------------------------------------------
# WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

stframe = st.empty()
status_text = st.empty()


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            st.error("Unable to access webcam.")
            break


        total_frames += 1


        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        result = landmarker.detect(mp_image)


        current_ear = 0.0


        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape


            # LEFT EYE
            left_eye = [
                (
                    int(landmarks[i].x * w),
                    int(landmarks[i].y * h)
                )
                for i in LEFT_EYE
            ]


            # RIGHT EYE
            right_eye = [
                (
                    int(landmarks[i].x * w),
                    int(landmarks[i].y * h)
                )
                for i in RIGHT_EYE
            ]


            # EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            current_ear = (
                left_ear + right_ear
            ) / 2


            # DROWSINESS
            drowsy = detector.check_drowsiness(
                current_ear
            )


            # EVENT COUNT
            if drowsy and not was_drowsy:

                drowsy_events += 1

            was_drowsy = drowsy


            # DROWSY FRAMES
            if drowsy:

                drowsy_frames += 1


            # EYE LANDMARKS
            for point in left_eye + right_eye:

                cv2.circle(
                    frame,
                    point,
                    3,
                    (0, 255, 0),
                    -1
                )


            # INFORMATION
            draw_text_box(
                frame,
                f"EAR: {current_ear:.2f}",
                (25, 20),
                font_info
            )


            draw_text_box(
                frame,
                f"Drowsy Events: {drowsy_events}",
                (25, 65),
                font_info
            )


            # STATUS
            if drowsy:

                start_alarm()

                draw_text_box(
                    frame,
                    "DROWSY!",
                    (25, 110),
                    font_status,
                    text_color=(255, 255, 255),
                    box_color=(255, 0, 0)
                )

                status_text.error(
                    "⚠️ DROWSINESS DETECTED"
                )


            else:

                stop_alarm()

                draw_text_box(
                    frame,
                    "ALERT",
                    (25, 110),
                    font_status,
                    text_color=(255, 255, 255),
                    box_color=(0, 130, 0)
                )

                status_text.success(
                    "✓ DRIVER IS ALERT"
                )


        # --------------------------------------------------
        # STATISTICS
        # --------------------------------------------------

        elapsed = time.time() - start_time

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        monitoring_time = f"{minutes:02d}:{seconds:02d}"


        if total_frames > 0:

            drowsiness_percentage = (
                drowsy_frames / total_frames
            ) * 100

        else:

            drowsiness_percentage = 0


        time_box.metric(
            "Monitoring Time",
            monitoring_time
        )

        event_box.metric(
            "Drowsy Events",
            drowsy_events
        )

        ear_box.metric(
            "Current EAR",
            f"{current_ear:.2f}"
        )

        percentage_box.metric(
            "Drowsiness",
            f"{drowsiness_percentage:.1f}%"
        )


        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        stframe.image(
            frame,
            channels="BGR",
            width="stretch"
        )


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

stop_alarm()
cap.release()