import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import pandas as pd
import time
import altair as alt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

h1, h2, h3 {
    color: white;
}

[data-testid="metric-container"] {
    background: #1C1F26;
    border: 1px solid #2E3440;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = YOLO("best.pt")

# =========================================================
# CLASS NAMES
# =========================================================
class_names = {
    0: 'auto',
    1: 'two_wheelers',
    2: 'bus',
    3: 'vehicle_truck',
    4: 'car',
    5: 'tractor',
    6: 'bicycle',
    7: 'tempo',
    8: 'ambulance'
}

# =========================================================
# BOX COLORS
# =========================================================
box_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 255),
    7: (255, 128, 0),
    8: (0, 0, 255)
}

# =========================================================
# SPEED COLORS
# =========================================================
speed_colors = {
    0: (0, 255, 255),
    1: (255, 255, 255),
    2: (255, 255, 0),
    3: (0, 255, 0),
    4: (0, 0, 255),
    5: (255, 0, 0),
    6: (255, 0, 255),
    7: (0, 165, 255),
    8: (0, 255, 255)
}

# =========================================================
# TITLE
# =========================================================
st.markdown("""
<h1 style='text-align:center;'>
🚦 AI SMART TRAFFIC MANAGEMENT SYSTEM
</h1>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙ Control Panel")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Detection", "Video Detection"]
)

# =========================================================
# 📌 NEW: INPUT SOURCE FOR EXTERNAL CAMERA
# =========================================================
camera_source = st.sidebar.radio(
    "Select Video Source",
    ["Upload Video", "Webcam", "IP Camera"]
)

ip_url = None
if camera_source == "IP Camera":
    ip_url = st.sidebar.text_input(
        "Enter IP Camera URL",
        "http://192.168.0.100:8080/video"
    )

# =========================================================
# IMAGE DETECTION
# =========================================================
if mode == "Image Detection":

    uploaded_image = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = cv2.resize(frame, (1000, 550))

        results = model(frame)
        boxes = results[0].boxes

        counts = {name: 0 for name in class_names.values()}
        emergency_detected = False

        for box in boxes:

            cls = int(box.cls[0])
            class_name = class_names[cls]
            counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = box_colors[cls]

            if class_name == "ambulance":
                emergency_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            cv2.putText(
                frame,
                class_name,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        total_vehicles = sum(counts.values())

        if total_vehicles < 10:
            traffic_status = "LOW"
        elif total_vehicles < 25:
            traffic_status = "MEDIUM"
        else:
            traffic_status = "HIGH"

        k1, k2, k3 = st.columns(3)

        k1.metric("🚗 Vehicles", total_vehicles)
        k2.metric("🚦 Traffic", traffic_status)
        k3.metric("🚑 Emergency", "ACTIVE" if emergency_detected else "NONE")

        c1, c2, c3 = st.columns([0.2, 5, 0.2])

        with c2:
            st.image(frame, channels="BGR", use_container_width=True)

# =========================================================
# VIDEO + CAMERA DETECTION (UPDATED)
# =========================================================
elif mode == "Video Detection":

    cap = None

    uploaded_video = None

    # ================================
    # SOURCE SELECTION LOGIC
    # ================================
    if camera_source == "Upload Video":

        uploaded_video = st.file_uploader(
            "📤 Upload Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

    elif camera_source == "Webcam":
        cap = cv2.VideoCapture(0)

    elif camera_source == "IP Camera":
        if ip_url:
            cap = cv2.VideoCapture(ip_url)

    # =====================================================
    # PROCESS ONLY IF CAP IS READY
    # =====================================================
    if cap is not None and cap.isOpened():

        kpi_row = st.empty()

        c1, c2, c3 = st.columns([0.2, 5, 0.2])

        with c2:
            st.subheader("🎥 Live Detection")
            video_placeholder = st.empty()

        left_bottom, right_bottom = st.columns(2)

        with left_bottom:
            st.subheader("📊 Vehicle Count Table")
            table_placeholder = st.empty()

        with right_bottom:
            st.subheader("📈 Vehicle Analytics")
            chart_placeholder = st.empty()

        status_placeholder = st.empty()

        counted_ids = set()
        total_counts = {name: 0 for name in class_names.values()}
        object_positions = {}
        object_speeds = {}
        accident_detected = False

        prev_time = time.time()
        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1000, 550))

            emergency_detected = False

            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes

            current_time = time.time()

            if boxes.id is not None:

                ids = boxes.id.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                for i, box in enumerate(xyxy):

                    x1, y1, x2, y2 = map(int, box)

                    obj_id = int(ids[i])
                    cls = int(classes[i])
                    class_name = class_names[cls]

                    if class_name == "ambulance":
                        emergency_detected = True

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if obj_id not in counted_ids:
                        counted_ids.add(obj_id)
                        total_counts[class_name] += 1

                    speed = 0

                    if obj_id in object_positions:

                        prev_x, prev_y = object_positions[obj_id]

                        distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                        time_diff = current_time - prev_time
                        speed = distance / (time_diff + 1e-5)

                        box_width = x2 - x1
                        box_height = y2 - y1
                        aspect_ratio = box_width / (box_height + 1e-5)

                        if speed > 150 and aspect_ratio > 2:
                            accident_detected = True

                    object_positions[obj_id] = (cx, cy)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colors[cls], 1)

                    cv2.putText(frame, class_name, (x1, y1 - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[cls], 1)

                    cv2.putText(frame, f"Speed:{speed:.1f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_colors[cls], 1)

            prev_time = current_time

            video_placeholder.image(frame, channels="BGR", use_container_width=True)

            frame_count += 1

            if frame_count % 10 == 0:

                df = pd.DataFrame(list(total_counts.items()), columns=["Vehicle Type", "Count"])
                total_vehicle_number = sum(total_counts.values())

                if total_vehicle_number < 10:
                    traffic_status = "LOW"
                elif total_vehicle_number < 25:
                    traffic_status = "MEDIUM"
                else:
                    traffic_status = "HIGH"

                kpi_row.empty()

                with kpi_row.container():

                    kp1, kp2, kp3, kp4 = st.columns(4)

                    kp1.metric("🚗 Vehicles", total_vehicle_number)
                    kp2.metric("🚦 Traffic", traffic_status)
                    kp3.metric("🚑 Emergency", "ACTIVE" if emergency_detected else "NONE")
                    kp4.metric("⚠ Accident", "DETECTED" if accident_detected else "SAFE")

                table_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                chart = alt.Chart(df).mark_bar().encode(
                    x='Vehicle Type',
                    y='Count'
                )

                chart_placeholder.altair_chart(chart, use_container_width=True)

        cap.release()

        st.success("✅ Video Processing Completed")

    else:
        st.warning("⚠ Please select a valid video source or connect camera.")