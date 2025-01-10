import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import pandas as pd
from datetime import datetime
import time
import math
import streamlit as st
import io
import threading

# Load the optimized MobileNet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

# Initialize dictionaries for color codes
colorcodes = {}

# Define real-world object heights (in meters)
object_heights = {
    'Person': 1.7,
    'Vehicle': 1.5,
    'Bird': 0.3,
    'Airplane': 3.0,
    'Drone': 3.0
}

# Focal length (in pixels)
focal_length = 800

# Reference GPS coordinates
reference_points = {
    'top_left': {'lat': 40.712776, 'lon': -74.005974, 'pixel': (0, 0)},
    'bottom_right': {'lat': 40.703776, 'lon': -73.995974, 'pixel': (1280, 720)}
}


# Function to calculate the distance to an object
def calculate_distance(object_height, perceived_height):
    if perceived_height > 0:
        return (focal_length * object_height) / perceived_height
    return None


# Function to convert pixel coordinates to GPS coordinates
def pixel_to_gps(pixel_x, pixel_y, ref_points, img_width, img_height):
    lat_proportion = pixel_y / img_height
    lon_proportion = pixel_x / img_width
    lat = ref_points['top_left']['lat'] + lat_proportion * (
                ref_points['bottom_right']['lat'] - ref_points['top_left']['lat'])
    lon = ref_points['top_left']['lon'] + lon_proportion * (
                ref_points['bottom_right']['lon'] - ref_points['top_left']['lon'])
    return lat, lon


# Function to draw bounding boxes and labels
def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color, distance=None):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=2)

    # Create label with distance
    label = namewithscore
    if distance is not None:
        label += f" Dist: {distance:.2f}m"

    cv2.putText(image, label, (left, top - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1,
                color=(255, 255, 255))


# Function to process detections and draw them on the image
def draw(image, boxes, classnames, scores, img_width, img_height):
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=10, iou_threshold=0.4, score_threshold=0.3)
    detections = []

    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        score = scores[i]

        # Calculate perceived height and distance
        perceived_height = (ymax - ymin) * img_height
        object_height = object_heights.get(classname.capitalize(), None)
        distance = calculate_distance(object_height, perceived_height) if object_height else None

        # Save detection information
        detections.append((classname, score, distance,
                           pixel_to_gps(int((xmin + xmax) * img_width // 2), int((ymin + ymax) * img_height // 2),
                                        reference_points, img_width, img_height)))

        if classname not in colorcodes:
            colorcodes[classname] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        drawbox(image, ymin, xmin, ymax, xmax, f"{classname}: {int(score * 100)}", colorcodes[classname], distance)

    return image, detections


# Function to capture video in a separate thread
def video_capture_thread(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame_queue.append(frame)


# Streamlit app
st.title("Real-Time Object Detection")

# Initialize webcam input
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open video.")
    st.stop()

# DataFrame for detections
if 'detections_df' not in st.session_state:
    st.session_state.detections_df = pd.DataFrame(
        columns=["Time", "Class", "Score", "Distance (m)", "GPS Coordinates (lat, lon)"])

# Video processing and streaming
frame_queue = []
frame_placeholder = st.empty()  # Placeholder for video frames

# Event to signal thread stop
stop_event = threading.Event()

# Start video capture thread
capture_thread = threading.Thread(target=video_capture_thread, args=(cap, frame_queue, stop_event), daemon=True)
capture_thread.start()

# Initialize previous_time
previous_time = time.time()

# Stop button
stop_stream = st.button("Stop Stream")

if stop_stream:
    st.write("Stopping stream...")
    stop_event.set()
    capture_thread.join()
    cap.release()

    # After exiting the loop, show the download button
    st.write("Stream stopped. You can download the detections.")

    # Create an Excel file in memory for download
    excel_file = io.BytesIO()
    st.session_state.detections_df.to_excel(excel_file, index=False, engine='openpyxl')
    excel_file.seek(0)

    # Create download button
    st.download_button(
        label="Download Detections",
        data=excel_file,
        file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_button"  # Unique key for the download button
    )
else:
    # Main detection loop while the stream is active
    while True:
        if frame_queue:
            frame = frame_queue.pop(0)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            converted_img = tf.image.convert_image_dtype(frame_rgb, tf.float32)[tf.newaxis, ...]

            detection = model(converted_img)
            result = {key: value.numpy() for key, value in detection.items()}

            current_time = time.time()
            time_diff = current_time - previous_time
            previous_time = current_time

            frame_with_boxes, detections = draw(frame, result['detection_boxes'], result['detection_class_entities'],
                                                result["detection_scores"], 640, 480)

            # Display frame in Streamlit
            img = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img, channels="RGB", use_column_width=True)

            # Create a list to hold new rows
            new_rows = []

            # Save detections to DataFrame
            for det in detections:
                class_name, score, distance, gps = det
                new_row = {
                    "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Class": class_name,
                    "Score": score,
                    "Distance (m)": distance,
                    "GPS Coordinates (lat, lon)": f"{gps[0]:.6f}, {gps[1]:.6f}"
                }
                new_rows.append(new_row)

            # Convert new rows to DataFrame and concatenate with existing DataFrame
            new_df = pd.DataFrame(new_rows)
            st.session_state.detections_df = pd.concat([st.session_state.detections_df, new_df], ignore_index=True)

