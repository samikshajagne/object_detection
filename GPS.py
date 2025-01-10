# this code is for the use when we use rasberypi
import gps
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import pandas as pd
from datetime import datetime
import time
import math

# GPS setup for Raspberry Pi
def fetch_gps_coordinates():
    session = gps.gps()  # Create a GPS session
    session.stream(gps.WATCH_ENABLE)  # Start streaming GPS data
    try:
        # Wait for the first fix
        while True:
            report = session.next()
            if report['class'] == 'TPV':
                lat = report.lat
                lon = report.lon
                return lat, lon
    except Exception as e:
        print(f"GPS error: {e}")
        return None, None

# Load the optimized MobileNet model
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

# Initialize a dictionary to store color codes and previous object positions for speed estimation
colorcodes = {}
previous_positions = {}

# Define some real-world object heights for estimation (in meters)
object_heights = {
    'Person': 1.7,
    'Vehicle': 1.5,
    'Bird': 0.3,
    'Airplane': 3.0,
    'Drone': 3.0
}

# Focal length (in pixels)
focal_length = 800

# Fetch initial GPS coordinates for the camera
camera_lat, camera_lon = fetch_gps_coordinates()
reference_points = {
    'top_left': {'lat': camera_lat + 0.0001, 'lon': camera_lon - 0.0001, 'pixel': (0, 0)},
    'bottom_right': {'lat': camera_lat - 0.0001, 'lon': camera_lon + 0.0001, 'pixel': (1280, 720)}
}

# (Include the rest of your existing code here, e.g., calculate_distance, pixel_to_gps, etc.)

# In the main loop, update the camera's GPS coordinates on each iteration
try:
    frame_count = 0
    previous_time = time.time()  # Initialize time tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        # Update GPS coordinates in each frame
        camera_lat, camera_lon = fetch_gps_coordinates()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        converted_img = tf.image.convert_image_dtype(frame_rgb, tf.float32)[tf.newaxis, ...]
        detection = model(converted_img)
        result = {key: value.numpy() for key, value in detection.items()}

        # (Include the rest of your existing detection and drawing code here)

finally:
    cap.release()
    cv2.destroyAllWindows()
