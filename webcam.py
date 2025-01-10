import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import pandas as pd
from datetime import datetime
import time
import math

# Load the optimized MobileNet model
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

# Initialize a dictionary to store color codes and previous object positions for speed estimation
colorcodes = {}
previous_positions = {}

# Define some real-world object heights for estimation (in meters)
object_heights = {
    'Person': 1.7,  # Average height of a person in meters
    'Vehicle': 1.5,  # Approximate height of a vehicle
    'Bird': 0.3,  # Average height of a bird
    'Airplane': 3.0,  # Approximate height of an airplane (from ground)
    'Drone': 3.0
}

# Focal length (in pixels)
focal_length = 800

# Known reference points (top-left and bottom-right GPS coordinates)
reference_points = {
    'top_left': {'lat': 19.380000, 'lon': 72.820000, 'pixel': (0, 0)},
    'bottom_right': {'lat': 19.375000, 'lon': 72.825000, 'pixel': (1280, 720)}
}


# Function to calculate the distance to an object using the pinhole camera model
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

# Function to calculate real-world distance moved between two positions in meters
def calculate_real_world_distance(prev_pos, curr_pos, prev_dist, curr_dist):
    if prev_pos is None or curr_pos is None or prev_dist is None or curr_dist is None:
        return None

    # Pixel distance moved
    dist_in_pixels = math.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)

    # Convert pixel distance to real-world distance (m) using the average of the previous and current distances
    avg_distance = (prev_dist + curr_dist) / 2  # Use the average distance as the object's approximate distance
    if avg_distance > 0:
        real_world_distance = (dist_in_pixels / focal_length) * avg_distance
        return real_world_distance
    return None

# Function to calculate speed of the object in meters per second
def calculate_speed(prev_pos, curr_pos, prev_dist, curr_dist, time_diff):
    real_world_distance = calculate_real_world_distance(prev_pos, curr_pos, prev_dist, curr_dist)
    if real_world_distance is not None and time_diff > 0:
        return real_world_distance / time_diff
    return None

# Function to draw bounding boxes, distances, and speed
def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color, distance=None, speed=None):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=2)

    FONT_SCALE = 0.5
    THICKNESS_SCALE = 1

    cv2.rectangle(image, (left, top - 20), (right, top), color=color, thickness=-1)

    label = namewithscore
    if distance is not None:
        label += f" Dist: {distance:.2f}m"
    if speed is not None:
        label += f" Speed: {speed:.2f} m/s"

    cv2.putText(image, label, (left, top - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE,
                thickness=THICKNESS_SCALE, color=(255, 255, 255))

# Function to process detections and draw them on the image
def draw(image, boxes, classnames, scores, img_width, img_height, time_diff):
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=10, iou_threshold=0.4, score_threshold=0.3)
    detections = []

    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        score = scores[i]
        height_in_pixels = ymax - ymin  # Perceived height of the object in image space
        perceived_height = height_in_pixels * img_height  # Height of the bounding box in pixels

        object_height = object_heights.get(classname.capitalize(), None)
        distance = None
        if object_height is not None:
            distance = calculate_distance(object_height, perceived_height)

        left, top, right, bottom = int(xmin * img_width), int(ymin * img_height), int(xmax * img_width), int(ymax * img_height)
        center_position = ((left + right) // 2, (top + bottom) // 2)  # Center of the object in pixels
        gps_coords = pixel_to_gps(center_position[0], center_position[1], reference_points, img_width, img_height)

        # Speed estimation
        prev_pos = previous_positions.get(classname, None)
        prev_dist = previous_positions.get(f"{classname}_dist", None)
        speed = calculate_speed(prev_pos, center_position, prev_dist, distance, time_diff)
        previous_positions[classname] = center_position  # Update previous position
        previous_positions[f"{classname}_dist"] = distance  # Update previous distance

        detections.append((classname, score, distance, speed, gps_coords))

        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0, 255, 30)
            c2 = random.randrange(0, 255, 25)
            c3 = random.randrange(0, 255, 50)
            colorcodes.update({classname: (c1, c2, c3)})
            color = colorcodes[classname]

        namewithscore = "{}:{}".format(classname, int(100 * score))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color, distance, speed)

    return image, detections

# Initialize webcam input
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

df = pd.DataFrame(columns=["Time", "Class", "Score", "Distance (m)", "Speed (m/s)", "GPS Coordinates (lat, lon)"])

# Initialize the last save time
last_save_time = time.time()
save_interval = 30  # Save data every 30 seconds

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        converted_img = tf.image.convert_image_dtype(frame_rgb, tf.float32)[tf.newaxis, ...]

        detection = model(converted_img)
        result = {key: value.numpy() for key, value in detection.items()}

        current_time = time.time()
        time_diff = current_time - previous_time  # Time between frames
        previous_time = current_time

        frame_with_boxes, detections = draw(frame, result['detection_boxes'], result['detection_class_entities'],
                                            result["detection_scores"], original_width, original_height, time_diff)

        cv2.imshow('Video Object Detection with Speed Estimation (m/s)',
                   cv2.resize(frame_with_boxes, (original_width, original_height)))

        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if detections:
            new_rows = pd.DataFrame(
                [(current_time_str, classname, score, dist, speed, gps) for classname, score, dist, speed, gps in
                 detections],
                columns=["Time", "Class", "Score", "Distance (m)", "Speed (m/s)", "GPS Coordinates (lat, lon)"])
            df = pd.concat([df, new_rows], ignore_index=True)

        # Check if 30 seconds have passed since the last save
        if current_time - last_save_time >= save_interval:
            # Save the DataFrame to a CSV file
            df.to_csv("detections_with_speed.csv", index=False)
            print("Detections saved in CSV format")
            last_save_time = current_time  # Update the last save time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Split the GPS coordinates into separate columns
    df[['Latitude', 'Longitude']] = pd.DataFrame(df['GPS Coordinates (lat, lon)'].tolist(), index=df.index)

    # Drop the original GPS column
    df = df.drop(columns=["GPS Coordinates (lat, lon)"])

    # Save the DataFrame to a CSV file at the end of the session
    df.to_csv("detections_with_speed_final.csv", index=False)
    print("Final detections saved in CSV format")
