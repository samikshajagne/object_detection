import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import pandas as pd
from datetime import datetime
import time
import math
from xml.dom.minidom import Document
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

        left, top, right, bottom = int(xmin * img_width), int(ymin * img_height), int(xmax * img_width), int(
            ymax * img_height)
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

# Function to save detections to KML format
def save_detections_to_kml(detections, output_file="detections.kml"):
    # Create a KML document
    doc = Document()

    # Create the KML element
    kml_element = doc.createElement('kml')
    kml_element.setAttribute('xmlns', 'http://www.opengis.net/kml/2.2')
    doc.appendChild(kml_element)

    # Create a Document element
    document_element = doc.createElement('Document')
    kml_element.appendChild(document_element)

    # Loop over the detections to create Placemark elements
    for classname, score, dist, speed, gps_coords in detections:
        if gps_coords is not None:
            lat, lon = gps_coords

            # Create Placemark element
            placemark_element = doc.createElement('Placemark')
            document_element.appendChild(placemark_element)

            # Create Name element
            name_element = doc.createElement('name')
            name_text = doc.createTextNode(f"{classname}: {int(score * 100)}%")
            name_element.appendChild(name_text)
            placemark_element.appendChild(name_element)

            # Create description element
            description_element = doc.createElement('description')
            description_text = doc.createTextNode(
                f"Class: {classname}, "
                f"Score: {f'{score:.2f}' if score is not None else 'N/A'}, "
                f"Distance: {f'{dist:.2f}' if dist is not None else 'N/A'}m, "
                f"Speed: {f'{speed:.2f}' if speed is not None else 'N/A'}m/s"
            )
            description_element.appendChild(description_text)
            placemark_element.appendChild(description_element)

            # Create Point element
            point_element = doc.createElement('Point')
            placemark_element.appendChild(point_element)

            # Create coordinates element
            coordinates_element = doc.createElement('coordinates')
            coordinates_text = doc.createTextNode(f"{lon},{lat},0")
            coordinates_element.appendChild(coordinates_text)
            point_element.appendChild(coordinates_element)

    # Save the KML document
    with open(output_file, "w") as f:
        f.write(doc.toprettyxml(indent="  "))

# Function to create charts
def create_charts(detections):
    if not detections:
        return

    classes = [d[0] for d in detections]
    scores = [d[1] for d in detections]
    distances = [d[2] for d in detections]
    speeds = [d[3] for d in detections]

    # Bar chart for scores
    plt.figure(figsize=(10, 5))
    sns.barplot(x=classes, y=scores)
    plt.title('Object Detection Scores')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Pie chart for classes
    class_counts = {cls: classes.count(cls) for cls in set(classes)}
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Object Detection Class Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Start video capture
video_source = 0  # Change to the appropriate video source
cap = cv2.VideoCapture(video_source)

frame_count = 0
frame_interval = 10  # Update charts every 10 frames
detections_data = []  # To store detections for charting

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width, _ = frame.shape
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform inference
    results = model(input_tensor)

    # Extract boxes, class names, and scores
    boxes = results["detection_boxes"][0].numpy()
    classnames = results["detection_class_entities"][0].numpy()
    scores = results["detection_scores"][0].numpy()

    # Process detections
    frame_with_detections, current_detections = draw(frame, boxes, classnames, scores, img_width, img_height, 1/30)  # Assuming a frame rate of 30 FPS

    # Append current detections to the list for charting
    detections_data.extend(current_detections)

    # Display the frame with detections
    cv2.imshow("Object Detection", frame_with_detections)

    # Update charts every `frame_interval` frames
    if frame_count % frame_interval == 0:
        create_charts(detections_data)  # Create charts with current detections

    frame_count += 1

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save detections to KML file when done
save_detections_to_kml(detections_data)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
