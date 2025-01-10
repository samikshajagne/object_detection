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

model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]

colorcodes = {}
previous_positions = {}


object_heights = {
    'Person': 1.7,  # Average height of a person in meters
    'Vehicle': 1.5,  # Approximate height of a vehicle
    'Bird': 0.3,  # Average height of a bird
    'Airplane': 3.0,  # Approximate height of an airplane (from ground)
    'Drone': 3.0
}


focal_length = 800


reference_points = {
    'top_left': {'lat': 19.380000, 'lon': 72.820000, 'pixel': (0, 0)},
    'bottom_right': {'lat': 19.375000, 'lon': 72.825000, 'pixel': (1280, 720)}
}


def calculate_distance(object_height, perceived_height):
    if perceived_height > 0:
        return (focal_length * object_height) / perceived_height
    return None

def pixel_to_gps(pixel_x, pixel_y, ref_points, img_width, img_height):
    lat_proportion = pixel_y / img_height
    lon_proportion = pixel_x / img_width

    lat = ref_points['top_left']['lat'] + lat_proportion * (
            ref_points['bottom_right']['lat'] - ref_points['top_left']['lat'])
    lon = ref_points['top_left']['lon'] + lon_proportion * (
            ref_points['bottom_right']['lon'] - ref_points['top_left']['lon'])

    return lat, lon


def calculate_real_world_distance(prev_pos, curr_pos, prev_dist, curr_dist):
    if prev_pos is None or curr_pos is None or prev_dist is None or curr_dist is None:
        return None


    dist_in_pixels = math.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)


    avg_distance = (prev_dist + curr_dist) / 2
    if avg_distance > 0:
        real_world_distance = (dist_in_pixels / focal_length) * avg_distance
        return real_world_distance
    return None


def calculate_speed(prev_pos, curr_pos, prev_dist, curr_dist, time_diff):
    real_world_distance = calculate_real_world_distance(prev_pos, curr_pos, prev_dist, curr_dist)
    if real_world_distance is not None and time_diff > 0:
        return real_world_distance / time_diff
    return None


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


def draw(image, boxes, classnames, scores, img_width, img_height, time_diff):
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=10, iou_threshold=0.4, score_threshold=0.3)
    detections = []

    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        score = scores[i]
        height_in_pixels = ymax - ymin
        perceived_height = height_in_pixels * img_height

        object_height = object_heights.get(classname.capitalize(), None)
        distance = None
        if object_height is not None:
            distance = calculate_distance(object_height, perceived_height)

        left, top, right, bottom = int(xmin * img_width), int(ymin * img_height), int(xmax * img_width), int(
            ymax * img_height)
        center_position = ((left + right) // 2, (top + bottom) // 2)
        gps_coords = pixel_to_gps(center_position[0], center_position[1], reference_points, img_width, img_height)

        # Speed estimation
        prev_pos = previous_positions.get(classname, None)
        prev_dist = previous_positions.get(f"{classname}_dist", None)
        speed = calculate_speed(prev_pos, center_position, prev_dist, distance, time_diff)
        previous_positions[classname] = center_position
        previous_positions[f"{classname}_dist"] = distance

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


def save_detections_to_kml(detections, output_file="detections.kml"):

    doc = Document()


    kml_element = doc.createElement('kml')
    kml_element.setAttribute('xmlns', 'http://www.opengis.net/kml/2.2')
    doc.appendChild(kml_element)


    document_element = doc.createElement('Document')
    kml_element.appendChild(document_element)


    for classname, score, dist, speed, gps_coords in detections:
        if gps_coords is not None:
            lat, lon = gps_coords


            placemark_element = doc.createElement('Placemark')
            document_element.appendChild(placemark_element)


            name_element = doc.createElement('name')
            name_text = doc.createTextNode(f"{classname}: {int(score * 100)}%")
            name_element.appendChild(name_text)
            placemark_element.appendChild(name_element)


            description_element = doc.createElement('description')
           #description_text = doc.createTextNode(f"Class: {classname}, Score: {score:.2f}, Distance: {dist:.2f}m, Speed: {speed:.2f}m/s")
            description_text = doc.createTextNode(
                f"Class: {classname}, "
                f"Score: {f'{score:.2f}' if score is not None else 'N/A'}, "
                f"Distance: {f'{dist:.2f}' if dist is not None else 'N/A'}m, "
                f"Speed: {f'{speed:.2f}' if speed is not None else 'N/A'}m/s"
            )

            description_element.appendChild(description_text)
            placemark_element.appendChild(description_element)


            point_element = doc.createElement('Point')
            placemark_element.appendChild(point_element)


            coordinates_element = doc.createElement('coordinates')
            coordinates_text = doc.createTextNode(f"{lon},{lat}")
            coordinates_element.appendChild(coordinates_text)
            point_element.appendChild(coordinates_element)


    with open(output_file, 'w') as f:
        f.write(doc.toprettyxml(indent="  "))


def save_detections_to_excel(detections, output_file="detections.xlsx"):

    df = pd.DataFrame(detections, columns=["Class", "Score", "Distance (m)", "Speed (m/s)", "Latitude", "Longitude"])


    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name="Detections", index=False)
    else:
        df.to_excel(output_file, sheet_name="Detections", index=False)




cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
    input_tensor = input_tensor / 255.0

    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    boxes = detections['detection_boxes'].numpy()
    classnames = detections['detection_class_entities'].numpy()
    scores = detections['detection_scores'].numpy()

    current_time = time.time()
    time_diff = current_time - start_time
    start_time = current_time

    output_image, detected_objects = draw(frame, boxes, classnames, scores, original_width, original_height, time_diff)


    cv2.imshow('Object Detection', output_image)


    save_detections_to_kml(detected_objects, output_file="detections.kml")
    save_detections_to_excel(
        [(classname, score, dist, speed, gps[0], gps[1])
         for classname, score, dist, speed, gps in detected_objects],
        output_file="detections.xlsx")


    # Define the path to Google Earth Pro and the KML file
   # google_earth_path = r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"
    #kml_file_path = r"E:\AAAAAA\detections.kml"

    # Open Google Earth with the KML file
    #os.startfile(google_earth_path)  # This will open Google Earth Pro
    #os.startfile(kml_file_path)  # This will open the KML file directly in Google Earth Pro

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("detections Saved..")

cap.release()
cv2.destroyAllWindows()
