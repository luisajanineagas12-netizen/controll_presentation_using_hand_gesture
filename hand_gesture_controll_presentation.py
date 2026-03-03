import cv2
import numpy as np
import os
import time

# --- Presentation Settings ---
frame_width, frame_height = 1280, 720
presentation_folder = "presentation"

# --- Camera Setup ---
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# --- Load Presentation Slides ---
slide_files = sorted(os.listdir(presentation_folder), key=len)
print("Slides loaded:", slide_files)

# --- Variables ---
current_slide_index = 0
preview_height, preview_width = 120, 213  # webcam preview size

# --- Cooldown Settings ---
last_slide_change_time = 0
slide_change_cooldown = 2.0  # seconds between slide changes

while True:
    frame_captured, webcam_frame = camera.read()
    if not frame_captured:
        break

    # --- Hand Detection ---
    hsv_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2HSV)
    lower_skin_range = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_range = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_frame, lower_skin_range, upper_skin_range)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(hand_contour)

        if contour_area > 5000:
            cv2.drawContours(webcam_frame, [hand_contour], 0, (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_slide_change_time > slide_change_cooldown:
                bounding_x, bounding_y, bounding_w, bounding_h = cv2.boundingRect(hand_contour)

                if bounding_x < frame_width // 3:  # Hand on LEFT side → NEXT slide
                    current_slide_index = (current_slide_index + 1) % len(slide_files)
                    last_slide_change_time = current_time
                    print("Next slide")
                elif bounding_x > 2 * frame_width // 3:  # Hand on RIGHT side → PREVIOUS slide
                    current_slide_index = (current_slide_index - 1) % len(slide_files)
                    last_slide_change_time = current_time
                    print("Previous slide")

    # --- Presentation Overlay ---
    slide_path = os.path.join(presentation_folder, slide_files[current_slide_index])
    slide_image = cv2.imread(slide_path)
    slide_image = cv2.resize(slide_image, (frame_width, frame_height))

    # Add webcam feed (with hand contour) into slide on the RIGHT side
    webcam_preview = cv2.resize(webcam_frame, (preview_width, preview_height))
    slide_image[0:preview_height, frame_width - preview_width:frame_width] = webcam_preview

    # Show both views
    cv2.imshow("Hand Detection", webcam_frame)
    cv2.imshow("Presentation Slides", slide_image)

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
