import os
import cv2

# Variables
width, height = 1280, 720
folder_path = "presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(1, width)
cap.set(2, height)

# Get the list of presentation images
path_images = sorted(os.listdir(folder_path), key=len)
print(path_images)

# Variables
img_number = 0
height_size, width_size = int(120*1), int(213*1)

while True:
    # Import Images
    success, img = cap.read()
    path_full_image = os.path.join(folder_path, path_images[img_number])
    img_current = cv2.imread(path_full_image)

    # Resize the presentation slide
    img_current = cv2.resize(img_current, (width, height))

    # Adding webcam
    img_small = cv2.resize(img,(width_size, height_size))
    image_height, image_width, _ = img_small.shape
    img_current[0:height_size, 0:image_width] = img_small

    cv2.imshow("image", img)
    cv2.imshow("slides", img_current)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

