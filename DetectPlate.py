import os
import shutil
import cv2
import imutils
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure required libraries are installed


filename = './video12.mp4'

# Create output directory
if os.path.exists('output'):
    shutil.rmtree('output')
os.makedirs('output')

# Read and save frames from the video
cap = cv2.VideoCapture(filename)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame', frame)
        cv2.imwrite(f"./output/frame{count}.jpg", frame)
        count += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# Process the last frame saved
car_image = imread(f"./output/frame{count-1}.jpg", as_gray=True)
car_image = imutils.rotate(car_image, 270)
print(car_image.shape)

# Convert to grayscale and then to binary image
gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()

# Label connected regions in binary image
label_image = measure.label(binary_car_image)
plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions

plate_objects_cordinates = []
plate_like_objects = []

fig, ax1 = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")
flag = 0

# Identify potential license plate regions
for region in regionprops(label_image):
    if region.area < 50:
        continue
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col

    if min_height <= region_height <= max_height and min_width <= region_width <= max_width and region_width > region_height:
        flag = 1
        plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

if flag == 1:
    plt.show()

if flag == 0:
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    plate_like_objects = []

    fig, ax1 = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")

    for region in regionprops(label_image):
        if region.area < 50:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        if min_height <= region_height <= max_height and min_width <= region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
    plt.show()
