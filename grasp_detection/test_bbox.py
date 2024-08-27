import cv2
import os

# give input jpg, output binary mask
image_path = os.path.join(os.getcwd(), 'test.jpg')
# print(image_path)
image = cv2.imread(image_path)
cv2.imshow('Original Image', image)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# threshold the image to binary
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# output the bounding box of the mask
def get_bbox(mask):
    x, y, w, h = cv2.boundingRect(mask)
    # output the bounding box in the form of (x_min, x_max, y_min, y_max)
    return x, x+w, y, y+h

# output the bounding box of the mask
x_min, x_max, y_min, y_max = get_bbox(mask)
print(f'Bounding Box: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}')

# visualize the bounding box on the mask
def visualize_bbox(mask, x_min, x_max, y_min, y_max):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow('Bounding Box', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# visualize the bounding box on the mask
visualize_bbox(mask, x_min, x_max, y_min, y_max)    
