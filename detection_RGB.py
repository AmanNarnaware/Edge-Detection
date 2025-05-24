<<<<<<< HEAD
import cv2
import numpy as np

# Sobel Filters
# Vertical Filter
v_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]  
])

# Horizontal Filter
h_filter = np.array([
    [-1,-2,-1],
    [0, 0, 0],
    [1, 2, 1]
])

image = cv2.imread("Path_to_your_image")

# Resize if needed 
image = cv2.resize(image, (192*5, 108*5))

cv2.imshow("Original Image", image)


def edge_detection(image, filter, channel):
    width,height,_ = image.shape
    filter_size = filter.shape[0]

    re_size_x = width - filter_size + 1
    re_size_y = height - filter_size + 1

    result = np.zeros((re_size_x, re_size_y))
    result = np.array(result)

    for i in range(re_size_x):
        for j in range(re_size_y):
            arr = image[i:i+filter_size, j:j+filter_size, channel]
            result[i, j] = np.sum(arr * filter)

    return result

vertical_B = edge_detection(image, v_filter, 0)
horizontal_B = edge_detection(image, h_filter, 0)
combined_B = np.sqrt(np.square(horizontal_B) + np.square(vertical_B))

vertical_G = edge_detection(image, v_filter, 1)
horizontal_G = edge_detection(image, h_filter, 1)
combined_G = np.sqrt(np.square(horizontal_G) + np.square(vertical_G))

vertical_R = edge_detection(image, v_filter, 2)
horizontal_R = edge_detection(image, h_filter, 2)
combined_R = np.sqrt(np.square(horizontal_R) + np.square(vertical_R))

combined_BGR = np.sqrt(np.square(combined_B) + np.square(combined_G) + np.square(combined_R))
combined_BGR = (combined_BGR / np.max(combined_BGR)) * 255
combined_BGR = combined_BGR.astype(np.uint8)

# cv2.imshow("Combined Edges B", combined_B)
# cv2.imshow("Combined Edges G", combined_G)
# cv2.imshow("Combined Edges R", combined_R)
cv2.imshow("Combined Edges (BGR)", combined_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.waitKey(0)
cv2.destroyAllWindows()
=======
import cv2
import numpy as np

# Sobel Filters
# Vertical Filter
v_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]  
])

# Horizontal Filter
h_filter = np.array([
    [-1,-2,-1],
    [0, 0, 0],
    [1, 2, 1]
])

image = cv2.imread("Path_to_your_image")

# Resize if needed 
image = cv2.resize(image, (192*5, 108*5))

cv2.imshow("Original Image", image)


def edge_detection(image, filter, channel):
    width,height,_ = image.shape
    filter_size = filter.shape[0]

    re_size_x = width - filter_size + 1
    re_size_y = height - filter_size + 1

    result = np.zeros((re_size_x, re_size_y))
    result = np.array(result)

    for i in range(re_size_x):
        for j in range(re_size_y):
            arr = image[i:i+filter_size, j:j+filter_size, channel]
            result[i, j] = np.sum(arr * filter)

    return result

vertical_B = edge_detection(image, v_filter, 0)
horizontal_B = edge_detection(image, h_filter, 0)
combined_B = np.sqrt(np.square(horizontal_B) + np.square(vertical_B))

vertical_G = edge_detection(image, v_filter, 1)
horizontal_G = edge_detection(image, h_filter, 1)
combined_G = np.sqrt(np.square(horizontal_G) + np.square(vertical_G))

vertical_R = edge_detection(image, v_filter, 2)
horizontal_R = edge_detection(image, h_filter, 2)
combined_R = np.sqrt(np.square(horizontal_R) + np.square(vertical_R))

combined_BGR = np.sqrt(np.square(combined_B) + np.square(combined_G) + np.square(combined_R))
combined_BGR = (combined_BGR / np.max(combined_BGR)) * 255
combined_BGR = combined_BGR.astype(np.uint8)

# cv2.imshow("Combined Edges B", combined_B)
# cv2.imshow("Combined Edges G", combined_G)
# cv2.imshow("Combined Edges R", combined_R)
cv2.imshow("Combined Edges (BGR)", combined_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.waitKey(0)
cv2.destroyAllWindows()
>>>>>>> 4baea2c172bf9e0011dd8f40af2171799906c6df
