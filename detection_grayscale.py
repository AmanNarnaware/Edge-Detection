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

# Function for edge detection
def edge_detection(image, filter):
    width,height = image.shape
    filter_size = filter.shape[0]

    re_size_x = width - filter_size + 1
    re_size_y = height - filter_size + 1

    result = np.zeros((re_size_x, re_size_y))
    result = np.array(result)

    for i in range(re_size_x):
        for j in range(re_size_y):
            arr = image[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.sum(arr * filter)

    return result

# Load image
image = cv2.imread(r"Vivian.png")

# Resize if needed 
image = cv2.resize(image, (192*5, 108*5))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert into Gray scale for 1 channeled image 

# Padding to retain image dimension 
gray = np.pad(gray[:, :], pad_width=1, mode='constant', constant_values=0)

cv2.imshow("Original Gray Image", gray) # Display gray image
print("Input image shape:", gray.shape)

# gray = padding(gray)

h_edges = edge_detection(gray, h_filter) # Find out horizontal edges
v_edges = edge_detection(gray, v_filter) # Find out vertical edges

combined_edges = np.sqrt(np.square(h_edges)+np.square(v_edges)) # combine both vertical and horizontal edges. Gradient Magnitude
combined_edges = np.array(combined_edges)                       # G = √(Gx² + Gy²)

# Normalize all the edges for grayscale display
h_edges = np.abs(h_edges)
h_edges = (h_edges / np.max(h_edges)) * 255
h_edges = h_edges.astype(np.uint8)

v_edges = np.abs(v_edges)
v_edges = (v_edges / np.max(v_edges)) * 255
v_edges = v_edges.astype(np.uint8)

combined_edges = np.abs(combined_edges)
combined_edges = combined_edges/np.max(combined_edges) * 255
combined_edges = combined_edges.astype(np.uint8)

# Display all the edges in image
cv2.imshow("Horizontal Edges",h_edges)
cv2.imshow("Vertical Edges",v_edges)
cv2.imshow("Combined Edges", combined_edges)
print("Output image shape:", combined_edges.shape)

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

# Function for edge detection
def edge_detection(image, filter):
    width,height = image.shape
    filter_size = filter.shape[0]

    re_size_x = width - filter_size + 1
    re_size_y = height - filter_size + 1

    result = np.zeros((re_size_x, re_size_y))
    result = np.array(result)

    for i in range(re_size_x):
        for j in range(re_size_y):
            arr = image[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.sum(arr * filter)

    return result

# Load image
image = cv2.imread(r"Vivian.png")

# Resize if needed 
image = cv2.resize(image, (192*5, 108*5))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert into Gray scale for 1 channeled image 

# Padding to retain image dimension 
gray = np.pad(gray[:, :], pad_width=1, mode='constant', constant_values=0)

cv2.imshow("Original Gray Image", gray) # Display gray image
print("Input image shape:", gray.shape)

# gray = padding(gray)

h_edges = edge_detection(gray, h_filter) # Find out horizontal edges
v_edges = edge_detection(gray, v_filter) # Find out vertical edges

combined_edges = np.sqrt(np.square(h_edges)+np.square(v_edges)) # combine both vertical and horizontal edges. Gradient Magnitude
combined_edges = np.array(combined_edges)                       # G = √(Gx² + Gy²)

# Normalize all the edges for grayscale display
h_edges = np.abs(h_edges)
h_edges = (h_edges / np.max(h_edges)) * 255
h_edges = h_edges.astype(np.uint8)

v_edges = np.abs(v_edges)
v_edges = (v_edges / np.max(v_edges)) * 255
v_edges = v_edges.astype(np.uint8)

combined_edges = np.abs(combined_edges)
combined_edges = combined_edges/np.max(combined_edges) * 255
combined_edges = combined_edges.astype(np.uint8)

# Display all the edges in image
cv2.imshow("Horizontal Edges",h_edges)
cv2.imshow("Vertical Edges",v_edges)
cv2.imshow("Combined Edges", combined_edges)
print("Output image shape:", combined_edges.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
>>>>>>> f0cc291 (Initial commit)
