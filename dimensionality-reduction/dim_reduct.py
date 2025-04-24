import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Define the directory where the images are located
image_directory = 'yalefaces'

# List of image file paths
image_files = []

# Resizing dimensions
resize_dimensions = (40,40)

# List of flattened images
flattened = []

# List to house all images
data = []

# ref -> https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
for file in os.listdir(image_directory):

    # Ignore Mac's hidden files
    if file.startswith('.'):
        continue

    full_path = os.path.join(image_directory, file)
    image_files.append(full_path)

    with Image.open(full_path) as img:
        resized = img.resize(resize_dimensions)
        
        # ref -> https://www.geeksforgeeks.org/python-flatten-a-2d-numpy-array-into-1d-array/
        #Convert image to numpy array so we can flatten, then append to list of flattened images
        flattened.append(np.array(resized).flatten())

data = np.array(flattened)

# Dimensionality Reduction with a PCA -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

mean_vec = np.mean(data, axis=0) # Mean
std_vec = np.std(data, axis=0) # Standard Deviation

# Standardize our data
standardized = (data - mean_vec) / std_vec

# ref -> https://numpy.org/doc/stable/reference/generated/numpy.cov.html
covariance = np.cov(standardized.T) # T: transpose

# ref -> https://numpy.org/devdocs/reference/generated/numpy.linalg.eigh.html
# Compute the eigenvectors and eigenvalues (ascending order, using 'eigh')
eigenvalues, eigenvectors = np.linalg.eigh(covariance)

# Select 2 greatest eigenvectors
projection = eigenvectors[:, -2:]

# Project data on 2 eigenvectors
transformed_data = standardized.dot(projection)

# Plot Transformed Data
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.title('2D PCA Projection of Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Eigenfaces -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Build the full path for subject02.centerlight
centerlight_path = os.path.join(image_directory, 'subject02.centerlight')

# Check if the full path is in the list and get its index
if centerlight_path in image_files:
    subject02_index = image_files.index(centerlight_path)

# Now extract the corresponding data from the standardized matrix
centerlight = standardized[subject02_index]

# Taken from BBLearn Resources -> Python Functions; create video
out = cv2.VideoWriter("homework1-eigenfaces.avi", cv2.VideoWriter_fourcc(*'XVID'), 15, (40, 40))

#𝐷 – Dimensionality of data vector (number of features)
D = data.shape[0]

# Loop through k components for reconstruction
for k in range(1, D):

    # Project the image onto the first k-principal components
    weights = centerlight.dot(eigenvectors[:, -k:])
    
    # Reconstruct the image using k-principal components
    reconstruction = weights.dot(eigenvectors[:, -k:].T)

    # Undo the standardization done to eigenvectors above
    reconstruction = (reconstruction * std_vec) + mean_vec

    # Reshape using dimensions, normalize the reconstructed image for video output
    # ref -> https://stackoverflow.com/questions/35302361/reshape-an-array-of-images
    reconstructed_image = reconstruction.reshape((40, 40)).astype(np.uint8)

    # Add grayscale color to video display (also taken from BBlearn resources)
    bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR)
    
    out.write(bgr) 

# Finalize our video file
out.release()

print("The video was saved as 'homework1-eigenfaces.avi'")


