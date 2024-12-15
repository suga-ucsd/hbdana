#!/usr/bin/env python3

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load image
image = Image.open("../Downloads/WhatsApp Image 2024-12-14 at 21.47.03.jpeg")
image = image.resize((100, 100))  # Resize for faster processing
image_array = np.array(image).reshape(-1, 3)

# Apply KMeans for color segmentation
kmeans = KMeans(n_clusters=10)  # Adjust number of clusters as needed
kmeans.fit(image_array)
segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(100, 100, 3).astype('uint8')

# Display segmented image
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
