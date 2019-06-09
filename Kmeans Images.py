"""
Contact:
Zachary Hunt
hunt.590@buckeyemail.osu.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from PIL import Image

n_colors = 20

# Test different numbers of n_colors

castle = Image.open("castle.jpg")
plt.imshow(castle)
# Replace any image with "castle.jpg"

basewidth = 640
wpercent = (basewidth/float(castle.size[0]))
hsize = int((float(castle.size[1])*float(wpercent)))
castle = castle.resize((basewidth,hsize), Image.ANTIALIAS)
plt.imshow(castle)
castle.save('castle2.jpg')
castle = np.array(castle, dtype=np.float64) / 255

# Transform into 2d numpy array
w, h, d = orginal_shape = tuple(castle.shape)
assert d == 3
image_array = np.reshape(castle, (w*h,d))
print(image_array.shape)

# Fit model on sub-sample of 1000 rows
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

# Predicting color indices from the original image
km_labels = kmeans.predict(image_array)

# Randomly getting color indices
image_random = shuffle(image_array, random_state=0)[:n_colors]
labels_random = pairwise_distances_argmin(image_random, image_array, axis=0)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(castle)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (10 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, km_labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (10 colors, Random)')
plt.imshow(recreate_image(image_random, labels_random, w, h))
plt.show()



