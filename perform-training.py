# For image processing
import cv2
# To performing path manipulations
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# For array manipulations
import numpy as np
# For saving histogram values
import joblib
# Utility Package
import cvutils
import glob

# Store the path of training images in train_images
# Store the path of training images in train_images
train_images500 = []
train_folder = r"C:\Users\sudik\OneDrive\Desktop\1Hundrednote\output"

# Modify the loop range based on the number of files you generated
i = 6
d = 30
for j in range(1, i + 1):
    for k in range(d):
        file_pattern = os.path.join(train_folder, f"id{j}_demo{k}", "*.jpg")
        ti = glob.glob(file_pattern)

        if ti:
            train_images500.extend(ti)
        

n = len(train_images500)
X_test500 = []

for train_image in train_images500:
    im = cv2.imread(train_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbours
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    X_test500.append(hist)

# Dump the data
joblib.dump((X_test500, n), "lbp.pkl", compress=3)
print("Images are being trained")
os.system("faked.py")
