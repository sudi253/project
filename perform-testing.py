# For image processing
import cv2
# To perform path manipulations
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# For array manipulations
import numpy as np
# For saving histogram values
import joblib
# Utility Package
import cvutils

# Displaying the fake result image
def fake_img(image_path):
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imshow('FAKE!!!!', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Error: Unable to load the fake image at {image_path}")
    else:
        print(f"Error: The fake image file does not exist at {image_path}")

# Displaying the genuine result image
def genuine_img(image_path):
    img = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if img is not None and not img.empty():
        cv2.imshow('GENUINE!!!!', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Unable to load the genuine image at {image_path}")

# Load the trained model and the LBP features
model, _ = joblib.load("lbp.pkl")

# Utility function to load an image from a file path
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Unable to read image: {path}")
        return None
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.shape[2] != 3:
        print(f"Image must have 3 or 4 channels, but has {image.shape[2]} channels.")
        return None
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Utility function to extract LBP features from an image
def extract_lbp_features(image, radius, no_points):
    assert image.ndim == 2, "Input image must be grayscale (1 channel)"
    image = cv2.resize(image, (500, 500))
    gray = image
    lbp = local_binary_pattern(gray, no_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist

# Function to test a note and return whether it is genuine or fake
def test_note(image_path):
    image = load_image(image_path)
    if image is None:
        return None
    features = extract_lbp_features(image, 3, 8 * 3)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0] == 0