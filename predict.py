import imageio
import numpy as np
import os
import base64
import io
from PIL import Image, ImageOps, ImageEnhance
from tensorflow.keras.models import load_model

# Suppress warnings, don't use gpu
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Test on an imgur url
def test(model, img_rows, img_cols):
    # Grab image
    im = imageio.imread("https://i.imgur.com/Wq9CTM1.png")

    # Process as grayscale
    gray = np.dot(im, [0.2, 0.1, 0.3])  # Define our own RGB to grayscale mapping

    # Reshape and normalise
    gray = gray.reshape(1, img_rows, img_cols, 1)
    gray /= 255

    # Predict the digit
    prediction = model.predict(gray)

    return prediction.argmax()

# Predict digits from a base64 string
def predict_from_string(base64_string, model, img_rows, img_cols):
    # Grab image, sharped and convert to grayscale
    print(base64_string)
    image = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image))
    image = image.resize((img_rows, img_cols))
    image = ImageEnhance.Sharpness(image).enhance(3)
    image = ImageOps.grayscale(image)

    # Convert to numpy array and normalise
    image_array = np.array(image).reshape(1, img_rows, img_cols, 1)
    image_array = np.divide(image_array, 255)
    image_array = np.subtract(1, image_array)
    image_array = np.power(image_array, 2)

    # Predict the digit
    prediction = model.predict(image_array)

    return prediction