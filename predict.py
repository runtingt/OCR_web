import imageio
import numpy as np
import os

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