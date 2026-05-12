import streamlit as st
from streamlit_drawable_canvas import st_canvas

import numpy as np
import cv2
import joblib

# -----------------------
# Load model
# -----------------------
model = joblib.load("ml_extra_trees_fixed_new.pkl")
#    r"C:\Python\Python_kurs\AI bok\ml_extra_trees_fixed_new.pkl"
#)

# -----------------------
# Center image
# -----------------------
def center_image(img):

    coords = np.column_stack(np.where(img > 0))

    if len(coords) == 0:
        return img

    cy, cx = coords.mean(axis=0)

    shiftx = int(np.round(img.shape[1] / 2.0 - cx))
    shifty = int(np.round(img.shape[0] / 2.0 - cy))

    M = np.float32([
        [1, 0, shiftx],
        [0, 1, shifty]
    ])

    centered = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        borderValue=0
    )

    return centered


# -----------------------
# FULL preprocessing
# SAME as training script
# -----------------------
def preprocess_digit(gray):

    # -----------------------
    # Threshold
    # -----------------------
    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------
    # Find digit
    # -----------------------
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    digit = thresh[y:y+h, x:x+w]

    # -----------------------
    # Resize keeping ratio
    # -----------------------
    h_, w_ = digit.shape

    target_size = 18

    if h_ > w_:
        new_h = target_size
        new_w = int(w_ * (target_size / h_))
    else:
        new_w = target_size
        new_h = int(h_ * (target_size / w_))

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # -----------------------
    # Pad to 28x28
    # -----------------------
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    digit = canvas

    # -----------------------
    # Center image
    # -----------------------
    digit = center_image(digit)

    # -----------------------
    # Blur to match MNIST
    # -----------------------
    digit = cv2.GaussianBlur(
        digit,
        (3, 3),
        0
    )

    # -----------------------
    # Normalize
    # -----------------------
    digit = digit.astype(np.float32) / 255.0

    return digit


# -----------------------
# Streamlit UI
# -----------------------
st.title("MNIST Digit Recognition")

st.write("Draw a digit below:")

# -----------------------
# Canvas
# -----------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=5,
    stroke_color="white",
    background_color="black",
    height=200,  #small writting area. Code handles only small written numbers
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------
# Predict
# -----------------------
if st.button("Predict"):

    if canvas_result.image_data is None:
        st.warning("Draw something first.")
        st.stop()

    # -----------------------
    # Convert to grayscale
    # -----------------------
    img = canvas_result.image_data.astype(np.uint8)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGBA2GRAY
    )

    # -----------------------
    # Full preprocessing
    # -----------------------
    digit = preprocess_digit(gray)

    # -----------------------
    # Show processed image
    # -----------------------
    st.image(
        digit,
        width=200,
        caption="Processed 28x28 Input"
    )

    # -----------------------
    # Flatten
    # -----------------------
    img_flat = digit.reshape(1, -1)

    # -----------------------
    # Predict
    # -----------------------
    prediction = model.predict(img_flat)[0]

    probabilities = model.predict_proba(img_flat)[0]

    # -----------------------
    # Display prediction
    # -----------------------
    st.subheader(f"Prediction: {prediction}")

    st.subheader("Probabilities")

    st.bar_chart({
        str(i): float(probabilities[i])
        for i in range(10)
    })