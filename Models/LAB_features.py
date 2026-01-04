import cv2
import csv
import os
from engin import ImageMask
import numpy as np
from skimage import img_as_ubyte
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_chroma(a, b):
    chroma = np.sqrt(a**2 + b**2)
    chroma_mean = np.mean(chroma)
    chroma_std = np.std(chroma)
    return chroma_mean, chroma_std

def visualize_lab_channels(l, a, b):
    st.title("LAB Color Space Visualization")

    # Create heatmaps for each channel
    st.subheader("L Channel (Lightness)")
    fig, ax = plt.subplots()
    sns.heatmap(l.reshape(-1, 1), cmap="YlGnBu", cbar=True, ax=ax)
    st.pyplot(fig)

    st.subheader("A Channel (Green-Red)")
    fig, ax = plt.subplots()
    sns.heatmap(a.reshape(-1, 1), cmap="RdYlGn", cbar=True, ax=ax)
    st.pyplot(fig)

    st.subheader("B Channel (Blue-Yellow)")
    fig, ax = plt.subplots()
    sns.heatmap(b.reshape(-1, 1), cmap="coolwarm", cbar=True, ax=ax)
    st.pyplot(fig)


def color_output(cap, fps):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Load the image
        image = frame

        # Apply mask and convert to LAB color space
        image, mask = ImageMask(image, 2)
        image = img_as_ubyte(image)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Create a 2D mask for selected pixels
        selectd_pixels = image != 255
        mask_2d = selectd_pixels[:, :, 0]

        # Extract individual LAB channels
        l = lab_image[:, :, 0]
        a = lab_image[:, :, 1]
        b = lab_image[:, :, 2]

        # Apply the mask to keep only selected pixels
        l[~mask_2d] = 0
        a[~mask_2d] = 0
        b[~mask_2d] = 0

        # Rescale LAB values to [0-100, -128-127, -128-127]
        l = l.astype('float') / 255 * 100
        a = a.astype('float') - 128
        b = b.astype('float') - 128

        yield l, a, b


            



