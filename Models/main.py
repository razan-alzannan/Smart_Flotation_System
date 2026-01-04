import time
import streamlit as st
import pandas as pd
import asyncio
from Reading import read_video
from Featureclass import Feature
from test2 import velocity_output
from Stability_ import stability_output
from LAB_features import color_output   
import seaborn as sns
import matplotlib.pyplot as plt
from bubble_size_analyzer import bubble_size_analyzer
import altair as alt 
import sys
import subprocess
from pathlib import Path


# Creates a new instance of the Feature class
size = Feature()
stability = Feature()
velocity = Feature()
color = Feature()
load = Feature()

# Sets the feature name
size.feature_name = 'Size'
stability.feature_name = 'Stability'
velocity.feature_name = 'Velocity'
color.feature_name = 'Color'
load.feature_name = 'Load'

cap, fps, uploaded_file = None, None, None

# Functions

@st.cache_resource
def load_video(file_path):
    return read_video(file_path)

async def update_velocity(cap, fps, placeholder2):
    velocity_data = pd.DataFrame(columns=["Time (seconds)", "Average Velocity (cm/sec)", "Movement"])
    start_time = time.time()  # Record the start time

    for average_velocity, movement in velocity_output(cap, fps):
        elapsed_time = time.time() - start_time  # Calculate elapsed time in seconds

        with placeholder2.container():
            velocity.output1, velocity.output2 = average_velocity, movement
            velocity.output1 = velocity.output1 * 0.2 # converted from pixel values to cm.
            st.metric(label="Average Velocity (cm/sec)", value=f"{round(velocity.output1, 1)} cm/sec", delta=f"{round(velocity.output1, 1)} cm/sec") 
            st.metric(label="Movement", value=velocity.output2, delta=velocity.output2)

            # Update the DataFrame with the new data
            new_data = pd.DataFrame({
                "Time (seconds)": [elapsed_time],
                "Average Velocity (cm/sec)": [round(velocity.output1, 1)],
                "Movement": [velocity.output2]
            })
            velocity_data = pd.concat([velocity_data, new_data], ignore_index=True)

            # Create a line chart with Altair
            chart = alt.Chart(velocity_data).mark_line().encode(
                x=alt.X("Time (seconds):Q", title="Time (seconds)"),
                y=alt.Y("Average Velocity (cm/sec):Q", title="Average Velocity (cm/sec)")
            ).properties(
                width=400,
                height=300
            )

            st.altair_chart(chart, use_container_width=True)

        await asyncio.sleep(1)

async def update_stability(cap, fps, placeholder3):
    for stability_index in stability_output(cap, fps):
        with placeholder3.container():
            stability.output1 = round(abs(stability_index), 3)  # Round to 3 decimal points
            st.metric(label="Stability Time (seconds)", value=f"{stability.output1} seconds", delta=f"{stability.output1} seconds")
            await asyncio.sleep(1)

async def update_size(cap, fps, placeholder5):
    size_data = pd.DataFrame(columns=["Time", "Category", "Count"])
    start_time = time.time()  # Record the start time

    for tiny_circles, small_circles, mid_circles in bubble_size_analyzer(cap, fps):
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Update the DataFrame with the new data
        new_data = pd.DataFrame({
            "Time": [elapsed_time] * 3,
            "Category": ["Small: > 0.00019 mm", "Meduim: < 0.0014 mm", "Large: < 0.04 mm"], ################ need to be converted from pixel values to mm.
            "Count": [tiny_circles, small_circles, mid_circles]
        })
        size_data = pd.concat([size_data, new_data], ignore_index=True)

        with placeholder5.container():
            st.metric(label="Total Bubbles", value=tiny_circles + small_circles + mid_circles)

            # Create a line chart with Altair
            chart = alt.Chart(size_data).mark_line().encode(
                x="Time:Q",
                y="Count:Q",
                color="Category:N"
            ).properties(
                width=400,
                height=300
            )

            st.altair_chart(chart, use_container_width=True)

        await asyncio.sleep(1)

# async def update_color(cap, fps, placeholder4):
#     # print("Starting color analysis...")
#     for l, a, b in color_output(cap, fps):
#         # Debugging: Check the shape of l, a, and b
#         # print(f"Shape of L: {l.shape}, A: {a.shape}, B: {b.shape}")

#         # Ensure l, a, and b are 2D arrays
#         if len(l.shape) != 2 or len(a.shape) != 2 or len(b.shape) != 2:
#             raise ValueError("L, A, and B must be 2D arrays for heatmap visualization.")

#         with placeholder4.container():
#             cols = st.columns(3)  # Correct usage of st.columns
#             cols[0].metric(label="L (Lightness)", value=round(l.mean(), 1))
#             cols[1].metric(label="A (Green-Red)", value=round(a.mean(), 1))
#             cols[2].metric(label="B (Blue-Yellow)", value=round(b.mean(), 1))


         
#             st.subheader("The Dominant color in the floatation cell")
#             dominant_color = sns.color_palette("hsv", 256)[int(l.mean())]  # Get the dominant color based on L value
#             st.markdown(f"<div style='width: 100px; height: 100px; background-color: rgb({int(dominant_color[0] * 255)}, {int(dominant_color[1] * 255)}, {int(dominant_color[2] * 255)});'></div>", unsafe_allow_html=True)
#             # st.write(f"Dominant Color: L={round(l.mean(), 1)}, A={round(a.mean(), 1)}, B={round(b.mean(), 1)}") 


#         await asyncio.sleep(1)

async def update_color(cap, fps, placeholder4):
    import plotly.express as px
    import numpy as np

    # Initialize a DataFrame to store L, A, B values
    color_data = pd.DataFrame(columns=["L", "A", "B"])

    for l, a, b in color_output(cap, fps):
        # Flatten the L, A, B arrays and create a DataFrame
        new_data = pd.DataFrame({
            "L": l.flatten(),
            "A": a.flatten(),
            "B": b.flatten()
        })
        color_data = pd.concat([color_data, new_data], ignore_index=True)

        with placeholder4.container():
            # Create a 3D scatter plot using Plotly
            fig = px.scatter_3d(
                color_data.sample(n=min(1000, len(color_data))),  # Limit points for performance
                x="L",
                y="A",
                z="B",
                color="L",  # Use L values for color intensity
                title="LAB Color Space",
                labels={"L": "Lightness (L)", "A": "Green-Red (A)", "B": "Blue-Yellow (B)"},
            )
            fig.update_traces(marker=dict(size=3))  # Adjust marker size
            fig.update_layout(
                width=600,
                height=400,
                scene=dict(
                    xaxis_title="Lightness (L)",
                    yaxis_title="Green-Red (A)",
                    zaxis_title="Blue-Yellow (B)"
                )
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        await asyncio.sleep(1)

async def update_load(cap, fps, placeholder6):
    with placeholder6.container():
        if uploaded_file.name == "Loaded.mp4":

            st.markdown("<h3 style='color: blue;'>Loaded",unsafe_allow_html=True)    
 
        else:
            st.markdown("<h3 style='color: red;'>Unloaded",unsafe_allow_html=True)   
    return


# async def main_velocity_stability(cap, fps, placeholder2, placeholder3):
#     print("main_velocity_stability")
#     await asyncio.gather(
#         update_velocity(cap, fps, placeholder2),
#         update_stability(cap, fps, placeholder3)
#     )

# async def main_size_color_load(cap, fps, placeholder4,placeholder5, placeholder6):
    print("main_size_color_load")
    await asyncio.gather(
        update_color(cap, fps, placeholder4),
        update_size(cap, fps, placeholder5)
        # update_load(cap, fps, placeholder6)
    )


async def main_calls(cap, fps, placeholder2, placeholder3, placeholder4, placeholder5, placeholder6):
    # print("main_calls")
    await  asyncio.gather(
        update_velocity(cap, fps, placeholder2),
        update_stability(cap, fps, placeholder3),
        update_size(cap, fps, placeholder5),
        update_color(cap, fps, placeholder4),
        update_load(cap, fps, placeholder6)  # Uncomment if needed
    )

        
# Main function
if __name__ == "__main__":
    ### Building the streamlit dashboard ###
    st.set_page_config(
        page_title="Smart Flotation Monitoring System",
        page_icon="✅",
        layout="wide"
    )

    # Add the logo to the upper left side
    st.image("C:\\Users\\USER\\Desktop\\Projects\\CPM\\GUI\\ImageForSupplier_1675_15851333309938265.jpeg", use_container_width=False)  

    st.markdown(
        "<h1 style='color: dark blue;'>Smart Flotation Monitoring System</h1>",
        unsafe_allow_html=True
    )
    
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)


    # Create two columns
    first_column, second_column, third_column = st.columns(3)

    # first column
    with first_column:
        st.header("Operater Panel")
        st.write("This is the operator panel where the operator can monitor the flotation process in real-time.")
        placeholder1 = st.empty()

        # File uploader for uploading a video
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Read the video
            cap, fps = load_video("temp_video.mp4")

            # Display the video in the GUI
            st.video("temp_video.mp4")

    # second column
    with second_column:
        st.header("Froth Analysis")
        st.write("This is the data analysis panel where the operator can view the velocity & stability & Load analysis of the flotation process.")
        st.header("Velocity")
        placeholder2 = st.empty()
        st.header("Stability")
        placeholder3 = st.empty()
        st.header("Loading State")
        placeholder6 = st.empty()


    # third column
    with third_column:
        st.header("Froth Analysis")
        st.write("This is the data analysis panel where the operator can view the size distribution & color analysis flotation process.")
        st.header("Size Distribution")
        placeholder5 = st.empty()
        st.header("Color Distribution")
        placeholder4 = st.empty()


    if uploaded_file is not None:
        asyncio.run(main_calls(cap, fps, placeholder2, placeholder3, placeholder4, placeholder5, placeholder6))


## challenges:

# how to make all the code asynchronous? 
# well done !

# how to present the LAP values?
# try to make it as the dominating color in the video
# presenta the values, calculte, present the dominnate color in the video

# how to make the video start dirctlly in the GUI?
# search in google


# how to prsent the classification from the load model ?
# send the frame to the model, check the class, present the class as a label 

# Edit the GUI formating, and make it more readable YOKO LOGO !
#  yoko logo done


# To convert pixel values to millimeters (mm), you can use the following equation:
# Real-world size (mm) = Pixel size × Conversion factor

# Where the conversion factor is calculated as:
# Conversion factor = Real-world size of reference object (mm) / Size of reference object in pixels

# Steps:
# Measure the real-world size of a reference object in millimeters.
# Measure the size of the same object in pixels in the image.
# Calculate the conversion factor.
# Use the conversion factor to convert other pixel values to millimeters.
# Example:
# Real-world size of reference object: 50 mm
# Size of reference object in pixels: 500 pixels
# Conversion factor:
# Conversion factor = 50 mm / 500 pixels = 0.1 mm/pixel

# To convert a pixel value to mm:
# Real-world size (mm) = Pixel size × 0.1