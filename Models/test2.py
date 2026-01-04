
import os
# os.chdir("C:\Users\USER\Desktop\Projects\CPM\GUI\Models")
import cv2 as cv
import numpy as np
import pandas as pd
import cv2
from ImageMask import ImageMask
import time
from shapely.geometry import Point, Polygon
from cv2 import calcOpticalFlowPyrLK
# from google.colab.patches import cv2_imshow 
 #https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/

# average_velocity, movement = 0, "towards"
# def updatevelocityOutput(average_velocity, movement):
#     print(f"Average velocity: {average_velocity} pixels per second.")
#     print(f"Movement: {movement} the reference area.")

# def get_velocity_output():
#     return average_velocity, movement 

def find_closest_point(reference_point, polygon):
    point = Point(reference_point)
    closest_point = polygon.boundary.interpolate(polygon.boundary.project(point))
    return np.array([closest_point.x, closest_point.y])


def velocity_output(cap, fps):
        # The video feed is read in as
        # a VideoCapture object
        cap = cap
        delta_t = 1/30
        ref_area = [(189, 109), (551, 289), (551, 288), (189, 108)]
        # ret = a boolean return value from
        # getting the frame, first_frame = the
        # first frame in the entire video sequence
        fps = cap.get(cv2.CAP_PROP_FPS)
        seconds = 4
        target_frame = int(seconds*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, first_frame = cap.read()
        polygon = Polygon(ref_area)

        # Converts frame to grayscale because we
        # only need the luminance channel for
        # detecting edges - less computationally
        # expensive
        prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        # cv.imwrite("cell1.jpg",prev_gray)
        # Creates an image filled with zero
        # intensities with the same dimensions
        # as the frame
        mask = np.zeros_like(first_frame)

        # Sets image saturation to maximum
        mask[..., 1] = 255
        i = 0
        velocities_per_second = []
        velocities = []

        while(cap.isOpened()):
            start = time.time()
            target_frame = int(seconds*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, first_frame = cap.read()
            try:
                first_frame, x, angle_mask, center_point = ImageMask(first_frame,3)
            except:
                break
            polygon = Polygon(ref_area)

            # Converts frame to grayscale because we
            # only need the luminance channel for
            # detecting edges - less computationally
            # expensive
            prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        # cv.imwrite("cell1.jpg",prev_gray)
            for j in range(1,31):
                target_frame = int(seconds*fps)+j
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                # ret = a boolean return value from getting
                # the frame, frame = the current frame being
                # projected in the video
                name ="output_frsmes\\"+ str(i) + '.png'
                i += 1
                ret, frame = cap.read()
                # cv2.imwrite(name, frame)
                if not ret:
                    break

                # Opens a new window and displays the input
                # frame
                
                frame, x, angle_mask, center_point = ImageMask(frame,3)
                # cv2.imshow("frame" ,frame)
                

                # Converts each frame to grayscale - we previously
                # only converted the first frame to grayscale
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # # Calculates dense optical flow by Farneback method
                flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                                    None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)

                # dual_tvl = cv2.optflow.DualTVL1OpticalFlow()
                # flow = dual_tvl.calc(prev_gray, gray, None)

                condition = gray < 255
                condition_3d = condition[:, :, np.newaxis]
                selected_pixels = np.where(condition_3d, flow, np.nan)
                mean_flow_vector = np.nanmean(selected_pixels, axis=(0, 1))

                # Find the closest point on the reference area boundary
                closest_point = find_closest_point(center_point, polygon)

                    # Calculate vector to the closest point
                vector_to_closest = np.array([closest_point[0] - center_point[0], closest_point[1] - center_point[1]])

                # Normalize vectors
                norm_mean_flow = mean_flow_vector / np.linalg.norm(mean_flow_vector)
                norm_vector_to_closest = vector_to_closest / np.linalg.norm(vector_to_closest)

                # Dot product
                dot_product = np.dot(norm_mean_flow, norm_vector_to_closest)

                # Determine the movement direction towards or away
                movement = "towards" if dot_product >= 0 else "away"

                # print(f"General movement is {movement} the reference area.")

                # Computes the magnitude and angle of the 2D vectors
                magnitude, angle = cv.cartToPolar(selected_pixels[..., 0], selected_pixels[..., 1])
                # edges_angle = angle[angle_mask > 0]
                # edges_angle[edges_angle < 4.7] = 1
                # edges_angle[edges_angle >= 4.7] = -1

                # Set the color of the mask based on the angle
                mask[..., 0] = angle * 180 / np.pi / 2

                # Sets image value according to the optical flow
                # magnitude (normalized)
                mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

                # Converts HSV to RGB (BGR) color representation
                rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

                # Opens a new window and displays the flow visualization
                # cv2.imshow( "Optical Flow'",rgb)
                # Calculate and display the average velocity magnitude
                average_velocity = np.nanmean(magnitude)/delta_t
                if movement == "away":
                    average_velocity = average_velocity*-1
                # print('Average Velocity:', average_velocity)
                velocities_per_second.append(abs(average_velocity))
                prev_gray = gray
                # updatevelocityOutput(average_velocity, movement)
                yield average_velocity, movement 

            # average_direction = np.mean(edges_angle)
            # sum_angles = np.sum(edges_angle)
            # print('Average Direction:', average_direction)
            # if average_direction < 0:
            #     print('negative Direction:', average_direction)
            #     time.sleep(3)
            average_velocity_per_second = np.mean(velocities_per_second)
            # print("average velocity per second = ", average_velocity_per_second)
            velocities.append(average_velocity_per_second)
            seconds = seconds+5
            velocities_per_second = []
            # Updates previous frame

            


            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # velocity_per_second_sensors = 0
        # Sequared_Error = 0

        # velocity_data = [velocities_per_second, velocity_per_second_sensors, Sequared_Error ]
        # record the velocities :
        df=pd.DataFrame(velocities ,columns=["velocities_per_second(Computer Vision)"])
        df.to_csv('velocity_data.csv', index=False, header=True)

        # The following frees up resources and
        # closes all windows
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    velocity_output()