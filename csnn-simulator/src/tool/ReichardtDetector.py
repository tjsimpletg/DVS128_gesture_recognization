import cv2
import numpy as np
from skimage.util import random_noise

THRESHOLD = 0.05
NOISE = False

def reichardt_detector(prev_gray_frame: np.ndarray,current_gray_frame: np.ndarray, frame: np.ndarray) -> None:
    # Left - Right direction
    shift_left_current_gray_frame = np.pad(current_gray_frame,((0,0),(1,0)), mode='constant')[:, :-1] 
    reichardt_left = np.multiply(prev_gray_frame,shift_left_current_gray_frame)
   
    shift_left_prev_gray_frame = np.pad(prev_gray_frame,((0,0),(1,0)), mode='constant')[:, :-1] 
    reichardt_right = np.multiply(current_gray_frame,shift_left_prev_gray_frame) 

    reichardt_full_h = reichardt_left - reichardt_right



    # Up - Down direction
    shift_up_current_gray_frame = np.pad(current_gray_frame,((1,0),(0,0)), mode='constant')[:-1, :] 
    reichardt_left = np.multiply(prev_gray_frame,shift_up_current_gray_frame)

    shift_up_prev_gray_frame = np.pad(prev_gray_frame,((1,0),(0,0)), mode='constant')[:-1, :] 
    reichardt_right = np.multiply(current_gray_frame,shift_up_prev_gray_frame) 

    reichardt_full_v = reichardt_left - reichardt_right



    # Up right diagonal direction
    shift_diag1_current_gray_frame = np.pad(current_gray_frame,((1,0),(0,1)), mode='constant')[:-1, 1:] 
    reichardt_left = np.multiply(prev_gray_frame,shift_diag1_current_gray_frame)

    shift_diag1_prev_gray_frame = np.pad(prev_gray_frame,((1,0),(0,1)), mode='constant')[:-1, 1:] 
    reichardt_right = np.multiply(current_gray_frame,shift_diag1_prev_gray_frame) 

    reichardt_full_d1 = reichardt_left - reichardt_right



    # Down right direction
    shift_diag2_current_gray_frame = np.pad(current_gray_frame,((0,1),(0,1)), mode='constant')[1:, 1:] 
    reichardt_left = np.multiply(prev_gray_frame,shift_diag2_current_gray_frame)

    shift_diag2_prev_gray_frame = np.pad(prev_gray_frame,((0,1),(0,1)), mode='constant')[1:, 1:] 
    reichardt_right = np.multiply(current_gray_frame,shift_diag2_prev_gray_frame) 

    reichardt_full_d2 = reichardt_left - reichardt_right

    # Extract the maximum response absolute values from all directions
    reichardt_full = np.array([reichardt_full_h,reichardt_full_v,reichardt_full_d1,reichardt_full_d2])
    response_max = np.maximum.reduce(reichardt_full)
    response_min = np.minimum.reduce(reichardt_full)
    response = np.where(np.abs(response_max) > np.abs(response_min), response_max, response_min)


    # Daw points on the current frame with respect to direction
    detected_coordinates = np.argwhere(response > THRESHOLD)
    for coord in detected_coordinates:
        cv2.circle(frame, (coord[1],coord[0]), 1, (0,255,0), -1)

    detected_coordinates = np.argwhere(response < -THRESHOLD)
    for coord in detected_coordinates:
        cv2.circle(frame, (coord[1],coord[0]), 1, (0,0,255), -1)

# define a video capture object
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255

# Add noises to the frame to test the strength of the Reichardt's detector
if NOISE == True : prev_gray_frame = random_noise(prev_gray_frame, mode='gaussian',mean = 0,var = 0.0001).astype(np.float32)


cv2.namedWindow('frame')

while(True):
      
    # Capture the video frame by frame
    ret, frame = cap.read()
    
    current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
    
    # Add noises to the frame to test the strength of the Reichardt's detector
    if NOISE == True : current_gray_frame = random_noise(current_gray_frame, mode='gaussian',mean = 0,var = 0.0001).astype(np.float32)

    reichardt_detector(prev_gray_frame,current_gray_frame,frame)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    prev_gray_frame = np.copy(current_gray_frame)

    # The 'q' button is set as the quitting button
    wk = cv2.waitKey(10)
    if wk == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()