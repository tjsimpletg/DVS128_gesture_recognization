import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

SCALER = 50

# The dimentions that the input will be resized to later.
WIDTH = 40
HEIGHT = 40
COLORS = 3

COUNTER = 0
SAMPLES_PER_VID = 10

# Number of frames that form the output 3D cube grid.
HORIZIMAGECOUNT = 2
VERTIMAGECOUNT = 2

actionArray = ["boxing", "running", "handclapping","handwaving","jogging","walking"]
ContainingFolder = "train"

for Action in actionArray:
    # Get the path of the training and testing datasets that is structured as shown below in the comments at the end of this class.
    # Choose the input videos that will be transformed into 3D cubes, this directory must contain a similar structure.
    inputDirectory = "/Dataset/KTH/"+ContainingFolder+"/"+Action
    # Choose where to save these generated cubes.
    outputDirectory = "/Output_Dataset/KTH/" + ContainingFolder+"/"+Action
    # Colect the file names, because they are used as labels.
    fileNames = os.listdir(inputDirectory)

    # creating a numpy array for accepting colored images of 4*4*3 subframes (48 subframes total)
    composite_image = np.zeros(
        (VERTIMAGECOUNT*HEIGHT, HORIZIMAGECOUNT*WIDTH, COLORS), np.uint8)

    for fileName in fileNames:
        # Taking a video.
        vid = cv2.VideoCapture(inputDirectory + "/" + fileName)

        # Counters to monitor the placements of added subframes.
        depthCounter = 0
        widthCounter = 0
        heightCounter = 0

        # To count errored frames to be discrded.
        emptyFrameFounter = 0
        imageFileNumber = 0

        # Reading the first frame and saving it into "oldFrame", to initialise that frame.
        ret, oldFrame = vid.read()
        if ret == False:
            continue  # If video file is damaged, skip it

        oldFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

        permissionToSave = False

        while (True):
            # Counts frames to stop the video if they were too much.
            if COUNTER == SAMPLES_PER_VID:
                COUNTER = 0
                break
            # Capture the video frame by frame
            ret, frame = vid.read()
            if ret == True:
                # Transform the frame into grey-scale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Get the velocity distribution
                flow = cv2.calcOpticalFlowFarneback(
                    oldFrame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Update the oldFrame.
                oldFrame = frame
                flow_parts = cv2.split(flow)
                flowSum = np.maximum(np.absolute(
                    flow_parts[0]), np.absolute(flow_parts[1]))
                # The flowSum is a very small number, so it is multiplied by a scalar to increase it.
                flowSum = flowSum*SCALER
                # Taking flowSum as an integer flowSumint.
                flowSumint = flowSum.astype(int)
                # This array is used to chech if the registered amount of motion is segnificant.
                maskarray = ((flowSumint > (SCALER))*1).astype(np.uint8)
                # If the sum of movement is greater than 25, then the movement is considered segnificant and worth saving.
                if np.sum(maskarray) > 25:
                    permissionToSave = True

                # sobel_image = cv2.Sobel() to be tried later as an option

                canny_img = cv2.Canny(frame, 80, 300)
                canny_img = np.multiply(canny_img, maskarray)
                canny_img = cv2.resize(canny_img, (WIDTH, HEIGHT))
                (thresh, blackAndWhiteImage) = cv2.threshold(
                    canny_img, 50, 255, cv2.THRESH_BINARY)
                
                movementSum = cv2.sumElems(blackAndWhiteImage)
                # print(Action)
                # print(movementSum)
                if movementSum[0] < 500:
                    permissionToSave = False   
               
                # In case action was not segnificant the rest is skippes, moving on to the next video.
                if permissionToSave == False:
                    continue
                # If the frame is empty, the emptyFrameFounter is increased.
                if np.sum(blackAndWhiteImage) == 0:
                    emptyFrameFounter = emptyFrameFounter + 1

                # The frame saving process, In case of big enough detected movement.
                for x in range(0, WIDTH):
                    for y in range(0, HEIGHT):
                        composite_image[y + heightCounter*HEIGHT, x + widthCounter *
                                        WIDTH, depthCounter] = blackAndWhiteImage[y, x]
                # After looping the first depth the depth counter is increaseed to fill the next depth.
                depthCounter = depthCounter + 1

                # In case of only 1 depth.
                if depthCounter > COLORS - 1:
                    depthCounter = 0

                    widthCounter = widthCounter + 1

                    if widthCounter > HORIZIMAGECOUNT - 1:
                        widthCounter = 0

                        heightCounter = heightCounter + 1

                        if heightCounter > VERTIMAGECOUNT - 1:
                            heightCounter = 0
                            if not os.path.exists(outputDirectory):
                                os.makedirs(outputDirectory)
                            if emptyFrameFounter < 2 * HORIZIMAGECOUNT*VERTIMAGECOUNT:
                                cv2.imwrite(outputDirectory+"/EG_ "+fileName +
                                            str(imageFileNumber)+".png", composite_image)
                                COUNTER += 1
                                imageFileNumber = imageFileNumber + 1

                            emptyFrameFounter = 0
                            permissionToSave = False
            # In case no video was detected.
            else:
                imageFileNumber = 0
                break

            # breaking from the inner loop:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # closing video file
        vid.release()

        # breaking from the outer loop:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# The input dataset should look like this: KTH is an example, any video dataset would work
# KTH
#     ├── test
# │   ├── boxing
# │   │   └── person02_boxing_d1_uncomp.avi ..
# │   ├── handclapping
# │   │   └── person02_handclapping_d1_uncomp.avi .. 
# │   ├── handwaving
# │   │   └── person02_handwaving_d1_uncomp.avi .. 
# │   ├── jogging
# │   │   └── person02_jogging_d1_uncomp.avi .. 
# │   ├── running
# │   │   └── person02_running_d1_uncomp.avi ..
# │   └── walking
# │       └── person02_walking_d1_uncomp.avi ..
# ├── train
# │   ├── boxing
# │   │   └── person01_boxing_d1_uncomp.avi ..
# │   ├── handclapping
# │   │   └── person01_handclapping_d1_uncomp.avi ..
# │   ├── handwaving
# │   │   └── person01_handwaving_d1_uncomp.avi ..
# │   ├── jogging
# │   │   └── person01_jogging_d1_uncomp.avi ..
# │   ├── running
# │   │   └── person01_running_d1_uncomp.avi ..
# │   └── walking
#         └── person01_walking_d1_uncomp.avi ..
