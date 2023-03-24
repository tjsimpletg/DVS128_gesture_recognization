from __future__ import print_function
import numpy as np
import os


input_dir = None
output_dir = None
log_dir = None

SCALER = 60
VARIATIONLIMIT = 255

#imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

#GLOBAL VARIABLES ZONE START
HORIZONTALSUBFRAMECOUNT = 4 # Will be multiplied by 4 to store right, left, up, and down movement sub frames
VERTICALSUBFRAMECOUNT = 20
# flag that flips frames if 1 and keeps the bframes as is if 0
FLIP = 1
#GLOBAL VARIABLES ZONE END

def main():

    ContainingFolder = "train"
    # for KTH
    actionArray = ["boxing", "running", "handclapping","handwaving","jogging","walking"]
    # for IXMAS
    # actionArray = ["check-watch", "cross-arms", "get-up",
    #                "kick", "pick-up", "point", "punch", "scratch-head", "sit-down", "turn-around", "walk", "wave"]
    # for Weizmann
    # actionArray = ["bend", "jack", "jump",
    #                            "pjump", "run", "side", "skip", "walk", "wave1", "wave2"]

    for Action in actionArray:
        # Get the path of the training and testing datasets that is structured as shown below in the comments at the end of this class.
        # Choose the input videos that will be transformed into 3D cubes.
        inputDirectory = "/Dataset/KTH/"+ContainingFolder+"/"+Action
        # Choose where to save these generated cubes, then input them into the SNN.
        outputDirectory = "/Output_Dataset/KTH/" + ContainingFolder + "/"+Action
        for dirName, subdirList, fileList in os.walk(inputDirectory):                            
            for fileName in fileList: 
            
                if os.path.isdir(inputDirectory + '/' + fileName):
                    continue
                if "tag" in fileName:
                    continue    
                # Create a VideoCapture object and read from input file
                cap = cv2.VideoCapture(inputDirectory + '/' + fileName) 
                # Check if video is opened successfully
                if (cap.isOpened()== False): 
                    print("Error opening video stream or file")
                    continue    
                subFrameDimensions = (0,0)  
                frameCountPerVideo = 0  
                ret, frame1 = cap.read()

                if FLIP == 1:
                    frame1 = cv2.flip(frame1, 1)

                if ret == True:
                     prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                     subFrameDimensions = prvs.shape    
                if subFrameDimensions == (0,0):
                    continue    
                frameQueueCounter1 = 0
                frameQueueCounter2 = 0
                frameQueueCounter3 = 0  
                frameQueueReach1 = 0
                frameQueueReach2 = 0
                frameQueueReach3 = 0    
                frameQueue1 = np.zeros((300,subFrameDimensions[0],subFrameDimensions[1]*4),dtype = np.uint8)
                frameQueue2 = np.zeros((300,subFrameDimensions[0],subFrameDimensions[1]*4),dtype = np.uint8)
                frameQueue3 = np.zeros((300,subFrameDimensions[0],subFrameDimensions[1]*4),dtype = np.uint8)    
                roleDistributer = 0 
                while(cap.isOpened()):  
                  # Capture frame-by-frame
                    ret, frame1 = cap.read()
                    if ret == True:
                        #Convert the frame into grayscale
                        next = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                        # Display the recent frame
                        # cv2.imshow("outputwindow",next) 
                        frameCountPerVideo += 1 
                        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)        
                        prvs = next 
                        upframefloat = (np.abs(flow[:,:,1]) - flow[:,:,1])/2
                        downframefloat = (np.abs(flow[:,:,1]) + flow[:,:,1])/2
                        leftframefloat = (np.abs(flow[:,:,0]) - flow[:,:,0])/2
                        rightframefloat = (np.abs(flow[:,:,0]) + flow[:,:,0])/2 
                        upframefloatscaled = upframefloat*SCALER
                        downframefloatscaled = downframefloat*SCALER
                        leftframefloatscaled = leftframefloat*SCALER
                        rightframefloatscaled = rightframefloat*SCALER  
                        upframe = upframefloatscaled.astype(np.uint8)
                        downframe = downframefloatscaled.astype(np.uint8)
                        leftframe = leftframefloatscaled.astype(np.uint8)
                        rightframe = rightframefloatscaled.astype(np.uint8) 
                        #clippling values greater than 255  
                        upframe[upframe > VARIATIONLIMIT] = VARIATIONLIMIT
                        downframe[downframe > VARIATIONLIMIT] = VARIATIONLIMIT
                        leftframe[leftframe > VARIATIONLIMIT] = VARIATIONLIMIT
                        rightframe[rightframe > VARIATIONLIMIT] = VARIATIONLIMIT    
                        
                        try:
                            subFrameAssembly = np.concatenate((upframe,downframe,leftframe,rightframe),1)   
                            # cv2.imshow("subframeassembly",subFrameAssembly) 
                            if (roleDistributer % 3) == 0:
                                frameQueue1[frameQueueReach1] = np.copy(subFrameAssembly)
                                frameQueueReach1 += 1   
                            elif (roleDistributer % 3) == 1:
                                frameQueue2[frameQueueReach2] = np.copy(subFrameAssembly)
                                frameQueueReach2 += 1   
                            elif (roleDistributer % 3) == 2:
                                frameQueue3[frameQueueReach3] = np.copy(subFrameAssembly)
                                frameQueueReach3 += 1   
                        except:
                            print("oops") 
                        roleDistributer += 1
                        ################################################################### 
                    else:
                        break
                    
                # When everything is done, release the video capture object
                cap.release()   
                print("frame dimensions: ",subFrameDimensions,"   ","frame count = ",frameCountPerVideo,"   ","file name: ",fileName)   
                minFrameQueueReach = 0  
                #for synchronizing output frames    
                if frameQueueReach1 <= frameQueueReach2 and frameQueueReach1 <= frameQueueReach3 :
                    minFrameQueueReach = frameQueueReach1
                if frameQueueReach2 <= frameQueueReach1 and frameQueueReach2 <= frameQueueReach3 :
                    minFrameQueueReach = frameQueueReach2
                if frameQueueReach3 <= frameQueueReach1 and frameQueueReach3 <= frameQueueReach2 :
                    minFrameQueueReach = frameQueueReach3   
                ################################################################### motionGrid1 
                motionGrid1 = ""    
                for y in range(0, VERTICALSUBFRAMECOUNT):   
                    horizontalMotionSlice1 = "" 
                    for x in range(0, HORIZONTALSUBFRAMECOUNT):
                        if x == 0:
                            horizontalMotionSlice1 = np.copy(frameQueue1[frameQueueCounter1]) 
                        else:
                            horizontalMotionSlice1 = np.concatenate((horizontalMotionSlice1,frameQueue1[frameQueueCounter1]),1) 
                        frameQueueCounter1 += 1
                        #frameQueueCounter1 = frameQueueCounter1 % frameQueueReach1
                        frameQueueCounter1 = frameQueueCounter1 % minFrameQueueReach    
                    if y == 0:
                        motionGrid1 = np.copy(horizontalMotionSlice1)
                    else:
                        motionGrid1 = np.concatenate((motionGrid1,horizontalMotionSlice1),0)    
                # cv2.imshow("motiongrid1",motionGrid1)   
                ################################################################### motionGrid1 
                ################################################################### motionGrid2 
                motionGrid2 = ""    
                for y in range(0, VERTICALSUBFRAMECOUNT):   
                    horizontalMotionSlice2 = "" 
                    for x in range(0, HORIZONTALSUBFRAMECOUNT):
                        if x == 0:
                            horizontalMotionSlice2 = np.copy(frameQueue2[frameQueueCounter2]) 
                        else:
                            horizontalMotionSlice2 = np.concatenate((horizontalMotionSlice2,frameQueue2[frameQueueCounter2]),1) 
                        frameQueueCounter2 += 1
                        #frameQueueCounter2 = frameQueueCounter2 % frameQueueReach2
                        frameQueueCounter2 = frameQueueCounter2 % minFrameQueueReach    
                    if y == 0:
                        motionGrid2 = np.copy(horizontalMotionSlice2)
                    else:
                        motionGrid2 = np.concatenate((motionGrid2,horizontalMotionSlice2),0)     
                ################################################################### motionGrid2 
                ################################################################### motionGrid3 
                motionGrid3 = ""    
                for y in range(0, VERTICALSUBFRAMECOUNT):   
                    horizontalMotionSlice3 = "" 
                    for x in range(0, HORIZONTALSUBFRAMECOUNT):
                        if x == 0:
                            horizontalMotionSlice3 = np.copy(frameQueue3[frameQueueCounter3]) 
                        else:
                            horizontalMotionSlice3 = np.concatenate((horizontalMotionSlice3,frameQueue3[frameQueueCounter3]),1) 
                        frameQueueCounter3 += 1
                        #frameQueueCounter3 = frameQueueCounter3 % frameQueueReach3
                        frameQueueCounter3 = frameQueueCounter3 % minFrameQueueReach    
                    if y == 0:
                        motionGrid3 = np.copy(horizontalMotionSlice3)
                    else:
                        motionGrid3 = np.concatenate((motionGrid3,horizontalMotionSlice3),0)    
                # cv2.imshow("motiongrid3",motionGrid3)   
                ################################################################### motionGrid3 
                if not os.path.exists(outputDirectory):
                    os.makedirs(outputDirectory)
                ################################################################### saving to files 
                motionGrid1 = cv2.resize(motionGrid1, (250, 250), None,)
                motionGrid2 = cv2.resize(motionGrid2, (250, 250), None,)
                motionGrid3 = cv2.resize(motionGrid3, (250, 250), None,)
                if FLIP == 0:
                    cv2.imwrite(outputDirectory + "/" + fileName +"o-"+ str(1)+".png",motionGrid1)
                    cv2.imwrite(outputDirectory + "/" + fileName +"o-"+ str(2)+".png",motionGrid2)
                    cv2.imwrite(outputDirectory + "/" + fileName +"o-"+ str(3)+".png",motionGrid3)   
                else:
                    cv2.imwrite(outputDirectory + "/" + fileName +"-"+ str(1)+".png",motionGrid1)
                    cv2.imwrite(outputDirectory + "/" + fileName +"-"+ str(2)+".png",motionGrid2)
                    cv2.imwrite(outputDirectory + "/" + fileName +"-"+ str(3)+".png",motionGrid3)  
                ################################################################### saving to files 
           
        # Closes all the frames 
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

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
