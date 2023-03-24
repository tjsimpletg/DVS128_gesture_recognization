import os
import PIL
from PIL import Image

MAGIC_NUMBER_FOR_TRAINING_SET_LABEL_FILE = 0x00000801
MAGIC_NUMBER_FOR_TRAINING_SET_IMAGE_FILE = 0x00000803

MAGIC_NUMBER_FOR_TESTSET_LABEL_FILE = 0x00000801
MAGIC_NUMBER_FOR_TESTSET_IMAGE_FILE = 0x00000803

IMAGEWIDTH = 160
IMAGEHEIGHT = 120

size = IMAGEWIDTH, IMAGEHEIGHT

SaveFileNumber = 0
# This variable formats the data into a 1 channel array in case of tuple
isTuple = False
# KTH or Weizmann, this changes the labels.
# datasetName = "Weizmann"
datasetName = "KTH"

TrainSampleMax = 1000000
TestSampleMax = 1000000

TrainSampleCount = 0
TestSampleCount = 0

train_input_directory = "/home/melassal/Workspace/Datasets/Original/KTH/Frames/5_frames/trainMix/" 
#Folder for input test images
test_input_directory = "/home/melassal/Workspace/Datasets/Original/KTH/Frames/5_frames/validMix/" 
#Folder to store output files
output_files_directory = "/home/melassal/Workspace/Datasets/Original/KTH/Frames/5_frames/" + datasetName + "_V_MN/"

# train_images_soret_list = os.listdir(train_input_directory)
train_images_soret_list = sorted(os.listdir(train_input_directory))

train_images = []
train_labels = []
train_images_IntArray = []

# And the byte and then move the last byte to the right
train_images.append((MAGIC_NUMBER_FOR_TRAINING_SET_IMAGE_FILE & 0xFF000000)>>24)
train_images.append((MAGIC_NUMBER_FOR_TRAINING_SET_IMAGE_FILE & 0xFF0000)>>16)
train_images.append((MAGIC_NUMBER_FOR_TRAINING_SET_IMAGE_FILE & 0xFF00)>>8)
train_images.append(MAGIC_NUMBER_FOR_TRAINING_SET_IMAGE_FILE & 0xFF)

train_images.append((len(train_images_soret_list) & 0xFF000000)>>24)
train_images.append((len(train_images_soret_list) & 0xFF0000)>>16)
train_images.append((len(train_images_soret_list) & 0xFF00)>>8)
train_images.append(len(train_images_soret_list) & 0xFF)

train_images.append((IMAGEHEIGHT & 0xFF000000)>>24)
train_images.append((IMAGEHEIGHT & 0xFF0000)>>16)
train_images.append((IMAGEHEIGHT & 0xFF00)>>8)
train_images.append(IMAGEHEIGHT & 0xFF)

train_images.append((IMAGEWIDTH & 0xFF000000)>>24)
train_images.append((IMAGEWIDTH & 0xFF0000)>>16)
train_images.append((IMAGEWIDTH & 0xFF00)>>8)
train_images.append(IMAGEWIDTH & 0xFF)

train_labels.append((MAGIC_NUMBER_FOR_TRAINING_SET_LABEL_FILE& 0xFF000000)>>24)
train_labels.append((MAGIC_NUMBER_FOR_TRAINING_SET_LABEL_FILE& 0xFF0000)>>16)
train_labels.append((MAGIC_NUMBER_FOR_TRAINING_SET_LABEL_FILE& 0xFF00)>>8)
train_labels.append(MAGIC_NUMBER_FOR_TRAINING_SET_LABEL_FILE & 0xFF)

train_labels.append((len(train_images_soret_list) & 0xFF000000)>>24)
train_labels.append((len(train_images_soret_list) & 0xFF0000)>>16)
train_labels.append((len(train_images_soret_list) & 0xFF00)>>8)
train_labels.append(len(train_images_soret_list) & 0xFF)

for train_image_name in train_images_soret_list:
    if TrainSampleMax <= TrainSampleCount:
        break
    # Extract the files one by one
    image =  Image.open(train_input_directory + "/" + train_image_name)
    
    image.thumbnail(size, Image.ANTIALIAS)

    # Extract each image pixel
    pix_val = list(image.getdata())
    # Add the bytes of the images to the array
    train_images.extend(pix_val)
    
    classnumber = 100 # initialise, 0 is already used for boxing so 100 instead.
    if datasetName == "KTH":
        if "boxing" in train_image_name:
            classnumber = 0
        if "handclapping" in train_image_name:
            classnumber = 1
        if "handwaving" in train_image_name:
            classnumber = 2
        if "jogging" in train_image_name:
            classnumber = 3
        if "running" in train_image_name:
            classnumber = 4
        if "walking" in train_image_name:
            classnumber = 5
    
    train_labels.append(classnumber)
    # TrainSampleCount += 1

print("train images size: " + str(len(train_images)))
print("train labels size: " + str(len(train_labels)))
train_images_bytes_array = bytes(train_images)
train_labels_bytes_array = bytes(train_labels)


if not os.path.exists(output_files_directory):
    os.makedirs(output_files_directory)
#print(binascii.hexlify(train_images_bytes_array))
output_train_image_file = open(output_files_directory + "/" + "train-images.idx3-ubyte", "wb")
output_train_image_file.write(train_images_bytes_array)
output_train_image_file.close()

while output_train_image_file.closed == False:
    pass

output_train_label_file = open(output_files_directory + "/" + "train-labels.idx1-ubyte", "wb")
output_train_label_file.write(train_labels_bytes_array)
output_train_label_file.close() 

while output_train_label_file.closed == False:
    pass

##########################################################################################
##########################################################################################
##########################################################################################

# test_images_soret_list = os.listdir(test_input_directory)
test_images_soret_list = sorted(os.listdir(test_input_directory))

test_images = []
test_labels = []
test_images_IntArray = []

test_images.append((MAGIC_NUMBER_FOR_TESTSET_IMAGE_FILE & 0xFF000000)>>24)
test_images.append((MAGIC_NUMBER_FOR_TESTSET_IMAGE_FILE & 0xFF0000)>>16)
test_images.append((MAGIC_NUMBER_FOR_TESTSET_IMAGE_FILE & 0xFF00)>>8)
test_images.append(MAGIC_NUMBER_FOR_TESTSET_IMAGE_FILE & 0xFF)

test_images.append((len(test_images_soret_list) & 0xFF000000)>>24)
test_images.append((len(test_images_soret_list) & 0xFF0000)>>16)
test_images.append((len(test_images_soret_list) & 0xFF00)>>8)
test_images.append(len(test_images_soret_list) & 0xFF)

test_images.append((IMAGEHEIGHT & 0xFF000000)>>24)
test_images.append((IMAGEHEIGHT & 0xFF0000)>>16)
test_images.append((IMAGEHEIGHT & 0xFF00)>>8)
test_images.append(IMAGEHEIGHT & 0xFF)

test_images.append((IMAGEWIDTH & 0xFF000000)>>24)
test_images.append((IMAGEWIDTH & 0xFF0000)>>16)
test_images.append((IMAGEWIDTH & 0xFF00)>>8)
test_images.append(IMAGEWIDTH & 0xFF)

test_labels.append((MAGIC_NUMBER_FOR_TESTSET_LABEL_FILE& 0xFF000000)>>24)
test_labels.append((MAGIC_NUMBER_FOR_TESTSET_LABEL_FILE& 0xFF0000)>>16)
test_labels.append((MAGIC_NUMBER_FOR_TESTSET_LABEL_FILE& 0xFF00)>>8)
test_labels.append(MAGIC_NUMBER_FOR_TESTSET_LABEL_FILE & 0xFF)
#####
test_labels.append((len(test_images_soret_list) & 0xFF000000)>>24)
test_labels.append((len(test_images_soret_list) & 0xFF0000)>>16)
test_labels.append((len(test_images_soret_list) & 0xFF00)>>8)
test_labels.append(len(test_images_soret_list) & 0xFF)

for test_image_name in test_images_soret_list:
    
    if TestSampleMax <= TestSampleCount:
        break
   
    image = PIL.Image.open(test_input_directory + "/" + test_image_name)

    image.thumbnail(size, Image.ANTIALIAS)

    # print(test_image_name)
    pix_val = list(image.getdata())
    test_images.extend(pix_val)
    
    classnumber = 100

    if datasetName == "KTH":
        if "box" in test_image_name:
            classnumber = 0
        if "cla" in test_image_name:
            classnumber = 1
        if "wav" in test_image_name:
            classnumber = 2
        if "jog" in test_image_name:
            classnumber = 3
        if "run" in test_image_name:
            classnumber = 4
        if "wal" in test_image_name:
            classnumber = 5

    test_labels.append(classnumber)
    # TestSampleCount += 1

print("test images size: " + str(len(test_images)))
print("test labels size: " + str(len(test_labels)))
test_images_byte_array = bytes(test_images)
test_labels_byte_array = bytes(test_labels)

####################
output_test_image_file = open(output_files_directory + "/" + "t10k-images.idx3-ubyte", "wb")
output_test_image_file.write(test_images_byte_array)
output_test_image_file.close()

while output_test_image_file.closed == False:
    pass

output_test_label_file = open(output_files_directory + "/" + "t10k-labels.idx1-ubyte", "wb")
output_test_label_file.write(test_labels_byte_array)
output_test_label_file.close() 

while output_test_label_file.closed == False:
    pass