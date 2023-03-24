# importing the module
import glob
from cgitb import grey
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
from pathlib import Path

file_names = ["TS_MG3_1"]

for file_name in file_names:
    with open("" + file_name + ".json", "r") as s1:
        data = s1.read()

    file_path = "/home/melassal/Workspace/Code/Draw_Tensor/drawn_tensors/" + file_name + "/"

    tensors = json.loads(data)

    i = 0
    for tensor in tensors:
        i += 1
        na = np.array(tensor["data"])
        draw_tensor = na.reshape(
            (tensor["dim_3"], tensor["dim_0"], tensor["dim_1"], tensor["dim_2"]))

        for filter_number in range(tensor["dim_3"]):  # the 64
            # filter_number = 0
            image_array_0 = draw_tensor[filter_number, :, :, 0] * 255
            image_array_1 = draw_tensor[filter_number, :, :, 1] * 255

            image_array = (image_array_0 + image_array_1)
            # image_array = image_array / np.sum(image_array)

            # Save
            final_name = f'tensor_l{tensor["label"]}_f{filter_number}_i{i}.png'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            cv2.imwrite(file_path + final_name, image_array)