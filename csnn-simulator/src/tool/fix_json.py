# importing the module
from cgitb import grey
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

file_name = "KTH_H_ONOF_spikes_1_conv1"
file_path = "/home/melassal/Workspace/Results/KTH_H_ONOF_spikes_1/time/"

with open(file_path + "/" + file_name + ".json", "r") as s1:
    data = s1.read()

if data[0] != "[":
    new_data = "[" + data[:-1] + "]"

    with open(file_path + "/" + file_name + ".json", "w") as s1:
        s1.write(new_data)
