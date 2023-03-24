# importing the module
from cgitb import grey
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path

file_name = "KTH6_OnOff_1D_timestamps"
file_path = "/home/melassal/Workspace/Results/OnOff_FS_Compare/KTH6_OnOff_1D/test/"

with open(file_path + file_name + ".json", "r") as s1:
    data = s1.read()

if data[0] != "[":
    with open(file_path + file_name + "_part.json", "a") as s3:
        new_data = "[" + data[:15000000] + "]}]"
        s3.write(new_data)

    file_name = file_name + "_part"
    with open(file_path + file_name + ".json", "r") as s4:
        data = s4.read()
    

tensors = json.loads(data)


def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        #x = round(i, 4)
        hist[i] = hist.get(i, 0) + 1
    return hist


for i in range(0, len(tensors)):
    na = np.array(tensors[i]["data"])

    na.sort()

    na = na[na < 1]
    na = na[na != 0]

    counted = count_elements(na)

    # for j in range(len(na)):  # rounding the elements
    #     na[j] = round(na[j], 4)
    #     na[j] = na[j] #* 100

    # with open(file_path + file_name + "_histogram.txt", "a") as s4:
    #     s4.write("_______________sample_index_"+ str(i) + "_______________\n")
    #     s4.write(str(counted) + "\n\n")

    plt.hist(na, color="skyblue", bins=len(counted))
    # Save the histogram
    plt.savefig(file_path + 'hist_test_' + str(i) + '.png')
    # if(i < 191):
    #     plt.savefig(file_path + 'OnOff-hist-train' + str(i) + '.png')
    # else:
    #     plt.savefig(file_path + 'OnOff-hist-test' + str(i) + '.png')
