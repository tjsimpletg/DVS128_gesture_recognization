# importing the module
from cgitb import grey
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path

file_path = "/home/melassal/Workspace/CSNN/csnn-simulator-build/Analyse/Timestamps/KTH_TT_1/"
file_name = "KTH_TT_1_conv1"
file_name2 = "KTH_TT_1_conv2"
file_name3 = "KTH_TT_1_conv3"

with open(file_path + file_name + ".json", "r") as s1:
    data = s1.read()
with open(file_path + file_name2 + ".json", "r") as s2:
    data2 = s2.read()
with open(file_path + file_name3 + ".json", "r") as s3:
    data3 = s3.read()

tensors = json.loads(data)
tensors2 = json.loads(data2)
tensors3 = json.loads(data3)


def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        # x = round(i, 4)
        hist[i] = hist.get(i, 0) + 1
    return hist

for i in range(0, len(tensors)):
    na = np.array(tensors[i]["data"])
    na2 = np.array(tensors2[i]["data"])
    na3 = np.array(tensors3[i]["data"])

    na.sort()
    na2.sort()
    na3.sort()

    na = na[na != 3.402823466e38]
    na2 = na2[na2 != 3.402823466e38]
    na3 = na3[na3 != 3.402823466e38]

    counted = count_elements(na)
    counted2 = count_elements(na2)
    counted3 = count_elements(na3)

    plt.hist(na, color="skyblue", bins=len(counted))
    plt.hist(na2, color="pink", bins=len(counted2))
    plt.hist(na3, color="red", bins=len(counted3))
    # Save the histogram
    plt.savefig(file_path + 'hist'+ str(i) +'.png')
