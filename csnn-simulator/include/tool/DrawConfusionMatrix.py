from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Open the file with read only permit
f = open('/home/melassal/Workspace/CSNN/falez-csnn-simulator-build/Weights/Raw_2D_KTH')
# use readline() to read the first line
line = f.readline()
y_true = []
y_pred = []
# If the file is not empty keep reading one line till the file is empty.
while line:
    # Directory and label holders _ To get names and directories.
    removeExtra = line.replace("Predecred / Correct (", "")
    removeExtra = removeExtra.replace(")\n", "")
    if ")" in removeExtra:
        removeExtra = removeExtra.replace(")", "")
    splitedLineName = removeExtra.split("/")

    y_true.append(int(splitedLineName[1]))
    if int(splitedLineName[0]) == 0:
        y_pred.append(2)
    if int(splitedLineName[0]) == 1:
        y_pred.append(3)
    if int(splitedLineName[0]) == 2:
        y_pred.append(5)
    if int(splitedLineName[0]) == 3:
        y_pred.append(1)
    if int(splitedLineName[0]) == 4:
        y_pred.append(0)
    if int(splitedLineName[0]) == 5:
        y_pred.append(4)
    line = f.readline()
f.close()


matrix = confusion_matrix(y_true, y_pred)


df_cm = pd.DataFrame(matrix, range(6), range(6))

sn.set(font_scale=1.4)  # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='g')  # font size
sn.heatmap(matrix/np.sum(matrix), annot=True,  annot_kws={"size": 10},
            fmt='.2%', cmap='Blues')
plt.show()
