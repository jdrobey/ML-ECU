import tkinter as tk
import pathlib
import sys
import pandas as pd
import numpy as np

from tkinter import filedialog,messagebox
from pathlib import os
root = tk.Tk()
root.withdraw()

file = filedialog.askopenfiles(initialdir = 'C:/',title='Select CSV files to merge',filetypes=[('All Files', '.*')])
result = []
n = 5000
for i in range(len(file)):
    myFile = np.genfromtxt(file[i].name, delimiter=',')
    if myFile[0]>= 3:
        for k in range(len(myFile)):
            if myFile[k]<3 and myFile[k+1]<3 and myFile[k+2]<3 :
                index = k
                myFile = myFile[index::]
                m = len(myFile)
                break
    else:
        for k in range(len(myFile)):
            if myFile[k]>=3 and myFile[k+1]>=3 and myFile[k+2]>=3:
                index = k
                break
        for j in range(index,len(myFile)):
            if myFile[j]<3 and myFile[j+1]<3 and myFile[j+2]<3:
                index = j
                myFile = myFile[index::]
                m = len(myFile)
                break
    if m < n:
        n = m
    result.append(myFile)

for i in range(len(result)):
    result[i] = result[i][0:n]
data = []
for i in range(len(result)):
    data.append(np.transpose(pd.DataFrame((result[i])))) 
    
data = pd.concat(data)
dir_ = os.path.dirname(file[0].name)
fid = dir_ + "_out.csv"
data.to_csv(fid, index=False)