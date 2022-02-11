import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sys
import os
args = sys.argv
folder = args[0]
folder = folder.split('\\')
folder = folder[0:-1]
folder = "/".join(folder)
file = 'DATA.csv'

start = time.time()
df = pd.read_csv (folder + '/' + file)
variables = list(df.columns)
features = variables[385:385+50]
Class = variables[-1]
print("Running Model....please wait")
train, test = train_test_split(df, test_size=0.3, shuffle=True)
ABS = []

for i in range(1,9):
    ABS.append(np.absolute((train.loc[train['ECU'] == i, features]-3.48)))

GR = []
A = []
e = math.e
epoch = 5000
for i in range(8):
    a = round(float(ABS[i].loc[:, [features[0]]].mean()), 2)
    A.append(a)
    index = list(ABS[i].index)
    x = np.array(list(range(len(features))))
    k = -0.1
    for t in range(0,epoch):
        y = a*e**(k*x)
        err = np.zeros(len(features))
        for row in index:
            error = ABS[i].loc[row,features]-y
            err = err+error
        e_average = np.array(err)/len(index)
        e_total = e_average.sum()
        if e_total > 0:
            k = k + .0001 
        else:
            k = k - .0001
    GR.append(k)
    
for i in range(8):
    y_hat = (A[i]*math.e**(GR[i]*x))
    plt.plot(y_hat)
plt.legend(['ECU1','ECU2','ECU3','ECU4','ECU5','ECU6','ECU7','ECU8'])
plt.show()

y = np.array(test.loc[:, Class])
index = list(test.index)
pred1 = []
for i in index:
    check = []
    for k in range(8):
        y_hat = A[k]*math.e**(GR[k]*x)
        h = np.absolute(test.loc[i, features]-3.5)
        MSE = ((h-y_hat)**2).sum()/len(features)
        check.append(MSE)
    pred1.append(check.index(min(check))+1)
pred1 = np.array(pred1)
guess = (y==pred1)*1
accuracy = guess.sum()/len(y)
print(accuracy)

end = time.time()
print("Time to run: ", end-start)

results = pd.DataFrame(np.transpose([np.array(pred1),np.array(y),np.array(guess)]), columns=['pred', 'actual', 'output'])
print(results)
print("Model Complete")
time.sleep(180)
