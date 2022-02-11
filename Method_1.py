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
features = variables[22:-1]
Class = variables[-1]

train, test = train_test_split(df, test_size=0.2, shuffle=True)
MEAN = [] 
STD = [] 

for i in range(1,9):
    MEAN.append((train.loc[train['ECU'] == i, features].mean()))
    STD.append((train.loc[train['ECU'] == i, features].std()))
test1 = test.loc[:,features]

y = np.array(test.loc[:, Class])
index = list(test.index)
pred1 = []
pred2 = []
pred3 = []
for i in index:
    check1 = []
    check2 = []
    check3 = []
    
    for k in range(len(MEAN)):
        check1.append((np.absolute((test1.loc[i]-MEAN[k]))>STD[k]*3.2).astype(int).sum())
        check2.append(((np.absolute((test1.loc[i]-MEAN[k])))).sum())
        check3.append((np.absolute((test1.loc[i]-MEAN[k]))<STD[k]*2).astype(int).sum())
        

    pred1.append(check1.index(min(check1))+1)
    pred2.append(check2.index(min(check2))+1)
    pred3.append(check3.index(max(check3))+1)
    
    
pred1 = np.array(pred1)
pred2 = np.array(pred2)
pred3 = np.array(pred3)
guess1 = (y==pred1)*1
guess2 = (y==pred2)*1
guess3 = (y==pred3)*1
accuracy = [guess1.sum()/len(y),guess2.sum()/len(y),guess3.sum()/len(y)]
print('Accuracy [STD>3.2,Mean Diff,STD<2] = ', accuracy)
end = time.time()
print('Time to run: ', end-start)

results = pd.DataFrame(np.transpose([np.array(pred1),np.array(y),np.array(guess1)]), columns=['pred', 'actual', 'output'])
print(results)
print("Model Complete")
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('MEAN and STD Features')

ax1.plot(features, np.array(MEAN).T.tolist())
ax1.set_xlabel('Features')
ax1.set_ylabel('Mean Values')

ax2.plot(features, np.array(STD).T.tolist())
ax2.set_xlabel('Features')
ax2.set_ylabel('Standard Deviation')
ax2.legend(['ECU1','ECU2','ECU3','ECU4','ECU5','ECU6','ECU7','ECU8'])
plt.show()
print("Window will close in 3 minutes")
time.sleep(180)
