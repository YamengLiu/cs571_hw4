import numpy as np
from sklearn.svm import SVC
import random
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve as roc, roc_auc_score as auc
import math
from cvxopt import matrix

def readData(filename):
    f=open(filename)
    samples=[]
    labels=[]
    for line in f:
        l=line.replace("\n","").split(",")
        instance=[]
        for i in range(len(l)-1):
          instance.append(float(l[i]))
        samples.append(instance)
        val=int(l[-1])
        labels.append(val)
    return (samples,labels)

def kernel(x,z):
  return np.dot(x,z.T)

x,y=readData("creditCard.csv")
trainx=[]
trainy=[]
testx=[]
testy=[]

for i in range(len(y)):
    if(random.randint(0,9)==9):
        testx.append(x[i])
        testy.append(y[i])
    else:
        trainx.append(x[i])
        trainy.append(y[i])

trainx=np.array(trainx)
trainy=np.array(trainy)
testx=np.array(testx)
testy=np.array(testy)

svc1=SVC(probability=True,kernel=kernel)
svc1.fit(trainx,trainy)
ans1=svc1.predict(testx)

ansProb1=svc1.predict_proba(testx)
prob1=ansProb1[:,1]
fpr1,tpr1,threshold1=roc(testy,prob1)
plt.plot(fpr1,tpr1,"r-",color="green")
auc1=auc(testy,prob1)

svc2=SVC(probability=True,kernel="rbf",C=2,gamma=0.5)
svc2.fit(trainx,trainy)
ans2=svc2.predict(testx)

ansProb2=svc2.predict_proba(testx)
prob2=ansProb2[:,1]
fpr2,tpr2,threshold2=roc(testy,prob2)
plt.plot(fpr2,tpr2,"r-",color="blue")
auc2=auc(testy,prob2)

svc3=SVC(probability=True,kernel="rbf",C=2,gamma=0.05)
svc3.fit(trainx,trainy)
ans3=svc3.predict(testx)

ansProb3=svc3.predict_proba(testx)
prob3=ansProb3[:,1]
fpr3,tpr3,threshold3=roc(testy,prob3)
plt.plot(fpr3,tpr3,"r-",color="black")
auc3=auc(testy,prob3)

plt.plot([0,1],[0,1],"r--",color="red")
plt.show()

print "Kernel1 AUC Score: "+str(auc1)
print "Kernel2 AUC Score: "+str(auc2)
print "Kernel2 AUC Score: "+str(auc3)