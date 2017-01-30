from cvxopt import matrix, solvers, mul
from cvxopt import matrix, solvers, mul
import random
def readData(file):
  f=open(file)
  data=[]
  label=[]
  for l in f:
    s=l.replace("\n","").split(",")
    data.append([float(s[0]),float(s[1]),float(s[2]),float(s[3])])
    if(s[4]=="Iris-setosa"):
      label.append(1.0)
    else:
      label.append(-1.0)
  f.close()
  return (data,label)

def train(x,y):
	x_=iter(x)
	y_=iter(y)

	X=matrix(x)
	transX=X.trans()
	X_=transX*X
	n=len(y)
	g=[]
	q=[]
	for i in range(n):
		for j in range(n):
			if(i!=j):
				g.append(0.0)
			else:
				g.append(-1.0)
			q.append(y[i]*y[j]*X_[i*n+j])
	Q=matrix(q,(n,n))
	G=matrix(g,(n,n))
	b=matrix(0.0)
	A=matrix(y).trans()
	p=matrix([-1.0]*n)
	h=matrix([0.0]*n)

	sol=solvers.qp(Q,p,G,h,A,b)
	ans=sol['x']

	lamdaStar=matrix([0.0]*len(x[0]))
	for i in range(n):
		lamdaStar=lamdaStar + ans[i]*y[i]*matrix(x[i])

	lamdaZero=0;
	for i in range(n):
		if(ans[i]<0.0001 and ans[i]>-0.0001):
			lamdaZero=1-lamdaStar.trans()*matrix(x[i])*y[i]

	return (lamdaStar,lamdaZero)

def predict(x,star,zero):
	y=[]
	for elem in x:
		result=star.trans()*matrix(elem)-zero
		if(result[0]>=0):
			y.append(1)
		else:
			y.append(-1)
	return y

(x,y)=readData("iris.data.txt")
testx=[]
testy=[]
trainx=[]
trainy=[]
for i in range(len(y)):
    if(random.randint(0,9)>=7):
        testx.append(x[i])
        testy.append(y[i])
    else:
        trainx.append(x[i])
        trainy.append(y[i])
(lamdaStar,lamdaZero)=train(trainx,trainy)
res=predict(testx,lamdaStar,lamdaZero)
rightNum=0
for i in range(len(testy)):
	if(testy[i]==res[i]):
		rightNum+=1
print "accuracy is+ "+str(float(rightNum)/len(testy))
