import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def readDataFile():
    trainingFile = 'irisdata.csv'
    print("training file: ", trainingFile)
    raw_train = open(trainingFile, 'rt')

    data_train = np.loadtxt(raw_train, usecols=(0, 1, 2, 3, 4), skiprows = 1, delimiter=",")

    return data_train

def activation(X, W):
	prediction = np.dot(X, W)
	return 1 / (1+(np.exp((-1)*prediction)))

def calcCost(X,W,Y):
	activated = activation(X, W)
	ifY1 = Y*np.log(activated)
	ifY0 = (1-Y) * np.log(1 - activated)
	return (-1/len(x)) * np.sum(ifY1 + ifY0)

def calcGradient(X,W,Y):
	pred = activation(X, W) - Y
	return X.T.dot(pred) / len(X)

data = readDataFile()

x = data[:, 0:4]

average = np.mean(x, axis=0)
rang = np.ptp(x, axis=0)

x = (x - average)/rang

bias  = np.ones((len(x),1))
x = np.concatenate((bias,x), axis =1)

y = data[:,4:] 
ohe = OneHotEncoder(categories = 'auto')
y = ohe.fit_transform(y).toarray()


w = np.array(np.zeros((x.shape[1],1)))

LR = 5

'''
costArray = []
'''


flowerWeights = np.array([])

zzzzz = y[:,0]
numFlowers = y.shape[1]

for k in range(numFlowers):
    maxIter = 21753
    i = 0

    minDiff = 0.0001
    vectDiff = 1

    w = np.array(np.zeros((x.shape[1],1)))

    yToUse = y[:,k].reshape(len(y), 1)

    while i < maxIter and vectDiff > minDiff:

        gradients = calcGradient(x, w, yToUse)
        
        w = w - (LR*gradients)
        
        
        vectDiff = np.linalg.norm(gradients)
        
        i+=1
    if k == 0:
        flowerWeights = w
    else:
        flowerWeights = np.concatenate((flowerWeights, w), axis=1)

prediction = activation(x, flowerWeights)
prediction = prediction.argmax(axis=1)
y = y.argmax(axis=1)

def comparePredActual(pred, actual):
    return np.count_nonzero(pred - actual)

incorrect = comparePredActual(prediction, y)

print("Number Incorrect:", incorrect)
print("Number Correct:", len(y) - incorrect)
print("Accuracy: %.3f" % ((len(y) - incorrect)/len(y)))

'''
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(maxIter), costArray, color='blue')
ax2.set(title = 'Cost vs. Iterations', xlabel = 'iterations', ylabel = 'cost')

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.set_title("Data plot")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
        
for d, sample in enumerate(x):
    # Plot the negative samples
    if y[d]==0:
        plt.plot(sample[1], sample[2], 'ro')
    # Plot the positive samples
    else:
        plt.plot(sample[1], sample[2], 'bx')
'''
