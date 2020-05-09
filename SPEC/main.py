# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:58:44 2019

@author: Justin
"""
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import csv


def plot_svc_decision_function1(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return X, Y, P


def plot_svc_decision_function2(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=0, alpha=0.5,
               linestyles='-')


    # plot support vectors
    #if plot_support:
    #    ax.scatter(model.support_vectors_[:, 0],
    #               model.support_vectors_[:, 1],
    #               s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return X, Y, P


with open('C:\\Users\\Justin\\Desktop\\SPEC Homework3\\data_feature.csv', 'r') as f:
    reader = csv.reader(f)
    Dataset = list(reader)
feature1 = [float(Dataset[i][0]) for i in range(len(Dataset))]
feature2 = [float(Dataset[i][1]) for i in range(len(Dataset))]
label = [int(Dataset[i][2]) for i in range(len(Dataset))]
Alldata = []
for i in range(len(feature1)):
    Alldata.append([feature1[i], feature2[i]])
Alldata = np.array(Alldata)

trainingdata = [Alldata[i] for i in range(len(Alldata))]
firstTrainingData = trainingdata[0:60]

firstTrainingData = np.array(firstTrainingData)
firstTrainingLabel = []
for i in range(60):
    if (i < 20):
        firstTrainingLabel.append(0)
    else:
        firstTrainingLabel.append(1)


secondTrainingData = trainingdata[20:60]
secondTrainingData = np.array(secondTrainingData)
secondTrainingLabel = []
for i in range(40):
    if (i < 20):
        secondTrainingLabel.append(1)
    else:
        secondTrainingLabel.append(2)




with open('C:\\Users\\Justin\\Desktop\\SPEC Homework3\\data_testing.csv', 'r') as f:
    reader = csv.reader(f)
    Dataset2 = list(reader)
x1 = [float(Dataset2[i][0]) for i in range(len(Dataset2))]
x2 = [float(Dataset2[i][1]) for i in range(len(Dataset2))]
answer = [int(Dataset2[i][2]) for i in range(len(Dataset2))]
testdata = []
for i in range(len(x1)):
    testdata.append([x1[i], x2[i]])
testdata = np.array(testdata)
testdata = testdata[0:30]
answer = answer[0:30]
firstTestData = testdata[0:30]
firstTestData = np.array(firstTestData)
firstTimeAnswer = []
for i in range(30):
    if (i < 10):
        firstTimeAnswer.append(0)
    else:
        firstTimeAnswer.append(1)

secondTestData = testdata[10:30]
secondTestData = np.array(secondTestData)
secondTimeAnswer = []
for i in range(20):
    if (i < 10):
        secondTimeAnswer.append(1)
    else:
        secondTimeAnswer.append(2)




supportVectorClassifier1 = SVC(kernel='linear', C=1E10)
supportVectorClassifier1.fit(firstTrainingData, firstTrainingLabel)
prediction1 = supportVectorClassifier1.predict(firstTestData)
print(confusion_matrix(firstTimeAnswer, prediction1))
print(classification_report(firstTimeAnswer, prediction1))
plt.scatter(firstTrainingData[0:20, 0], firstTrainingData[0:20, 1], c='b',label='Healthy')
plt.scatter(firstTrainingData[20:60, 0], firstTrainingData[20:60, 1], c='r',label='Unbalance')
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('SVM - Healthy & Unbalance')

plt.legend()
X_axis1, Y_axis1, P1 = plot_svc_decision_function1(supportVectorClassifier1)



supportVectorClassifier2 = SVC(kernel='linear', C=1E10)
supportVectorClassifier2.fit(secondTrainingData, secondTrainingLabel)
prediction2 = supportVectorClassifier2.predict(secondTestData)
print(confusion_matrix(secondTimeAnswer, prediction2))
print(classification_report(secondTimeAnswer, prediction2))
plt.figure()
plt.scatter(secondTrainingData[0:20, 0], secondTrainingData[0:20, 1], c='r',label='Unbalance 1')
plt.scatter(secondTrainingData[20:40, 0], secondTrainingData[20:40, 1], c='g',label='Unbalance 2')
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('SVM - Unbalance 1 & Unbalance 2')

X_axis2, Y_axis2, P2 = plot_svc_decision_function1(supportVectorClassifier2)
plt.legend()

plt.figure()
plt.scatter(Alldata[0:20, 0], Alldata[0:20, 1], c='b', label='Healthy')
plt.scatter(Alldata[20:40, 0], Alldata[20:40, 1], c='r', label='Unbalance 1')
plt.scatter(Alldata[40:60, 0], Alldata[40:60, 1], c='g', label='Unbalance 2')
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('Training SVM')
ax = plt.gca()
X_axis1, Y_axis1, P1 = plot_svc_decision_function1(supportVectorClassifier1)
X_axis2, Y_axis2, P2 = plot_svc_decision_function1(supportVectorClassifier2)

# ax.contour(X_axis1, Y_axis1, P1, colors='k',
#            levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
#
# ax.contour(X_axis2, Y_axis2, P2, colors='k',
#            levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
#
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)

plt.legend()
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('SVM Training Model')
plt.show()

plt.figure()
plt.scatter(x1, x2)
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('Testing Data')
plt.show()

plt.figure()
x1=np.array(x1)
x2=np.array(x2)
plt.scatter(x1[0:10], x2[0:10], c='b',label='Healthy')
plt.scatter(x1[10:20], x2[10:20], c='r',label='Unbalance 1')
plt.scatter(x1[20:30], x2[20:30], c='g',label='Unbalance 2')

plt.legend()
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('Testing Data')
plt.show()

plt.figure()
plt.scatter(x1[0:10], x2[0:10], c='b',label='Healthy')
plt.scatter(x1[10:20], x2[10:20], c='r',label='Unbalance 1')
plt.scatter(x1[20:30], x2[20:30], c='g',label='Unbalance 2')
plt.xlabel('20.8Hz')
plt.ylabel('84.6Hz')
plt.title('SVM Testing Result')
plot_svc_decision_function2(supportVectorClassifier1)
plot_svc_decision_function2(supportVectorClassifier2)
plt.legend()
plt.show()