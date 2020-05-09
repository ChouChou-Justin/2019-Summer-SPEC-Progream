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
from sklearn.preprocessing import StandardScaler

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

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return X, Y, P

with open('C:\\Users\\Justin\\Desktop\\SPEC Final Project\\trainingData_2.csv', 'r') as f:
    reader = csv.reader(f)
    Dataset = list(reader)
EntropyTrainData = [Dataset[i][0] for i in range(len(Dataset))]
EntropyTrainData.remove('Entropy')
VarianceTrainData = [Dataset[i][1] for i in range(len(Dataset))]
VarianceTrainData.remove('Variance')

with open('C:\\Users\\Justin\\Desktop\\SPEC Final Project\\validationData_2.csv', 'r') as f:
    reader = csv.reader(f)
    Dataset2 = list(reader)
EntropyValidationData = [Dataset2[i][0] for i in range(len(Dataset2))]
EntropyValidationData.remove('Entropy')
VarianceValidationData = [Dataset2[i][1] for i in range(len(Dataset2))]
VarianceValidationData.remove('Variance')

Alldata = []
for i in range(len(EntropyTrainData)):
    Alldata.append([EntropyTrainData[i], VarianceTrainData[i]])
Alldata = np.array(Alldata, dtype='double')
trainingdata = [Alldata[i] for i in range(len(Alldata))]

firstTrainingData = trainingdata[0:1536]
firstTrainingData = np.array(firstTrainingData)
firstTrainingLabel = []
for i in range(1536):
    if (i < 768):
        firstTrainingLabel.append(1)
    else:
        firstTrainingLabel.append(2)

secondTrainingData = trainingdata[0:768]
secondTrainingData = np.array(secondTrainingData)
secondTrainingLabel = []
for i in range(768):
    if (i < 192):
        secondTrainingLabel.append(1)
    else:
        secondTrainingLabel.append(2)

thirdTrainingData = trainingdata[192:768]
thirdTrainingData = np.array(thirdTrainingData)
thirdTrainingLabel = []
for i in range(576):
    if (i < 192):
        thirdTrainingLabel.append(2)
    else:
        thirdTrainingLabel.append(3)

forthTrainingData = trainingdata[384:768]
forthTrainingData = np.array(forthTrainingData)
forthTrainingLabel = []
for i in range(384):
    if (i < 192):
        forthTrainingLabel.append(3)
    else:
        forthTrainingLabel.append(4)

fifthTrainingData = trainingdata[768:1536]
fifthTrainingData = np.array(fifthTrainingData)
fifthTrainingLabel = []
for i in range(768):
    if (i < 576):
        fifthTrainingLabel.append(7)
    else:
        fifthTrainingLabel.append(8)

sixthTrainingData = trainingdata[768:1344]
sixthTrainingData = np.array(sixthTrainingData)
sixthTrainingLabel = []
for i in range(576):
    if (i < 384):
        sixthTrainingLabel.append(6)
    else:
        sixthTrainingLabel.append(7)

seventhTrainingData = trainingdata[768:1152]
seventhTrainingData = np.array(seventhTrainingData)
seventhTrainingLabel = []
for i in range(384):
    if (i < 192):
        seventhTrainingLabel.append(5)
    else:
        seventhTrainingLabel.append(6)

testdata = []
for i in range(len(EntropyValidationData)):
    testdata.append([EntropyValidationData[i], VarianceValidationData[i]])
testdata = np.array(testdata)
firstTestData = testdata[0:512]
firstTestData = np.array(firstTestData)
firstTimeAnswer = []
for i in range(512):
    if (i < 256):
        firstTimeAnswer.append(1)
    else:
        firstTimeAnswer.append(2)

secondTestData = testdata[0:256]
secondTestData = np.array(secondTestData)
secondTimeAnswer = []
for i in range(256):
    if (i < 64):
        secondTimeAnswer.append(1)
    else:
        secondTimeAnswer.append(2)

thirdTestData = testdata[64:256]
thirdTestData = np.array(thirdTestData)
thirdTimeAnswer = []
for i in range(192):
    if (i < 64):
        thirdTimeAnswer.append(2)
    else:
        thirdTimeAnswer.append(3)

forthTestData = testdata[128:256]
forthTestData = np.array(forthTestData)
forthTimeAnswer = []
for i in range(128):
    if (i < 64):
        forthTimeAnswer.append(3)
    else:
        forthTimeAnswer.append(4)

fifthTestData = testdata[256:512]
fifthTestData = np.array(fifthTestData)
fifthTimeAnswer = []
for i in range(256):
    if (i < 192):
        fifthTimeAnswer.append(7)
    else:
        fifthTimeAnswer.append(8)

sixthTestData = testdata[256:448]
sixthTestData = np.array(sixthTestData)
sixthTimeAnswer = []
for i in range(192):
    if (i < 128):
        sixthTimeAnswer.append(6)
    else:
        sixthTimeAnswer.append(7)

seventhTestData = testdata[256:384]
seventhTestData = np.array(seventhTestData)
seventhTimeAnswer = []
for i in range(128):
    if (i < 64):
        seventhTimeAnswer.append(5)
    else:
        seventhTimeAnswer.append(6)

firstTrainingData = np.array(firstTrainingData, dtype='double')
secondTrainingData = np.array(secondTrainingData, dtype='double')
thirdTrainingData = np.array(thirdTrainingData, dtype='double')
forthTrainingData = np.array(forthTrainingData, dtype='double')
fifthTrainingData = np.array(fifthTrainingData, dtype='double')
sixthTrainingData = np.array(sixthTrainingData, dtype='double')
seventhTrainingData = np.array(seventhTrainingData, dtype='double')

firstTestData = np.array(firstTestData, dtype='double')
secondTestData = np.array(secondTestData, dtype='double')
thirdTestData = np.array(thirdTestData, dtype='double')
forthTestData = np.array(forthTestData, dtype='double')
fifthTestData = np.array(fifthTestData, dtype='double')
sixthTestData = np.array(sixthTestData, dtype='double')
seventhTestData = np.array(seventhTestData, dtype='double')

scaler = StandardScaler()
supportVectorClassifier1 = SVC(kernel='linear')
firstTrainingData = scaler.fit_transform(firstTrainingData)
firstTestData=scaler.fit_transform(firstTestData)
supportVectorClassifier1.fit(firstTrainingData, firstTrainingLabel)
prediction1 = supportVectorClassifier1.predict(firstTestData)

supportVectorClassifier2 = SVC(kernel='linear')
secondTrainingData = scaler.fit_transform(secondTrainingData)
secondTestData=scaler.fit_transform(secondTestData)
supportVectorClassifier2.fit(secondTrainingData, secondTrainingLabel)
prediction2 = supportVectorClassifier2.predict(secondTestData)
print(confusion_matrix(secondTimeAnswer, prediction2))
print(classification_report(secondTimeAnswer, prediction2))

supportVectorClassifier3 = SVC(kernel='linear')
thirdTrainingData = scaler.fit_transform(thirdTrainingData)
thirdTestData=scaler.fit_transform(thirdTestData)
supportVectorClassifier3.fit(thirdTrainingData, thirdTrainingLabel)
prediction3 = supportVectorClassifier3.predict(thirdTestData)
print(confusion_matrix(thirdTimeAnswer, prediction3))
print(classification_report(thirdTimeAnswer, prediction3))

supportVectorClassifier4 = SVC(kernel='linear')
forthTrainingData = scaler.fit_transform(forthTrainingData)
forthTestData=scaler.fit_transform(forthTestData)
supportVectorClassifier4.fit(forthTrainingData, forthTrainingLabel)
prediction4 = supportVectorClassifier4.predict(forthTestData)
print(confusion_matrix(forthTimeAnswer, prediction4))
print(classification_report(forthTimeAnswer, prediction4))

supportVectorClassifier5 = SVC(kernel='linear')
fifthTrainingData = scaler.fit_transform(fifthTrainingData)
fifthTestData=scaler.fit_transform(fifthTestData)
supportVectorClassifier5.fit(fifthTrainingData, fifthTrainingLabel)
prediction5 = supportVectorClassifier5.predict(fifthTestData)
print(confusion_matrix(fifthTimeAnswer, prediction5))
print(classification_report(fifthTimeAnswer, prediction5))

supportVectorClassifier6 = SVC(kernel='linear')
sixthTrainingData = scaler.fit_transform(sixthTrainingData)
sixthTestData=scaler.fit_transform(sixthTestData)
supportVectorClassifier6.fit(sixthTrainingData, sixthTrainingLabel)
prediction6 = supportVectorClassifier6.predict(sixthTestData)
print(confusion_matrix(sixthTimeAnswer, prediction6))
print(classification_report(sixthTimeAnswer, prediction6))

supportVectorClassifier7 = SVC(kernel='linear')
seventhTrainingData = scaler.fit_transform(seventhTrainingData)
seventhTestData=scaler.fit_transform(seventhTestData)
supportVectorClassifier7.fit(seventhTrainingData, seventhTrainingLabel)
prediction7 = supportVectorClassifier7.predict(seventhTestData)
print(confusion_matrix(seventhTimeAnswer, prediction7))
print(classification_report(seventhTimeAnswer, prediction7))

accuracycounter = 0
for i in range(len(secondTimeAnswer)):
    if secondTimeAnswer[i] != prediction2[i]:
        accuracycounter += 1
for i in range(len(thirdTimeAnswer)):
    if thirdTimeAnswer[i] != prediction3[i]:
        accuracycounter += 1
for i in range(len(forthTimeAnswer)):
    if forthTimeAnswer[i] != prediction4[i]:
        accuracycounter += 1
for i in range(len(fifthTimeAnswer)):
    if fifthTimeAnswer[i] != prediction5[i]:
        accuracycounter += 1
for i in range(len(sixthTimeAnswer)):
    if sixthTimeAnswer[i] != prediction6[i]:
        accuracycounter += 1
for i in range(len(seventhTimeAnswer)):
    if seventhTimeAnswer[i] != prediction7[i]:
        accuracycounter += 1

accuracy2 = (512-accuracycounter)/512
print('accuracy2: ', accuracy2)

plt.scatter(firstTrainingData[0:768, 0], firstTrainingData[0:768, 1], c='b', label='First Group')
plt.scatter(firstTrainingData[768:1536, 0], firstTrainingData[768:1536, 1], c='r', label='Second Group')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Two Groups')
X_axis1, Y_axis1, P1 = plot_svc_decision_function1(supportVectorClassifier1)
plt.legend()

plt.figure()
plt.scatter(secondTrainingData[0:192, 0], secondTrainingData[0:192, 1], c='r', label='Normal')
plt.scatter(secondTrainingData[192:768, 0], secondTrainingData[192:768, 1], c='g', label='Roller defect')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Normal & Roller defect')
X_axis2, Y_axis2, P2 = plot_svc_decision_function1(supportVectorClassifier2)
plt.legend()

plt.figure()
plt.scatter(thirdTrainingData[0:192, 0], thirdTrainingData[0:192, 1], c='g', label='Roller defect')
plt.scatter(thirdTrainingData[192:576, 0], thirdTrainingData[192:576, 1], c='orange', label='Inner defect')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Roller defect & Inner defect')
X_axis3, Y_axis3, P3 = plot_svc_decision_function1(supportVectorClassifier3)
plt.legend()

plt.figure()
plt.scatter(forthTrainingData[0:192, 0], forthTrainingData[0:192, 1], c='orange', label='Inner defect')
plt.scatter(forthTrainingData[192:384, 0], forthTrainingData[192:384, 1], c='cyan', label='Outer defect')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Inner defect & Outer defect')
X_axis4, Y_axis4, P4 = plot_svc_decision_function1(supportVectorClassifier4)
plt.legend()

plt.figure()
plt.scatter(fifthTrainingData[0:768, 0], fifthTrainingData[0:768, 1], c='orange', label='Outer&Inner&Roller defect')
plt.scatter(fifthTrainingData[576:768, 0], fifthTrainingData[576:768, 1], c='cyan', label='Outer&Roller defect')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Outer&Inner&Roller defect & Outer&Roller defect')
X_axis5, Y_axis5, P5 = plot_svc_decision_function1(supportVectorClassifier5)
plt.legend()

plt.figure()
plt.scatter(sixthTrainingData[0:576, 0], sixthTrainingData[0:576, 1], c='cyan', label='Inner&Roller defect')
plt.scatter(sixthTrainingData[384:576, 0], sixthTrainingData[384:576, 1], c='black', label='Outer&Inner&Roller defect')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Inner&Roller defect & Outer&Inner&Roller defect')
X_axis6, Y_axis6, P6 = plot_svc_decision_function1(supportVectorClassifier6)
plt.legend()

plt.figure()
plt.scatter(seventhTrainingData[0:384, 0], seventhTrainingData[0:384, 1], c='black', label='Inner&Roller defect')
plt.scatter(seventhTrainingData[192:384, 0], seventhTrainingData[192:384, 1], c='grey', label='Inner&Outer defect')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('SVM - Inner&Roller defect & Inner&Outer defect')
X_axis7, Y_axis7, P7 = plot_svc_decision_function1(supportVectorClassifier7)
plt.legend()

plt.figure()
plt.scatter(Alldata[0:192, 0], Alldata[0:192, 1], c='b', label='Normal')
plt.scatter(Alldata[192:384, 0], Alldata[192:384, 1], c='r', label='Defect 1')
plt.scatter(Alldata[384:576, 0], Alldata[384:576, 1], c='g', label='Defect 2')
plt.scatter(Alldata[576:768, 0], Alldata[576:768, 1], c='orange', label='Defect 3')
plt.scatter(Alldata[768:960, 0], Alldata[768:960, 1], c='cyan', label='Defect 4')
plt.scatter(Alldata[960:1152, 0], Alldata[960:1152, 1], c='black', label='Defect 5')
plt.scatter(Alldata[1152:1344, 0], Alldata[1152:1344, 1], c='grey', label='Defect 6')
plt.scatter(Alldata[1344:1536, 0], Alldata[1344:1536, 1], c='purple', label='Defect 7')
plt.xlabel('Entropy')
plt.ylabel('Variance')
plt.title('Training SVM')
plt.legend()
plt.show()