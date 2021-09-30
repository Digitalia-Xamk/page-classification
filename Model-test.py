from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import pickle
import joblib
#import metrics as m 

names = ['Conf_ave','Zeros','Boxes', 'Low', 'Count_low', 'High', 'Count_high', 'smode','mini','maxi','var','Class','Dummy']
dataset = read_csv('final-psm-3-testiaineisto.csv', names=names)
print(dataset.shape)
array = dataset.values

X = array[:,0:5]
y = array[:,11]


loaded_model = joblib.load('psm3-KNN.sav')
print(' KNN ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('re-psm3-KNN.sav')
print(' re-KNN ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('psm3-LDA.sav')
print(' LDA ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
for i in range(0,len(y_pred)):
    if y_pred[i] == 'tyhjat' and y[i] != 'tyhjat':
        print(array[i,12])
        print(array[i,:])

result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))


loaded_model = joblib.load('re-psm3-LDA.sav')
print(' re-LDA ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('psm3-LR.sav')
print(' LR ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('re-psm3-LR.sav')
print(' re-LR ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('psm3-NB.sav')
print(' NB ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('re-psm3-NB.sav')
print(' re-NB ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))


loaded_model = joblib.load('psm3-SVM.sav')
print(' SVM ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('re-psm3-SVM.sav')
print(' re-SVM ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))


loaded_model = joblib.load('psm3-CART.sav')
print(' CART ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

loaded_model = joblib.load('re-psm3-CART.sav')
print(' re-CART ')
#print(loaded_model.predict(X))
y_pred = loaded_model.predict(X)
result = loaded_model.score(X,y)
print(classification_report(y,y_pred))
print(confusion_matrix(y,y_pred))

j = 0 
k = 0 
#predictions = model.predict(X_validation)
for i in range(0,len(y_pred)):
    if y_pred[i] != 'tyhjat' and  X[i,0]<=0.001  and y[i]=='tyhjat':
       # print(y_pred[i],array[i,12])
        k = k +1 
    elif array[i,11] != 'tyhjat' and X[i,0] <= 0.001:
        j = j + 1
        #print(y_pred[i],array[i,12])


print(j, k)
#print(classification_report(y,y_pred))
#predictions = model.predict(X_validation)
#print(result)

# Evaluate predictions
#print(accuracy_score(y, loaded_model))

