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
#from imblearn.
import pickle
import joblib

names = ['Conf_ave','Zeros','Boxes', 'Low', 'count_low' ,'High', 'count_high','smode','mini','maxi','var','Class','Dummy']
dataset = read_csv('final-psm-3-opetus-4class.csv', names=names)
print(dataset.shape)
array = dataset.values
X = array[:,0:5]
y = array[:,11]
rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

#print(sorted(Counter(y_resampled).items()))

X_train, X_validation, Y_train, Y_validation = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='newton-cg')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	clf = model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
	filename = 're-psm3-' + name + '.sav'
	print(filename)
	clf = model.fit(X_train, Y_train)
	#filename = 'psm-3-lisa-tyhjilla-kasin-over.sav'
	joblib.dump(clf,filename)
	with open('clf.pickle','wb') as f:
		pickle.dump(clf,f)
#model = SVC(gamma='auto')
#model = KNeighborsClassifier()
#model = LogisticRegression(solver='newton-cg', multi_class='ovr')
#model  = LinearDiscriminantAnalysis()
#model = DecisionTreeClassifier()
#model = GaussianNB()


#predictions = model.predict(X_validation)
	
# Evaluate predictions
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#print(models)
