#lib import
from codecs import xmlcharrefreplace_errors
from pandas import read_csv #work with online data
from pandas.plotting import scatter_matrix #need to check wtf
from matplotlib import pyplot #check 
from sklearn.model_selection import train_test_split #check
from sklearn.model_selection import cross_val_score #check 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression #way to ml
from sklearn.tree import DecisionTreeClassifier # ml
from sklearn.neighbors import KNeighborsClassifier #ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB #ml
from sklearn.svm import SVC #ml

#download dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#shape. Checking how many data we got.
print(dataset.shape)
#checking first 20 pos of data 
print(dataset.head(20))
#statistic by desribe method
print(dataset.describe())
#class atribute sort
print(dataset.groupby('class').size())

#box and whiskers diagram 
#dataset.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False)
#pyplot.show()
#histogram of data
#dataset.hist()
#pyplot.show()
#scatter matrix of data
#scatter_matrix(dataset)
#pyplot.show()

#divide data for two pieces in 80% and 20% for learning and testing
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.20, random_state = 1)

#making algs for models
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))

#making score-models
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#pyplot.boxplot(results, labels = names)
#pyplot.title('Algorithm Comprasion')
#pyplot.show()

#make an predict on test array
model = SVC(gamma = 'auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

#score predict
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

print ('all good, go forward! ')
