import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)
classes = ["malin", "benin"]

classifier = svm.SVC(kernel='linear', C=2) #il y a bcp de parametres avec lesquelles on peut jouer pour augmenter la precision  .SVC(kernel='poly', degree= 2)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

acc1 = metrics.accuracy_score(y_test, y_predict)
print("your SVM acc is ", acc1)
modelKnn = KNeighborsClassifier(n_neighbors=9)
modelKnn.fit(x_train, y_train)
acc2 = modelKnn.score(x_test, y_test)
print("your KNN acc is ", acc2)
