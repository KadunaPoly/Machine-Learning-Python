import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
###print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
###print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
X_train,  X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.1)
    ###
    # on forme 4 tableaux, lordre est des variable est important
    # testsize size de notre model en % on ne prend que 10 % car on veut qu'il s'entraine sur un petit echantillon
    # si on entraine le model sur l'echantillon entier ce n'est plus vraiment de la prediction car l'ordinateur va retrouver les valeurs sur lesquelles ils s'est entrainer,
    # (retrouver des memes paternes)
    # X_train et Y_train sont des sections de chaque tableau
    # les test sont ici pour verifiers la precision de notre model

bestAcc = 0
nbIteration = 30
#on fait une loop 30 fois arbitraire pour ecrire sur les 30 teste le model qui a la meilleure acc
for i in range(nbIteration):
    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train)
    acc = linear.score(X_test, Y_test)
    print("your accuracy is :", acc)
    if acc > bestAcc:
        bestAcc = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
        # pickle permet de sauvegarder le model pour ne pas avoir a le reproduire a chaque fois lorsqu'on utilise des grosses qté de data
        #je nomme mon fichier comme je le veux, ici studentmodel et je le dump(creeer un fichier) avec un model linear
        #wb pour ecrire et rb pour lire (je pense)
        #on va lire dedqns par la suite

print("out of the", nbIteration, "the best model has an acc of", bestAcc)

pickle_model = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_model)
#une fois le pickle créé on peut se debarrasser de tout la partit qui créé le model lineair et de la partie qui créer le pickle car il existe deja apres le premier run


print("coef de linearisation: ", linear.coef_)
#le coef est les m de lequation y = mx+b  , nous avons parametre etudiés donc 5 coef
print("en y=o , x=", linear.intercept_)

predictions = linear.predict(X_test)
for i in range(len(predictions)):
    print(predictions[i], X_test[i], Y_test[i])

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"]) # notre axe x et y , ici y serra toujours G3 donc pas besoin de variables
pyplot.xlabel(p)
pyplot.ylabel("final grade")
#nommer les axes
pyplot.show() #print pour un plot
