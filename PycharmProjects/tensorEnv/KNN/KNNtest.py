import sklearn as sk
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# convertir nos données en valeurs numerique
le = preprocessing.LabelEncoder()
# "le" pour Label Encoder mais c'est ue variable donc on s'en fou du nom
buying = le.fit_transform(list(data["buy ]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
safety = le.fit_transform(list(data["safety"]))
classes = le.fit_transform(list(data["class"]))

print(buying)
predict = "class"
#zip permet de rassembler les list que l'on a créées
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
Y = list(classes)


bestAcc1 = 0
bestAcc2 = 0
accMoy1 = 0
accMoy2 = 0
nb_neighbors1 = 9
nb_neighbors2 = 7
accAvrg1 = accMoy1/50
accAvrg2 = accMoy2/50

 for i in range(50):
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.1)
    model1 = KNeighborsClassifier(n_neighbors=nb_neighbors1)
    model1.fit(X_train, Y_train)
    acc1 = model1.score(X_test, Y_test)
    accMoy1 += acc1

    model2 = KNeighborsClassifier(n_neighbors=nb_neighbors2)
    model2.fit(X_train, Y_train)
    acc2 = model2.score(X_test, Y_test)
    accMoy2 += acc2

    if acc1 > bestAcc1:
        bestAcc1 = acc1
    if acc2 > bestAcc2:
        bestAcc2 = acc2
accAvrg1 = accMoy1/50
accAvrg2 = accMoy2/50

print("your best acc1 is:", bestAcc1)
print("your avrg acc1 is:", accAvrg1)
print("your best acc2 is:", bestAcc2)
print("your avrg acc2 is:", accAvrg2)

#finding here whats the best n ,higher is not always good mais ce nest pas utile je ais ca juste pour le fun
if accAvrg2 < accAvrg1:
    bestModel = model1
    bestNb_neighbours = nb_neighbors1
else:
    bestModel = model2
    bestNb_neighbours = nb_neighbors2

predicted = bestModel.predict(X_test)
names = ["unacc", "acc", "good", "verygood"]

for i in range(len(predicted)):
    print("predicted value:", names[predicted[i]], "    Actual values:", names[Y_test[i]], "    data: ", X_test[i])
    knb = bestModel.kneighbors([X_test[i]], bestNb_neighbours, True) #get info about the nearest neighbours
    print(knb)