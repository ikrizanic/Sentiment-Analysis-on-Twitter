from random import randrange

from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


def cross_validate(classifier, features, labels, number_of_cv=5):
    scores = list()
    for i in range(number_of_cv):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=randrange(100000))
        print("Training cv number: " + str(i+1))
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        scores.append(recall_score(y_test, prediction, average="micro"))

    return scores
