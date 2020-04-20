from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import recall_score


def my_macro_recall_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return recall_score(y_true=y, y_pred=predictions, average="macro")


def svc_linear(features, labels, c=1, cv=5):
    clf = svm.SVC(kernel='linear', C=c)
    scores = cross_val_score(clf, features, labels,
                             scoring=my_macro_recall_scorer, cv=cv)
    return scores.mean(), scores.std() * 2

def svc_rbf(features, labels, rs=1, cv=5):
    classifier = SVC(kernel='rbf', random_state=rs)
    scores = cross_val_score(classifier, features, labels, scoring=my_macro_recall_scorer, cv=cv)
    print(scores)
    return scores.mean(), scores.std() * 2
