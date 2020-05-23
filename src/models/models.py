from keras.utils import to_categorical
from src.models.lstm import calc_recall2
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def my_macro_recall_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return recall_score(y_true=y, y_pred=predictions, average="macro")


def svc_linear_cross(features, labels, c=1, cv=5):
    clf = SVC(kernel='linear', C=c)
    scores = cross_val_score(clf, features, labels,
                             scoring=my_macro_recall_scorer, cv=cv)
    return scores.mean(), scores.std() * 2


def svc_linear(train_features, train_labels, test_features, test_labels, c=1):
    clf = SVC(kernel='linear', C=c)
    print("Fitting..")
    clf.fit(train_features, train_labels)
    test_labels_cat = to_categorical(test_labels)
    predictions = to_categorical(clf.predict(test_features))
    print("Recall...")
    recall = calc_recall2(predictions, test_labels)
    return recall


def parallel_svm(train_features, train_labels, test_features, test_labels, c=1, n_jobs=12):
    clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'),
                                                max_samples=1.0 / n_jobs, n_estimators=n_jobs, bootstrap=False))
    clf.fit(train_features, train_labels)
    predictions = to_categorical(clf.predict(test_features))
    recall = calc_recall2(predictions, test_labels)
    return recall
