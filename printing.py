"""
Contains functions used for printing in the notebooks.
"""
from sklearn import metrics


def print_classifier_stats(classifier, test_x, test_y):
    test_acc = metrics.accuracy_score(classifier.predict(test_x), test_y)
    print("Test accuracy: %0.3f" % (test_acc))

    test_bacc = metrics.balanced_accuracy_score(classifier.predict(test_x), test_y)
    print("Test balanced accuracy: %0.3f" % (test_bacc))

    test_prec = metrics.precision_score(classifier.predict(test_x), test_y)
    print("Test precision: %0.3f" % (test_prec))

    test_rec = metrics.recall_score(classifier.predict(test_x), test_y)
    print("Test recall: %0.3f" % (test_rec))

    test_f1 = metrics.f1_score(classifier.predict(test_x), test_y)
    print("Test F1: %0.3f" % (test_f1))
