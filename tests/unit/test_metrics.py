import pandas as pd
import numpy as np
import pytest

from epic_kitchens.metrics import compute_metrics, precision_recall, topk_accuracy


#def test_compute_metrics():
#    groundtruth_df = pd.DataFrame({
#        'verb_class': [0, 1, 0, 1, 0, 1],
#        'noun_class': [4, 4, 114, 114, 114, 114]
#    }, index=np.array([1924, 1925, 1926, 1927, 1928, 1929]))
#
#    scores = {
#        'verb': np.array([]),
#        'noun':  np.array([]),
#        'action': [{(0, 4): 0.12}]
#    }
#
#    many_shot_verbs = np.array([0, 1])
#    many_shot_nouns = np.array([4])

#    compute_metrics(groundtruth_df, scores, many_shot_verbs, many_shot_nouns)


def test_precision_all_tp():

    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([1, 2, 3])
    precision, _ = precision_recall(ranks, labels)

    assert np.all(precision == np.array([1, 1, 1]))


def test_precision_all_fp():

    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([3, 1, 2])
    precision, _ = precision_recall(ranks, labels)

    assert np.all(precision == np.array([0, 0, 0]))


def test_precision_no_fp_and_no_tp():
    ranks = np.array([[4, 2, 3],
                      [4, 3, 1],
                      [4, 1, 2]])
    labels = np.array([3, 1, 2])
    precision, _ = precision_recall(ranks, labels)

    assert np.all(precision == np.array([0, 0, 0]))


def test_precision_recall_filter_existing_class():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([1, 2, 3])
    precision, recall = precision_recall(ranks, labels, classes=np.array([1]))

    assert np.all(precision == np.array([1]))
    assert np.all(recall == np.array([1]))


def test_precision_recall_filter_nonexisting_class():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        precision, _ = precision_recall(ranks, labels, classes=np.array([4]))


def test_precision_recall_throws_exception_if_labels_and_ranks_are_different_lengths():
    ranks = np.array([[2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        precision, _ = precision_recall(ranks, labels)


def test_recall_all_tp():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([1, 2, 3])

    _, recall = precision_recall(ranks, labels)

    assert np.all(recall == np.array([1, 1, 1]))


def test_recall_all_fn():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([2, 3, 1])

    _, recall = precision_recall(ranks, labels)

    assert np.all(recall == np.array([0, 0, 0]))


def test_accuracy_at_1():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2]])
    labels = np.array([1, 2, 3])

    accuracy = topk_accuracy(ranks, labels, ks=1)

    assert accuracy == 1


def test_accuracy_at_2():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2],
                      [1, 2, 3]])
    labels = np.array([1, 3, 2, 1])

    accuracy = topk_accuracy(ranks, labels, ks=2)

    assert accuracy == 0.75


def test_accuracy_at_3():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2],
                      [1, 2, 3]])
    labels = np.array([1, 3, 2, 1])

    accuracy = topk_accuracy(ranks, labels, ks=3)

    assert accuracy == 1


def test_accuracy_at_1_and_3():
    ranks = np.array([[1, 2, 3],
                      [2, 3, 1],
                      [3, 1, 2],
                      [1, 2, 3]])
    labels = np.array([1, 3, 2, 1])

    accuracy = topk_accuracy(ranks, labels, ks=(1, 3, 5))

    assert np.all(accuracy == np.array([0.5, 1, 1]))
