from typing import List, Tuple, Dict, Union, Sequence
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import logging

from .scoring import scores_dict_to_ranks

LOG = logging.getLogger(__name__)


def compute_metrics(groundtruth_df: pd.DataFrame,
                    scores: Dict[str, np.ndarray],
                    many_shot_verbs: Sequence[int],
                    many_shot_nouns: Sequence[int],
                    many_shot_actions: Sequence[int]):
    """
    Parameters
    ----------
    groundtruth_df
        DataFrame containing 'verb_class': int, 'noun_class': int and 'action_class': Tuple[int, int] columns.
    scores
        Dictionary containing three entries: 'verb', 'noun' and 'action' entries should map to a 2D
        np.ndarray of shape (instance_count, class_count) where each element is the predicted score
        of that class.
        TODO
    many_shot_verbs
        The set of verb classes that are considered many shot
    many_shot_nouns
        The set of noun classes that are considered many shot
    many_shot_actions
        The set of action classes that are considered many shot

    Returns
    -------
    (class_aware_metrics, class_agnostic_metrics)
        A tuple of two dictionaries containing nested metrics.

    Raises
    ------
    ValueError
        If the shapes of the score arrays are not correct, or the lengths of the groundtruth_df and the
        scores array are not equal, or if the grountruth_df doesn't have the specified columns.
    """
    for entry in 'verb', 'noun', 'action':
        class_col = entry + '_class'
        if class_col not in groundtruth_df.columns:
            raise ValueError("Expected '{}' column in groundtruth_df".format(class_col))

    ranks = scores_dict_to_ranks(scores)
    top_k = (1, 5)

    accuracies = compute_class_aware_metrics(groundtruth_df, ranks, top_k)
    precision_recall_metrics = compute_class_agnostic_metrics(groundtruth_df, ranks,
                                                              many_shot_verbs, many_shot_nouns,
                                                              many_shot_actions)

    return {
        'accuracy': {
            'verb': accuracies['verb'],
            'noun': accuracies['noun'],
            'action': accuracies['action'],
        },
        **precision_recall_metrics
    }


def compute_class_aware_metrics(groundtruth_df: pd.DataFrame, ranks: Dict[str, np.ndarray],
                                top_k: Union[int, Tuple[int, ...]] = (1, 5)) -> \
        Dict[str, Union[float, Union[float, List[float]]]]:
    """
    Compute class aware metrics dictionary

    Parameters
    ----------
    groundtruth_df
        DataFrame containing 'verb_class': int, 'noun_class': int and 'action_class': Tuple[int, int] columns.
    ranks
        Dictionary containing three entries: 'verb', 'noun' and 'action' entries should map to a 2D
        np.ndarray of shape (instance_count, class_count) where each element is the predicted rank of that class.
        The 'action' rank array should be
        TODO
    top_k
        The set of k values to compute top-k accuracy for.

    Returns
    -------
    Dict[str, Union[float, Union[float, List[float]]]]
        Dictionary with two keys 'precision' and 'recall', each sub dictionary has the keys 'action',
        'verb', 'noun' and 'verb_per_class'. The 'verb' and 'noun' entries of the metric dictionaries
        are the macro-averaged mean precision/recall over the set of many shot classes, whereas the
        'verb_per_class' entry is a breakdown for each verb_class in the format of a dictionary mapping
        stringified verb class to that class' precision/recall.
    """
    verb_accuracies = topk_accuracy(ranks['verb'], groundtruth_df['verb_class'].values,
                                    ks=top_k)
    noun_accuracies = topk_accuracy(ranks['noun'], groundtruth_df['noun_class'].values,
                                    ks=top_k)
    action_accuracies = topk_accuracy(ranks['action'], groundtruth_df['action_class'].values)
    return {
        'verb': verb_accuracies,
        'noun': noun_accuracies,
        'action': action_accuracies,
    }


def compute_class_agnostic_metrics(groundtruth_df: pd.DataFrame, ranks: Dict[str, np.ndarray],
                                   many_shot_verbs: Sequence[int],
                                   many_shot_nouns: Sequence[int],
                                   many_shot_actions: Sequence[int]) -> \
        Dict[str, Dict[str, Union[np.float, Dict[str, np.float]]]]:
    """
    Compute class agnostic metrics dictionary

    Parameters
    ----------
    groundtruth_df
        DataFrame containing 'verb_class': int, 'noun_class': int and 'action_class': Tuple[int, int] columns.
    ranks
        Dictionary containing three entries: 'verb', 'noun' and 'action' entries should map to a 2D
        np.ndarray of shape (instance_count, class_count) where each element is the predicted rank of that class.
        The 'action' rank array should be
    many_shot_verbs
        The set of verb classes that are considered many shot
    many_shot_nouns
        The set of noun classes that are considered many shot
    many_shot_actions
        The set of action classes that are considered many shot

    Returns
    -------
    Dict[str, Dict[str, Union[np.float, Dict[str, np.float]]]]
        Dictionary with two keys 'precision' and 'recall', each sub dictionary has the keys 'action',
        'verb', 'noun' and 'verb_per_class'. The 'verb' and 'noun' entries of the metric dictionaries
        are the macro-averaged mean precision/recall over the set of many shot classes, whereas the
        'verb_per_class' entry is a breakdown for each verb_class in the format of a dictionary mapping
        stringified verb class to that class' precision/recall.
    """

    many_shot_verbs = _exclude_non_existent_classes(many_shot_verbs, groundtruth_df['verb_class'])
    many_shot_nouns = _exclude_non_existent_classes(many_shot_nouns, groundtruth_df['noun_class'])
    many_shot_actions = _exclude_non_existent_classes(many_shot_actions, groundtruth_df['action_class'])

    verb_precision, verb_recall = precision_recall(ranks['verb'], groundtruth_df.verb_class,
                                                   classes=many_shot_verbs)
    noun_precision, noun_recall = precision_recall(ranks['noun'], groundtruth_df.noun_class,
                                                   classes=many_shot_nouns)
    LOG.debug('{} many shot actions before intersecting with actions present in test'.format(
            len(many_shot_actions)))
    LOG.info('{} many shot actions after intersecting with actions present in test'.format(
            len(many_shot_actions)))
    action_precision, action_recall = precision_recall(ranks['action'], groundtruth_df.action_class,
                                                       classes=many_shot_actions)
    precision_many_shot_verbs = {str(verb): score for verb, score in zip(many_shot_verbs, verb_precision)}
    recall_many_shot_verbs = {str(verb): score for verb, score in zip(many_shot_verbs, verb_recall)}

    return {
        'precision': {
            'action': action_precision.mean(),
            'verb': verb_precision.mean(),
            'noun': noun_precision.mean(),
            'verb_per_class': precision_many_shot_verbs
        },
        'recall': {
            'action': action_recall.mean(),
            'verb': verb_recall.mean(),
            'noun': noun_recall.mean(),
            'verb_per_class': recall_many_shot_verbs
        },
    }


def topk_accuracy(rankings: np.ndarray, labels: np.ndarray,
                  ks: Union[Tuple[int, ...], int] = (1, 5)) -> Union[float, List[float]]:
    """
    Computes TOP-K accuracies for different values of k

    Parameters:
    -----------
    rankings
        2D rankings array: shape = (instance_count, label_count)
    labels
        1D correct labels array: shape = (instance_count,)
    ks
        The k values in top-k, either an int or a list of ints.

    Returns:
    --------
    list of float: TOP-K accuracy for each k in ks

    Raises:
    -------
    ValueError
         If the dimensionality of the rankings or labels is incorrect, or
         if the length of rankings and labels aren't equal
    """
    if isinstance(ks, int):
        ks = (ks,)
    _check_label_predictions_preconditions(rankings, labels)

    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    accuracies = [tp[:, :k].max(1).mean() for k in ks]
    if len(accuracies) == 1:
        return accuracies[0]
    else:
        return accuracies


def precision_recall(rankings: np.ndarray, labels: np.ndarray, classes=None) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Computes precision and recall
    Parameters:
    -----------
    rankings: numpy.ndarray, shape = (instance_count, label_count)
    labels: numpy.ndarray, shape = (instance_count,)
    classes: numpy.ndarray, shape = (relevant_label_count,)

    Returns:
    --------
    (np.ndarray, np.ndarray)
        of dimension 1
        precision values: np.ndarray, shape = (relevant_label_count,)
        recal values: np.ndarray float, shape = (relevant_label_count,)

    Raises:
    -------
    ValueError
         If the dimensionality of the rankings or labels is incorrect, or if the length of the
         rankings and labels are equal, or if the set of the provided classes is not a subset
         of the classes present in the labels.
    """
    _check_label_predictions_preconditions(rankings, labels)
    y_pred = rankings[:, 0]
    if classes is None:
        classes = np.unique(labels)
    else:
        provided_class_presence = np.in1d(classes, np.unique(labels))
        if not np.all(provided_class_presence):
            raise ValueError("Classes {} are not in labels".format(classes[provided_class_presence]))
    precision, recall, _, _ = precision_recall_fscore_support(labels, y_pred, labels=classes,
                                                              average=None, warn_for=tuple('recall'))
    return precision, recall


def _exclude_non_existent_classes(classes: Sequence[int], labels: pd.Series):
    return np.intersect1d(classes, labels.unique())


def _check_label_predictions_preconditions(rankings: np.ndarray, labels: np.ndarray):
    if not len(rankings.shape) == 2:
        raise ValueError("Rankings should be a 2D matrix")
    if not len(labels.shape) == 1:
        raise ValueError("Labels should be a 1D vector")
    if not labels.shape[0] == rankings.shape[0]:
        raise ValueError("Number of labels provided does not match number of predictions")
