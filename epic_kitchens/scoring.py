from typing import List, Dict, Union

import numpy as np


def _scores_array_to_ranks(scores: np.ndarray):
    """
    The rank vector contains classes and is indexed by the rank

    Parameters
    ----------

    Examples
    --------
        >>> _scores_array_to_ranks(np.array([[0.1, 0.15, 0.25,  0.3, 0.5], \
                                             [0.5, 0.3, 0.25,  0.15, 0.1], \
                                             [0.2, 0.4,  0.1,  0.25, 0.05]]))
        array([[4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4],
               [1, 3, 0, 2, 4]])
    """
    assert scores.ndim == 2, "Expected scores to be 2 dimensional: [n_instances, n_classes]"
    return scores.argsort(axis=-1)[:, ::-1]


def _scores_dict_to_ranks(scores: List[Dict[int, float]]) -> np.ndarray:
    """
    Compute ranking from class to score dictionary

    Examples
    --------
        >>> _scores_dict_to_ranks([{0: 0.15, 1: 0.75, 2: 0.1},\
                                   {0: 0.85, 1: 0.10, 2: 0.05}])
        array([[1, 0, 2],
               [0, 1, 2]])
    """
    ranks = []
    for score in scores:
        class_ids = np.array(list(score.keys()))
        score_array = np.array([score[class_id] for class_id in class_ids])
        ranks.append(class_ids[np.argsort(score_array)[::-1]])
    return np.array(ranks)


def scores_to_ranks(scores: Union[np.ndarray, List[Dict[int, float]]]) -> np.ndarray:
    """
    Parameters
    ----------
    scores
        2D array of scores of shape (instance_count, class_count) for 'verb' or 'noun' tasks, or  a list of
        dictionaries, where each dictionary contains the mapping between 'action' classes
        ('verb'-'noun' pairs) and its score for a given segment.
    Returns
    -------
    ranks
        2D array of ranks computed from scores of shape (instance_count, class_count). Each element
        represents the position of that class in the descending order
    """
    if isinstance(scores, np.ndarray):
        return _scores_array_to_ranks(scores)
    elif isinstance(scores, list):
        return _scores_dict_to_ranks(scores)
    raise ValueError("Cannot compute ranks for type {}".format(type(scores)))


def scores_dict_to_ranks(scores_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    scores_dict
        Dictionary containing mapping between tasks ('verb', 'noun', 'action') and the corresponding
        2D array of scores of shape (instance_count, class_count)

    Returns
    -------
    ranks
        Dictionary containing mapping between tasks and their corresponding 2D array of ranks of shape
        (instance_count, class_count) computed from scores.
        Each element represents the position of that class in the descending order.
    """
    return {key: scores_to_ranks(scores) for key, scores in scores_dict.items()}