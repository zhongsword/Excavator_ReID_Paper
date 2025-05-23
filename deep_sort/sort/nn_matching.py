# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    # print(f"{x}, {type(x)}_cosine_distance")
    distances = _cosine_distance(x, y)

    return distances.min(axis=0)

def IoU(a, b):
    """
    Compute the IoU between two bbox
    Parameters:
    ----------
    a: array_like
        A bbox in (min x min y max x max y) format.
    b: array_like
        A bbox in (min x min y max x max y) format.
    """
    x1_min, y1_min, x1_max, y1_max = a
    x2_min, y2_min, x2_max, y2_max = b
    max_x_min, max_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    min_x_max, min_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, min_x_max - max_x_min) * max(0, min_y_max - max_y_min)
    union_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter_area
    IoU = inter_area / union_area

    return IoU


def _nn_iou(ax, b):
    """
    """
    min_iou = 1
    # ax = ax.tolist()
    for a in ax:
        iou = IoU(a, b)
        if iou < min_iou:
            min_iou = iou
    return min_iou


def _nn_iou_metric(a, bx):
    """
    Compute the IoU between two sets of bboxes
    Parameters
    ----------
    a : array_like
        An Nx4 matrix of N samples of bbox in (min x min y max x max y) format.
    b : array_like
        An Lx4 matrix of L samples of bbox in (min x min y max x max y) format.
    """
    restult = []
    # print(f"{a}, {type(a)}_iou")
    for i in bx:
        restult.append(_nn_iou(a, i))
    return 1. - np.array(restult)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self._nn_iou_metric = _nn_iou_metric
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        self.box_samples = {}

    def partial_fit(self, features, targets, active_targets, pre_bboxs, box_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        # self.box_samples.setdefault(target, pre_bboxs)
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        for bbox, active_target in zip(pre_bboxs, box_targets):
            self.box_samples.setdefault(active_target, []).append(bbox)
            # print(self.box_samples[active_target])
            if len(self.box_samples[active_target]) > 10:
                self.box_samples[active_target] = self.box_samples[active_target][-10:]

        self.samples = {k: self.samples[k] for k in active_targets}
        self.box_samples = {k: self.box_samples[k] for k in active_targets}

    def distance(self, features, targets, bbox):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        bbox : List[int]
            A list of detection_bbox in (min x min y max x max y) format

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        feature_cost_matrix = np.zeros((len(targets), len(features)))
        IoU_cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            feature_cost_matrix[i, :] = self._metric(self.samples[target], features)
            IoU_cost_matrix[i, :] = self._nn_iou_metric(self.box_samples[target], bbox)

        # print(f"{feature_cost_matrix},1")
        # print(f"{IoU_cost_matrix},2")
        cost_matrix = 0.9 * feature_cost_matrix + 0.1 * IoU_cost_matrix
        return cost_matrix
