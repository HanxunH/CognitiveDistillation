import numpy as np
import paddle
from pyod.models.mad import MAD


def min_max_normalization(x):
    x_min = paddle.min(x)
    x_max = paddle.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


def get_ss_score(full_cov):
    """
    https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    """
    full_mean = np.mean(full_cov, axis=0, keepdims=True)
    centered_cov = full_cov - full_mean
    u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    eigs = v[0:1]
    # shape num_top, num_active_indices
    corrs = np.matmul(eigs, np.transpose(full_cov))
    scores = np.linalg.norm(corrs, axis=0)
    return scores


class SSAnalysis():
    def __init__(self):
        """
            Note that we assume the backdoor target label is unknown, 
            this may impacts performance of SS
        """
        return

    def train(self, data, targets, cls_idx):
        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            temp_feats = data[idx]
            scores = get_ss_score(temp_feats.flatten(start_axis=1).numpy())
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)[0]
            s = cls_scores[c][c_i[0]]
            scores.append(s)
        scores = np.array(scores)
        self.mean = np.mean(scores)
        self.std = np.std(scores)
        return

    def predict(self, data, targets, cls_idx, t=1):
        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            temp_feats = data[idx]
            scores = get_ss_score(temp_feats.flatten(start_axis=1).numpy())
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)[0]
            s = cls_scores[c][c_i[0]]
            scores.append(s)
        scores = np.array(scores)
        p = np.abs((self.mean - scores)) / self.std
        p = np.where((p > t), 1, 0)
        return p

    def analysis(self, data, targets, cls_idx):
        """
            data (paddle.Tensor) b,c,h,w: data is the extracted feature from the model
        """

        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            temp_feats = data[idx]
            scores = get_ss_score(temp_feats.flatten(start_axis=1).numpy())
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)[0]
            s = cls_scores[c][c_i[0]]
            scores.append(s)
        scores = np.array(scores).reshape(-1, 1)
        clf = MAD()  # This improves SS performance
        clf.fit(scores)
        return clf.decision_scores_