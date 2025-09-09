import numpy as np
import mindspore as ms
from pyod.models.mad import MAD


def min_max_normalization(x):
    """
    Note: This function is defined but not used in the original code
    Keeping it for compatibility
    """
    if isinstance(x, ms.Tensor):
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = np.min(x)
        x_max = np.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


def get_ss_score(full_cov):
    """
    https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    Note: This function expects numpy arrays as input
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
        # Convert MindSpore tensor to numpy if needed
        if isinstance(data, ms.Tensor):
            data_np = data.asnumpy()
        else:
            data_np = data
            
        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            temp_feats = data_np[idx]
            # Flatten the features starting from dimension 1
            if len(temp_feats.shape) > 2:
                temp_feats_flat = temp_feats.reshape(temp_feats.shape[0], -1)
            else:
                temp_feats_flat = temp_feats
            scores = get_ss_score(temp_feats_flat)
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data_np.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i][0]
            scores.append(s)
        scores = np.array(scores)
        self.mean = np.mean(scores)
        self.std = np.std(scores)
        return

    def predict(self, data, targets, cls_idx, t=1):
        # Convert MindSpore tensor to numpy if needed
        if isinstance(data, ms.Tensor):
            data_np = data.asnumpy()
        else:
            data_np = data
            
        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            temp_feats = data_np[idx]
            # Flatten the features starting from dimension 1
            if len(temp_feats.shape) > 2:
                temp_feats_flat = temp_feats.reshape(temp_feats.shape[0], -1)
            else:
                temp_feats_flat = temp_feats
            scores = get_ss_score(temp_feats_flat)
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data_np.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i][0]
            scores.append(s)
        scores = np.array(scores)
        p = np.abs((self.mean - scores)) / self.std
        p = np.where((p > t), 1, 0)
        return p

    def analysis(self, data, targets, cls_idx):
        """
            data (ms.Tensor or np.ndarray) b,c,h,w: data is the extracted feature from the model
        """
        # Convert MindSpore tensor to numpy if needed
        if isinstance(data, ms.Tensor):
            data_np = data.asnumpy()
        else:
            data_np = data

        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            temp_feats = data_np[idx]
            # Flatten the features starting from dimension 1
            if len(temp_feats.shape) > 2:
                temp_feats_flat = temp_feats.reshape(temp_feats.shape[0], -1)
            else:
                temp_feats_flat = temp_feats
            scores = get_ss_score(temp_feats_flat)
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data_np.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i][0]
            scores.append(s)
        scores = np.array(scores).reshape(-1, 1)
        clf = MAD()  # This improves SS performance
        clf.fit(scores)
        return clf.decision_scores_