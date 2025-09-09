import numpy as np
import mindspore as ms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class ACAnalysis():
    def __init__(self):
        # Based on https://github.com/JonasGeiping/data-poisoning/blob/main/forest/filtering_defenses.py
        return

    def train(self, data, targets, cls_idx, clusters=2):
        # Convert MindSpore tensor to numpy if needed (per mindspore-exemptions.md #3)
        if isinstance(data, ms.Tensor):
            data_np = data.asnumpy()
        else:
            data_np = data
            
        # Iterating over all labels and normalize scores
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            temp_feats = data_np[idx]
            # Already numpy array, no need for .cpu().numpy()
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            score = silhouette_samples(temp_feats, kmeans.labels_)
            cls_scores.append(score)

        # extract score back to original sequence
        scores = []
        for i in range(data_np.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i]
            scores.append(s)
        scores = np.array(scores).flatten()
        self.mean = np.mean(scores)
        self.std = np.std(scores)

    def predict(self, data, targets, cls_idx, clusters=2, t=1):
        # Convert MindSpore tensor to numpy if needed (per mindspore-exemptions.md #3)
        if isinstance(data, ms.Tensor):
            data_np = data.asnumpy()
        else:
            data_np = data
            
        # Iterating over all labels and normalize scores
        cls_scores = []
        for idx in cls_idx:
            temp_feats = data_np[idx]
            # Already numpy array, no need for .cpu().numpy()
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            score = silhouette_samples(temp_feats, kmeans.labels_)
            cls_scores.append(score)

        # extract score back to original sequence
        scores = []
        for i in range(data_np.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i]
            scores.append(s)
        scores = np.array(scores).flatten()
        # Higher for backdoor
        p = (scores - self.mean) / self.std
        p = np.where((p > t) & (p > 0), 1, 0)
        return p

    def analysis(self, data, targets, cls_idx, clusters=2):
        """
            data (ms.Tensor or np.ndarray) b,c,h,w: data is the extracted feature from the model
        """
        # Convert MindSpore tensor to numpy if needed (per mindspore-exemptions.md #3)
        if isinstance(data, ms.Tensor):
            data_np = data.asnumpy()
        else:
            data_np = data
            
        # Iterating over all labels and normalize scores
        cls_scores = []
        for idx in cls_idx:
            temp_feats = data_np[idx]
            # Already numpy array, no need for .cpu().numpy()
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            score = silhouette_samples(temp_feats, kmeans.labels_)
            cls_scores.append(score)

        # extract score back to original sequence
        scores = []
        for i in range(data_np.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i]
            scores.append(s)
        scores = np.array(scores).flatten()
        return scores