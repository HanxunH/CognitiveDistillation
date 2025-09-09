import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class ACAnalysis():
    def __init__(self):
        # Based on https://github.com/JonasGeiping/data-poisoning/blob/main/forest/filtering_defenses.py
        return

    def train(self, data, targets, cls_idx, clusters=2):
        # Iterating over all labels and normalize scores
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            # Handle TensorFlow tensor indexing
            if isinstance(data, tf.Tensor):
                temp_feats = tf.gather(data, idx).numpy()
            elif hasattr(data, 'cpu'):  # Handle PyTorch tensors if present
                temp_feats = data[idx].cpu().numpy()
            else:  # Handle numpy arrays
                temp_feats = data[idx]
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            score = silhouette_samples(temp_feats, kmeans.labels_)
            cls_scores.append(score)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i]
            scores.append(s)
        scores = np.array(scores).flatten()
        self.mean = np.mean(scores)
        self.std = np.std(scores)

    def predict(self, data, targets, cls_idx, clusters=2, t=1):
        # Iterating over all labels and normalize scores
        cls_scores = []
        for idx in cls_idx:
            # Handle TensorFlow tensor indexing
            if isinstance(data, tf.Tensor):
                temp_feats = tf.gather(data, idx).numpy()
            elif hasattr(data, 'cpu'):  # Handle PyTorch tensors if present
                temp_feats = data[idx].cpu().numpy()
            else:  # Handle numpy arrays
                temp_feats = data[idx]
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            score = silhouette_samples(temp_feats, kmeans.labels_)
            cls_scores.append(score)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
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
            data (tf.Tensor) b,c,h,w: data is the extracted feature from the model
        """
        # Iterating over all labels and normalize scores
        cls_scores = []
        for idx in cls_idx:
            # Handle TensorFlow tensor indexing
            if isinstance(data, tf.Tensor):
                temp_feats = tf.gather(data, idx).numpy()
            elif hasattr(data, 'cpu'):  # Handle PyTorch tensors if present
                temp_feats = data[idx].cpu().numpy()
            else:  # Handle numpy arrays
                temp_feats = data[idx]
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            score = silhouette_samples(temp_feats, kmeans.labels_)
            cls_scores.append(score)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i]
            scores.append(s)
        scores = np.array(scores).flatten()
        return scores