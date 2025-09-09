import mindspore as ms


def min_max_normalization(x):
    # MindSpore ops.min/max return tuple (value, index) when called without axis
    # We only need the value part
    x = x.astype(ms.float32)
    x_min = ms.ops.min(x)
    x_max = ms.ops.max(x)
    # Handle tuple return - ops.min/max return (value, index)
    if isinstance(x_min, tuple):
        x_min = x_min[0]
    if isinstance(x_max, tuple):
        x_max = x_max[0]
    norm = (x - x_min) / (x_max - x_min)
    return norm


class ABLAnalysis():
    def __init__(self):
        return

    def analysis(self, data):
        """
            data np.array
            sample-wise training loss shape (epoch, n_samples)
            return first 20 epoch avgs for each sample
        """
        # lower for bd, use 1 - score
        # Convert numpy array mean to float32 first
        avg_data = data[:20, :].mean(axis=0).astype('float32')
        return 1 - min_max_normalization(ms.Tensor(avg_data)).asnumpy()