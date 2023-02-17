import numpy as np


class ClassBatchesSampler(object):
    def __init__(self, targets, num_class_samples=1, shuffle = True, oversample_seed = 200):
        self.targets = targets
        self.shuffle = shuffle
        self.num_batches = num_class_samples
        self.num_classes = len(np.unique(self.targets))
        maxLen = np.max(np.unique(targets, return_counts=True))
        # Find the indices of the samples for each class

        self.class_idxs = {}
        for i in range(self.num_classes):

            idxs = np.where(self.targets == i)[0]
            rng = np.random.default_rng(oversample_seed)
            extra = rng.choice(idxs, maxLen-len(idxs))
            if len(idxs) > 0:
                self.class_idxs[i] = np.append(idxs, extra)


    def __iter__(self):
        local_indices = self.class_idxs.copy()
        rng = np.random.default_rng()


        if self.shuffle:
            for i in list(local_indices.keys()):
                rng.shuffle(local_indices[i])

        for i in range(len(self.class_idxs[0])//self.num_batches):
            batch_idxs = []
            for k in range(self.num_batches):
                for i in list(local_indices.keys()):  # Make a copy of the keys
                    idxs = local_indices[i][:1]
                    batch_idxs.extend(idxs)
                    local_indices[i] = np.delete(local_indices[i], np.arange(len(idxs)))
            yield batch_idxs

    def __len__(self):
        return sum([len(idxs) for idxs in self.class_idxs.values()])
