from sklearn.decomposition import PCA
import numpy as np


class LabPCA(PCA):
    def transform_noncentered(self, X):
        originCoefficients = np.zeros(shape=X.shape)
        return self.transform(X) - self.transform(originCoefficients)

    @property
    def components_noncentered_(self):
        return self.components_ + self.mean_