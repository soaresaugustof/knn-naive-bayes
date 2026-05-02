import numpy as np


class NaiveBayesMultivariado:
    """
    Classificador Naive Bayes Gaussiano — Caso Multivariado.

    Modela a distribuição conjunta de todas as features usando a
    distribuição normal multivariada, capturando correlações entre elas:

        p(x | c) = 1 / sqrt((2π)^d * |Σ|) * exp(-1/2 * (x-μ)^T Σ^{-1} (x-μ))

    onde μ é o vetor de médias e Σ é a matriz de covariância da classe c.
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self._classes = None
        self._mean = None
        self._cov = None
        self._priors = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._cov = []
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            cov = np.cov(X_c, rowvar=False)
            self._cov.append(cov + np.eye(n_features) * self.epsilon)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _multivariate_pdf(self, class_idx: int, x: np.ndarray) -> float:
        mean = self._mean[class_idx]
        cov = self._cov[class_idx]
        d = mean.shape[0]
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        norm = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)
        diff = x - mean
        exponent = -0.5 * diff.T.dot(inv_cov).dot(diff)
        return norm * np.exp(exponent)

    def _predict_single(self, x: np.ndarray):
        posteriors = []
        for idx in range(len(self._classes)):
            prior = np.log(self._priors[idx])
            pdf_val = self._multivariate_pdf(idx, x)
            if pdf_val <= 0:
                pdf_val = 1e-300
            posteriors.append(prior + np.log(pdf_val))
        return self._classes[np.argmax(posteriors)]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x) for x in X_test])
