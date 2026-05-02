import numpy as np


class NaiveBayesUnivariado:
    """
    Classificador Naive Bayes Gaussiano — Caso Univariado.

    Assume independência entre as features. Modela cada feature
    separadamente com uma distribuição normal univariada:

        p(x | c) = p(x1|c) * p(x2|c) * ... * p(xn|c)
    """

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + self.epsilon
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x: np.ndarray):
        posteriors = []
        for idx in range(len(self._classes)):
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + likelihood)
        return self._classes[np.argmax(posteriors)]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x) for x in X_test])
