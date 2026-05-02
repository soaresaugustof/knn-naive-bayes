import numpy as np


class RegressaoLinear:
    """
    Regressão Linear Múltipla.

    Encontra os coeficientes β que minimizam o erro quadrático:

        β = (X^T X)^{-1} X^T y
    """

    def __init__(self):
        self._coef = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Adiciona coluna de 1s para o intercepto
        X_b = np.column_stack([np.ones(len(X_train)), X_train])
        # Mínimos quadrados via numpy (mais estável que inversa direta)
        self._coef, _, _, _ = np.linalg.lstsq(X_b, y_train, rcond=None)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_b = np.column_stack([np.ones(len(X_test)), X_test])
        return X_b @ self._coef
