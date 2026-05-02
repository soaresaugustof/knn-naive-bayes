import numpy as np


def criarFolds(X, y, k=10, random_state=42):
    """
    Cria k folds para validação cruzada.

    Parâmetros:
    -----------
    X : np.ndarray
        Matriz de features (n_samples, n_features)
    y : np.ndarray
        Vetor de labels (n_samples,)
    k : int
        Número de folds (padrão: 10)
    random_state : int
        Seed para reprodutibilidade

    Retorna:
    --------
    folds : list of tuples
        Lista com k tuplas (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    tamanho_fold = n_samples // k
    folds = []

    for i in range(k):
        inicio = i * tamanho_fold
        fim = n_samples if i == k - 1 else inicio + tamanho_fold

        test_indices = indices[inicio:fim]
        train_indices = np.concatenate([indices[:inicio], indices[fim:]])

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        folds.append((X_train, X_test, y_train, y_test))

    return folds
