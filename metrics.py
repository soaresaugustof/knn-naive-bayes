import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted", zero_division: int = 0) -> float:
    classes = np.unique(y_true)
    precisions = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))

        if tp + fp == 0:
            precisions.append(zero_division)
        else:
            precisions.append(tp / (tp + fp))

    if average == "weighted":
        weights = np.array([np.sum(y_true == cls) for cls in classes]) / len(y_true)
        return np.sum(np.array(precisions) * weights)

    return np.mean(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted", zero_division: int = 0) -> float:
    classes = np.unique(y_true)
    recalls = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        if tp + fn == 0:
            recalls.append(zero_division)
        else:
            recalls.append(tp / (tp + fn))

    if average == "weighted":
        weights = np.array([np.sum(y_true == cls) for cls in classes]) / len(y_true)
        return np.sum(np.array(recalls) * weights)

    return np.mean(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted", zero_division: int = 0) -> float:
    prec = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average, zero_division=zero_division)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)
