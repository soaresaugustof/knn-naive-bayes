import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models import KNN, NaiveBayesUnivariado, NaiveBayesMultivariado, RegressaoLinear
from utils import criarFolds
from metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score


DATA_DIR = Path("data")
DEFAULT_DATASET = DATA_DIR / "dataset.csv"


def _is_numeric_type(attr_type: str) -> bool:
    if _is_nominal_type(attr_type):
        return False
    return any(t in attr_type for t in ('numeric', 'real', 'integer'))


def _is_nominal_type(attr_type: str) -> bool:
    return attr_type.strip().startswith('{')


def load_arff(file_path: str, target_column: str = None, exclude_columns: list = None) -> tuple:

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    attributes = []
    attribute_types = {}
    data_started = False
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line.lower().startswith('@attribute'):
            rest = line[10:].strip()
            # nome pode estar entre aspas
            if rest.startswith("'"):
                end = rest.index("'", 1)
                attr_name = rest[1:end]
                attr_type = rest[end + 1:].strip().lower()
            else:
                parts = rest.split(None, 1)
                attr_name = parts[0]
                attr_type = parts[1].lower() if len(parts) > 1 else 'string'
            attributes.append(attr_name)
            attribute_types[attr_name] = attr_type
        elif line.lower().startswith('@data'):
            data_started = True
            continue
        elif data_started and line:
            data_lines.append(line)

    # Determina índice do alvo
    if target_column is None:
        target_idx = len(attributes) - 1
    else:
        if target_column not in attributes:
            raise ValueError(f"Target column '{target_column}' not found. Available: {attributes}")
        target_idx = attributes.index(target_column)

    exclude = set(exclude_columns or [])
    feature_names = [a for i, a in enumerate(attributes) if i != target_idx and a not in exclude]

    # Coleta todos os valores brutos para montar dicionários de label encoding
    raw_rows = []
    for line in data_lines:
        if '%' in line:
            line = line[:line.index('%')]
        line = line.strip()
        if not line:
            continue
        values = []
        current = ""
        in_quotes = False
        for char in line:
            if char == '"' or char == "'":
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current.strip().strip('"').strip("'"))
                current = ""
            else:
                current += char
        if current:
            values.append(current.strip().strip('"').strip("'"))
        if len(values) == len(attributes):
            raw_rows.append(values)

    # Constrói dicionários de label encoding para atributos nominais
    label_maps = {}
    for attr_name in attributes:
        attr_type = attribute_types[attr_name]
        if _is_nominal_type(attr_type):
            # extrai categorias do tipo "{a, b, c}"
            cats_str = attr_type.strip().strip('{}')
            cats = [c.strip() for c in cats_str.split(',')]
            label_maps[attr_name] = {cat: i for i, cat in enumerate(cats)}

    X = []
    y = []

    for values in raw_rows:
        row_all = []
        for val, attr_name in zip(values, attributes):
            attr_type = attribute_types[attr_name]
            if _is_numeric_type(attr_type):
                try:
                    row_all.append(float(val))
                except ValueError:
                    row_all.append(float('nan'))
            elif attr_name in label_maps:
                row_all.append(float(label_maps[attr_name].get(val, -1)))
            else:
                row_all.append(val)

        feat_row = [row_all[i] for i, a in enumerate(attributes) if i != target_idx and a not in exclude]
        target_val = row_all[target_idx]
        X.append(feat_row)
        y.append(target_val)

    return np.array(X, dtype=float), np.array(y), feature_names


def load_dataset(dataset_path: Path, target_column: str, feature_columns: Optional[List[str]] = None):
    with open(dataset_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames)

    if target_column not in columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {columns}")

    if feature_columns is None:
        feature_columns = [col for col in columns if col != target_column]

    missing = [col for col in feature_columns if col not in columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    X = np.array([[float(row[col]) for col in feature_columns] for row in rows], dtype=float)
    y = np.array([row[target_column] for row in rows])

    return X, y, feature_columns


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def regression_metrics(y_true, y_pred, n_features) -> Dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {"r2": r2, "adjusted_r2": adjusted_r2}


def run_classification_cv(X, y, k_folds=10):
    folds = criarFolds(X, y, k=k_folds)

    models = {
        "knn_euclidean": lambda: KNN(k=5, task="classification", distance="euclidean"),
        "knn_manhattan": lambda: KNN(k=5, task="classification", distance="manhattan"),
        "naive_bayes_univariado": NaiveBayesUnivariado,
        "naive_bayes_multivariado": NaiveBayesMultivariado,
    }

    results = {name: {m: [] for m in ["accuracy", "precision", "recall", "f1"]} for name in models}
    train_times = {name: [] for name in models}
    test_times = {name: [] for name in models}

    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{k_folds}")

        for model_name, model_fn in models.items():
            model = model_fn()

            start_train = time.time()
            model.fit(X_train, y_train)
            train_times[model_name].append(time.time() - start_train)

            start_test = time.time()
            y_pred = model.predict(X_test)
            test_times[model_name].append(time.time() - start_test)

            for metric_name, value in classification_metrics(y_test, y_pred).items():
                results[model_name][metric_name].append(value)

    final_results = {}
    for model_name in models:
        final_results[model_name] = {}
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            values = results[model_name][metric_name]
            final_results[model_name][metric_name] = (np.mean(values), np.std(values))
        final_results[model_name]["train_time"] = (np.mean(train_times[model_name]), np.std(train_times[model_name]))
        final_results[model_name]["test_time"] = (np.mean(test_times[model_name]), np.std(test_times[model_name]))

    return final_results


def run_regression_cv(X, y, k_folds=10):
    folds = criarFolds(X, y, k=k_folds)

    models = {
        "knn_euclidean":  lambda: KNN(k=5, task="regression", distance="euclidean"),
        "knn_manhattan":  lambda: KNN(k=5, task="regression", distance="manhattan"),
        "regressao_linear": RegressaoLinear,
    }

    results = {name: {m: [] for m in ["r2", "adjusted_r2"]} for name in models}
    train_times = {name: [] for name in models}
    test_times  = {name: [] for name in models}

    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{k_folds}")

        for model_name, model_fn in models.items():
            model = model_fn()

            start_train = time.time()
            model.fit(X_train, y_train)
            train_times[model_name].append(time.time() - start_train)

            start_test = time.time()
            y_pred = model.predict(X_test)
            test_times[model_name].append(time.time() - start_test)

            n_features = X_test.shape[1]
            for metric_name, value in regression_metrics(y_test, y_pred, n_features).items():
                results[model_name][metric_name].append(value)

    final_results = {}
    for model_name in models:
        final_results[model_name] = {}
        for metric_name in ["r2", "adjusted_r2"]:
            values = results[model_name][metric_name]
            final_results[model_name][metric_name] = (np.mean(values), np.std(values))
        final_results[model_name]["train_time"] = (np.mean(train_times[model_name]), np.std(train_times[model_name]))
        final_results[model_name]["test_time"]  = (np.mean(test_times[model_name]),  np.std(test_times[model_name]))

    return final_results


# Nomes de exibição para cada modelo
_NOME = {
    "knn_euclidean":          "K-vizinhos (Dist. Euclidiana)",
    "knn_manhattan":          "K-vizinhos (Dist. Manhattan)",
    "naive_bayes_univariado":  "Bayesiano (Univariado)",
    "naive_bayes_multivariado":"Bayesiano (Multivariado)",
    "regressao_linear":        "Regressão Linear Múltipla",
}


def _fmt(mean, std):
    return f"{mean:.2f} ± {std:.2f}"


def _tabela(titulo, col_modelo, headers, rows):
    CW = 16
    NW = max(len(col_modelo), max(len(r[0]) for r in rows)) + 2

    total = NW + CW * len(headers)
    h_line = "─" * total

    print(f"\n{titulo:^{total}}")
    print(h_line)
    header_str = f"{col_modelo:<{NW}}" + "".join(f"{h:^{CW}}" for h in headers)
    print(header_str)
    print(h_line)

    for r in rows:
        print(f"{r[0]:<{NW}}" + "".join(f"{v:^{CW}}" for v in r[1:]))

    print(h_line)


def print_tabela_classificacao(results):
    headers = ["Acurácia", "Precisão", "Recall", "F1-Score", "Tempo Treino (s)", "Tempo Teste (s)"]
    rows = []
    for name, m in results.items():
        rows.append([
            _NOME.get(name, name),
            _fmt(*m["accuracy"]),
            _fmt(*m["precision"]),
            _fmt(*m["recall"]),
            _fmt(*m["f1"]),
            _fmt(*m["train_time"]),
            _fmt(*m["test_time"]),
        ])
    _tabela(
        "Tabela comparativa dos classificadores, com médias e desvios padrão (±)",
        "Classificador",
        headers,
        rows,
    )


def print_tabela_regressao(results):
    headers = ["R²", "R² Ajustado", "Tempo Treino (s)", "Tempo Teste (s)"]
    rows = []
    for name, m in results.items():
        rows.append([
            _NOME.get(name, name),
            _fmt(*m["r2"]),
            _fmt(*m["adjusted_r2"]),
            _fmt(*m["train_time"]),
            _fmt(*m["test_time"]),
        ])
    _tabela(
        "Tabela comparativa dos regressores, com médias e desvios padrão (±)",
        "Regressor",
        headers,
        rows,
    )


def main():
    DATASET_CLASSIFICACAO = Path("data/phplE7q6h.arff")  # EEG Eye State
    TARGET_CLASSIFICACAO  = "Class"

    DATASET_REGRESSAO     = Path("data/dataset.arff")    # Bike Sharing Demand
    TARGET_REGRESSAO      = "count"
    EXCLUIR_REGRESSAO     = ["casual", "registered"]

    K_FOLDS               = 4

    if not DATASET_CLASSIFICACAO.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_CLASSIFICACAO}")
    if not DATASET_REGRESSAO.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_REGRESSAO}")

    print(f"Folds: {K_FOLDS}\n")

    # Classificação
    print(">>> Carregando dados para classificação...")
    X_c, y_c, features_c = load_arff(str(DATASET_CLASSIFICACAO), target_column=TARGET_CLASSIFICACAO)
    print(f"    Dataset:  {DATASET_CLASSIFICACAO}")
    print(f"    Target:   {TARGET_CLASSIFICACAO}")
    print(f"    Amostras: {X_c.shape[0]}  |  Features: {X_c.shape[1]}")
    print(f"    Features: {features_c}")
    print(">>> Rodando classificação com validação cruzada...")
    resultados_class = run_classification_cv(X_c, y_c, k_folds=K_FOLDS)
    print_tabela_classificacao(resultados_class)

    print()

    # Regressão
    print(">>> Carregando dados para regressão...")
    X_r, y_r, features_r = load_arff(str(DATASET_REGRESSAO), target_column=TARGET_REGRESSAO, exclude_columns=EXCLUIR_REGRESSAO)
    print(f"    Dataset:  {DATASET_REGRESSAO}")
    print(f"    Target:   {TARGET_REGRESSAO}")
    print(f"    Amostras: {X_r.shape[0]}  |  Features: {X_r.shape[1]}")
    print(f"    Features: {features_r}")
    print(">>> Rodando regressão com validação cruzada...")
    resultados_reg = run_regression_cv(X_r, y_r, k_folds=K_FOLDS)
    print_tabela_regressao(resultados_reg)


if __name__ == "__main__":
    main()
