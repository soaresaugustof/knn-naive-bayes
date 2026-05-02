# KNN e Naive Bayes — Avaliação de Algoritmos de Aprendizado Supervisionado

Projeto desenvolvido para a disciplina **Inteligência Artificial Computacional** (UNIFOR).  
Implementa e avalia classificadores e regressores com validação cruzada k-fold, **sem uso de scikit-learn ou pandas**.

---

## Estrutura do projeto

```
knn-naive-bayes/
├── models/
│   ├── __init__.py
│   ├── knn.py                      # KNN para classificação e regressão
│   ├── naive_bayes_univariado.py   # Naive Bayes Gaussiano — Caso Univariado
│   ├── naive_bayes_multivariado.py # Naive Bayes Gaussiano — Caso Multivariado
│   └── regressao_linear.py         # Regressão Linear Múltipla (mínimos quadrados)
├── data/
│   └── dataset.arff                # Dataset: Bike Sharing Demand (OpenML)
├── main.py                         # Ponto de entrada — roda classificação e regressão
├── metrics.py                      # Métricas: acurácia, precisão, recall, F1, R², R² ajustado
├── utils.py                        # Validação cruzada k-fold
└── requirements.txt
```

---

## Algoritmos implementados

### Classificação
| Modelo | Descrição |
|---|---|
| KNN Euclidiana | K vizinhos mais próximos com distância Euclidiana |
| KNN Manhattan | K vizinhos mais próximos com distância Manhattan |
| Naive Bayes Univariado | Distribuição normal por feature (independência entre features) |
| Naive Bayes Multivariado | Distribuição normal multivariada com matriz de covariância |

### Regressão
| Modelo | Descrição |
|---|---|
| KNN Euclidiana | Média dos k vizinhos mais próximos (distância Euclidiana) |
| KNN Manhattan | Média dos k vizinhos mais próximos (distância Manhattan) |
| Regressão Linear Múltipla | Mínimos quadrados: β = (XᵀX)⁻¹Xᵀy |

---

## Dataset

**Bike Sharing Demand** — disponível no [OpenML](https://www.openml.org/)

- **Instâncias:** 17.379
- **Atributos:** 15 (season, year, month, hour, holiday, weekday, workingday, weather, temp, feel_temp, humidity, windspeed, casual, registered, count)
- **Target classificação:** `season` (estação do ano: spring, summer, fall, winter)
- **Target regressão:** `count` (total de bicicletas alugadas)

---

## Como executar

Configure os parâmetros diretamente no `main.py`:

```python
DATASET              = Path("data/dataset.arff")
TARGET_CLASSIFICACAO = "season"
TARGET_REGRESSAO     = "count"
K_FOLDS              = 4
```

Depois execute:

```bash
python main.py
```

---

## Métricas avaliadas

**Classificação:** Acurácia, Precisão, Recall, F1-Score, Tempo de Treino, Tempo de Teste

**Regressão:** R², R² Ajustado, Tempo de Treino, Tempo de Teste

Todos os valores são apresentados como **média ± desvio padrão** entre os folds.

---

## Dependências

```
numpy
```
