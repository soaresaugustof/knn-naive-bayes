# Universidade de Fortaleza - UNIFOR
**Centro de Ciências Tecnológicas - CCT**  
**Disciplina:** Inteligência Artificial Computacional  
**Professora:** Dra. Cynthia Moreira Maia

---

**OBS: Projeto Individual ou em Dupla**

**Dia da apresentação:** 05/05/2026  
**Envio dos slides até:** 04/05/2026

---

## 1 Problema

O objetivo deste trabalho é selecionar um dataset de classificação e Regressão (um para cada tarefa) disponível no *OpenML* (https://www.openml.org/) e realizar a avaliação de diferentes algoritmos de aprendizado de máquina supervisionado.

O banco de dados escolhido deve atender aos seguintes critérios:

- Conter mais de 10 atributos (variáveis preditoras);
- Ter mais de 1000 instâncias (amostras).

Cada um deverá informar antecipadamente, por meio de formulário (https://forms.gle/gqLHCnVFUbsGRSuf6), qual conjunto de dados será utilizado. Os classificadores a serem avaliados incluem os seguintes algoritmos:

**(a) K-Nearest Neighbors (kNN):** treine o classificador utilizando diferentes medidas de distância;

**1. Distância Euclidiana**

$$d_{\text{Euclidiana}}(X, Y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**2. Distância Manhattan**

$$d_{\text{Manhattan}}(X, Y) = \sum_{i=1}^{n}|x_i - y_i|$$

**(b)** Treine um classificador bayesiano - **Caso Univariado**. A função de densidade da distribuição normal univariada é dada por:

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left[-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right]$$

**(c)** Treine um classificador bayesiano - **Caso Multivariado**. A função de densidade normal multi-variada em $d$ dimensões é dada por:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}}} \exp\left[-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^t \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right]$$

Matriz de covariância:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \sigma_{13} \\ \sigma_{21} & \sigma_2^2 & \sigma_{23} \\ \sigma_{31} & \sigma_{32} & \sigma_3^2 \end{bmatrix}$$

- $\mathbf{x}$ é um vetor coluna com $d$ componentes
- $\boldsymbol{\mu}$ é o vetor de médias de $d$ componentes
- $\Sigma$ é a matriz de covariâncias $d \times d$
- $|\Sigma|$ é o determinante de $\Sigma$
- $\Sigma^{-1}$ é a inversa de $\Sigma$
- $(\mathbf{x} - \boldsymbol{\mu})^t$ é a transposta de $(\mathbf{x} - \boldsymbol{\mu})$

> **OBS:** No Python esta matriz pode ser encontrada ao utilizar a biblioteca numpy e aplicar `np.cov`.

**(d)** Calcule a média e o desvio padrão das métricas de avaliação, considerando todos os *folds* da validação cruzada. Em seguida, apresente os resultados em uma tabela comparativa, conforme o modelo abaixo.

### Exemplo de Tabela com Médias, Desvio Padrão e Tempos

| Classificador | Acurácia | Precisão | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|
| K-vizinhos (Dist. Euclidiana) | 0.85 ± 0.03 | 0.84 ± 0.04 | 0.84 ± 0.04 | 0.12 ± 0.01 | 0.03 ± 0.00 |
| K-vizinhos (Dist. Manhattan) | 0.88 ± 0.02 | 0.87 ± 0.03 | 0.87 ± 0.03 | 0.11 ± 0.01 | 0.02 ± 0.00 |
| Bayesiano (Univariado) | 0.87 ± 0.03 | 0.86 ± 0.03 | 0.86 ± 0.03 | 0.07 ± 0.01 | 0.01 ± 0.00 |

*Tabela 1: Análise comparativa do desempenho dos classificadores, com médias, desvios padrão (±) e tempos de treino e teste.*

**(e)** Os regressores a serem avaliados incluem os seguintes algoritmos:

- K-Nearest Neighbors (kNN): Distância Euclidiana e Manhattan;
- Regressão Linear Múltipla;

A comparação de desempenho será feita por meio das métricas:

- *Accuracy*
- *F1-score*
- *Precision*
- *Recall*
- *R2 score*
- *R2 score ajustado*

---

## 2 Estrutura Mínima do Trabalho

O projeto deve conter, no mínimo, as seguintes seções:

1. **Introdução**
2. **Algoritmos de Aprendizagem de Máquina**
3. **Experimentos**
   - 3.1. Banco de Dados
   - 3.2. Métricas de Avaliação
   - 3.3. Resultados
4. **Conclusões**
5. **Referências**

### Análise Comparativa dos Classificadores e Regressores

Com base na Tabela 1, elabore uma análise crítica considerando:

- Comparação das métricas entre os classificadores (Acurácia e F1-Score);
- Comparação das métricas entre os regressores (Acurácia e F1-Score);
- Diferenças nos tempos de treino e teste;
- Relação entre desempenho e eficiência computacional;
- Pontos fortes e limitações observadas em cada algoritmo.

Apresente, ao final, qual classificador e regressor apresentou melhor equilíbrio entre desempenho e tempo de execução para o problema analisado.

---

## Observações Importantes

- A entrega das implementações é **obrigatória**. Trabalhos sem código entregue receberão **nota zero**.
- A pontualidade será um critério avaliativo.
- Não é permitido o uso de bibliotecas que implementem diretamente os algoritmos (por exemplo, `scikit-learn`) nem o uso de `pandas`. Todo o código deve ser implementado manualmente.
- Os slides da apresentação deverão ser enviados no AVA dentro do prazo estabelecido. Devem conter:
  - Introdução e justificativa do dataset escolhido;
  - Explicação do código e funcionamento dos algoritmos;
  - Resultados e comparações;
  - Conclusões gerais.
- Não é necessário entregar um relatório escrito além dos slides.
- Caso algum aluno não consiga apresentar, deverá agendar uma avaliação individual. Caso contrário, será atribuída **nota zero**.