# Documentação do Projeto: Teste Inicial com MLflow e Naive Bayes

## Visão Geral
Este projeto foi desenvolvido como parte de um estudo para experimentar o uso do MLflow em pipeline de aprendizado de máquina. Ele registra as etapas de treino, avaliação e registro de um modelo Naive Bayes utilizando o MLflow.

## Objetivo
Registrar um modelo Naive Bayes e suas métricas de desempenho em um experimento do MLflow.

## Estrutura do Código
### 1. Configuração do Experimento
```python
mlflow.set_experiment('naive_bayes_exp')
```
Definimos um experimento no MLflow chamado `naive_bayes_exp`.

### 2. Início do Experimento
Dentro do contexto do experimento, realizamos as seguintes ações:

### 3. Treinamento do Modelo
O modelo Naive Bayes foi treinado usando os conjuntos de dados `X_train` e `y_train`.
```python
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
previsoes = naive_bayes.predict(X_test)
```

### 4. Métricas de Avaliação
As seguintes métricas foram calculadas para avaliar o desempenho do modelo:
- **Acurácia**
- **Recall**
- **Precisão**
- **F1-Score**
- **AUC (Area Under Curve)**
- **Log Loss**

Código:
```python
acuracia = accuracy_score(y_test, previsoes)
recall = recall_score(y_test, previsoes)
precisao = precision_score(y_test, previsoes)
f1 = f1_score(y_test, previsoes)
auc = roc_auc_score(y_test, previsoes)
log = log_loss(y_test, previsoes)
```
As métricas foram registradas no MLflow:
```python
mlflow.log_metric('acuracia', acuracia)
mlflow.log_metrics({'recall': recall, 'precisao': precisao, 'f1': f1, 'auc': auc, 'log': log})
```

### 5. Gráficos de Desempenho
Foram gerados dois gráficos para visualizar o desempenho do modelo:
- **Matriz de Confusão**
- **Curva ROC**

Os gráficos foram salvos como arquivos PNG:
```python
confusion = ConfusionMatrixDisplay.from_estimator(naive_bayes, X_test, y_test)
plt.savefig('confusion.png')
roc = RocCurveDisplay.from_estimator(naive_bayes, X_test, y_test)  
plt.savefig('roc.png')
plt.close()
```
E, em seguida, registrados como artefatos no MLflow:
```python
mlflow.log_artifact('confusion.png')
mlflow.log_artifact('roc.png')
```

### 6. Informações Adicionais
Tags adicionais foram configuradas para descrever o modelo:
```python
mlflow.set_tag('modelo', 'naive_bayes')
mlflow.set_tag('dataset', 'credit')
```

### 7. Registro do Modelo
O modelo treinado foi registrado no MLflow:
```python
mlflow.sklearn.log_model(naive_bayes, 'naive_bayes')
```

### 8. Finalização do Experimento
O experimento foi finalizado:
```python
mlflow.end_run()
```

## Resultados
Os resultados do experimento incluem:
- Métricas registradas no MLflow.
- Artefatos gerados (gráficos de matriz de confusão e curva ROC).
- Modelo Naive Bayes treinado e registrado.

## Como Executar
1. Certifique-se de que as dependências do projeto estejam instaladas:
   ```bash
   pip install mlflow scikit-learn matplotlib
   ```

2. Configure o MLflow para apontar para o servidor ou utilizar localmente.

3. Execute o script em um ambiente Python configurado.

## Pontos de Aprendizado
Este projeto introduz os seguintes conceitos:
- Registro de métricas e artefatos usando o MLflow.
- Visualização de gráficos para avaliação de modelos.
- Gerenciamento de experimentos de aprendizado de máquina.

