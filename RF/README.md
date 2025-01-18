# Documentação do Projeto: Teste com Random Forest e MLflow

## Visão Geral
Este projeto utiliza MLflow para registrar experimentos com o modelo Random Forest, incluindo o registro de hiperparâmetros, métricas, gráficos e informações do modelo. A função desenvolvida permite configurar o número de estimadores (árvores na floresta) e avaliar o desempenho do modelo.

## Objetivo
Explorar o uso de Random Forest em combinação com MLflow para:
- Avaliar o impacto de diferentes valores de `n_estimators` no desempenho do modelo.
- Registrar experimentos e artefatos de forma organizada.

## Estrutura do Código

### 1. Definição do Experimento
```python
mlflow.set_experiment('Random Forest')
```
Um experimento no MLflow chamado `Random Forest` é configurado para centralizar os registros.

### 2. Configuração e Treinamento do Modelo
A função `treina_rf(n_estimators)` recebe o número de estimadores como argumento. Um modelo `RandomForestClassifier` é treinado usando os dados de teste (`X_test` e `y_test`):
```python
modelrf = RandomForestClassifier(n_estimators=n_stimators, random_state=123)
modelorf = modelrf.fit(X_test, y_test)
previsoes = modelorf.predict(X_test)
```

### 3. Registro de Hiperparâmetros
O hiperparâmetro `n_estimators` é registrado no MLflow:
```python
mlflow.log_param('n_estimators', n_stimators)
```

### 4. Registro de Métricas
As seguintes métricas são calculadas e registradas:
- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **ROC AUC**
- **Log Loss**

Código:
```python
mlflow.log_metric('accuracy', accuracy_score(y_test, previsoes))
mlflow.log_metric('precision', precision_score(y_test, previsoes))
mlflow.log_metric('recall', recall_score(y_test, previsoes))
mlflow.log_metric('f1', f1_score(y_test, previsoes))
mlflow.log_metric('roc_auc', roc_auc_score(y_test, previsoes))
mlflow.log_metric('log_loss', log_loss(y_test, previsoes))
```

### 5. Gráficos de Avaliação
Dois gráficos são gerados:
- **Matriz de Confusão**
- **Curva ROC**

Os gráficos são salvos como arquivos PNG:
```python
confusion = ConfusionMatrixDisplay.from_estimator(modelorf, X_test, y_test)
plt.savefig('confusion.png')
roc = RocCurveDisplay.from_estimator(modelorf, X_test, y_test)  
plt.savefig('roc.png')
plt.close()
```
E registrados como artefatos no MLflow:
```python
mlflow.log_artifact('confusion.png')
mlflow.log_artifact('roc.png')
```

### 6. Registro de Informações Adicionais
Tags descritivas foram adicionadas:
```python
mlflow.set_tag('modelo', 'random forest')
mlflow.set_tag('dataset', 'credit')
mlflow.set_tag('owner', 'Leandro')
```

### 7. Registro do Modelo
O modelo treinado foi registrado:
```python
mlflow.sklearn.log_model(modelorf, 'random-forest-model')
```

### 8. Finalização do Experimento
O experimento foi finalizado:
```python
mlflow.end_run()
```

## Resultados
Os resultados do experimento incluem:
- **Hiperparâmetros**: Valor de `n_estimators`.
- **Métricas**: Acurácia, Precisão, Recall, F1-Score, ROC AUC e Log Loss.
- **Artefatos**: Gráficos da Matriz de Confusão e Curva ROC.
- **Modelo**: Random Forest registrado no MLflow.

## Como Executar
1. Certifique-se de instalar as dependências necessárias:
   ```bash
   pip install mlflow scikit-learn matplotlib
   ```
2. Configure o MLflow para uso local ou em servidor.
3. Execute a função passando o valor desejado para `n_estimators`. Exemplo:
   ```python
   treina_rf(50)
   ```

## Pontos de Aprendizado
Este projeto explora:
- Registro de experimentos utilizando MLflow.
- Avaliação de hiperparâmetros e desempenho de modelos.
- Registro e gerenciamento de artefatos de gráficos.

