import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

class_names = ['Nothing', 'One pair', 'Two pairs', 'Three of a kind', 'Straight',
               'Flush', 'Full house', 'Four of a kind', 'Straight flush', 'Royal flush']

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ----------------------------------------------------------------
# Leer Datos de entrenamiento y testeo:
# ----------------------------------------------------------------

data_train = pd.read_csv(filepath_or_buffer="poker-hand-training-true.data", sep=',', header=None)
data_test = pd.read_csv(filepath_or_buffer="poker-hand-testing.data", sep=',', header=None)
# ----------------------------------------------------------------
# printear el número de datos de entrenamiendo y te testeo
# ----------------------------------------------------------------

print(f"Datos de entrenamiento: {data_train.shape}")
print(f"Datos de testeo: {data_test.shape}")

# ----------------------------------------------------------------
# Preparar los datos para el entrenamiento y el testeo:
# ----------------------------------------------------------------

# Preparar los datos de entreno:
array_train = data_train.values
data_train = array_train[:, 0:10]
label_train = array_train[:, 10]
# Preparar los datos de testeo
array_test = data_test.values
data_test = array_test[:, 0:10]
label_test = array_test[:, 10]

# ----------------------------------------------------------------
# Escalamos los datos para nuestro modelo principal:
# ----------------------------------------------------------------

# Escalamos los datos para hacer que la red neuronal converja más fácilmente y mejore el rendimiento del modelo:
scaler = StandardScaler()
# Entrenamos sólo los datos de entrenamiento:
scaler.fit(data_train)
# Transformamos los datos de entreno y testeo (Esto asegura que ambos conjuntos de datos tengan la misma escala):
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# ----------------------------------------------------------------
# Aplicamos el clasificador MLP:
# ----------------------------------------------------------------

# Inicialización MLP
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 64),
                    activation='tanh', learning_rate_init=0.02, max_iter=2000,
                    random_state=1, early_stopping=True)
# Lo entrenamos
result = clf.fit(data_train, label_train)
loss_clf = result.loss_curve_
accuracy_values_clf = result.validation_scores_

# mostramos las gráficas de pérdida y precisión durante el entrenamiento:
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_clf)
plt.title('Pérdida durante el entrenamiento del MLP')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')

plt.subplot(1, 2, 2)
plt.plot(accuracy_values_clf)
plt.title('Precisión durante el entrenamiento del MLP')
plt.xlabel('Iteraciones')
plt.ylabel('Precisión')

plt.show()

# Predecimos
prediction = clf.predict(data_test)

# Obtenemos la precisión del testeo:
acc = accuracy_score(label_test, prediction)
print(classification_report(label_test, prediction, target_names=class_names))

# mostramos la matriz de confusión:
print(confusion_matrix(label_test, prediction))
plot_confusion_matrix(label_test, prediction, 'Confusion Matrix - MLP Classifier')

print("Precisión usando el clasificador MLP: ", str(acc))

# ----------------------------------------------------------------
# Inicializamos los modelos SVM para comparar:
# ----------------------------------------------------------------

models = [svm.SVC(kernel='linear', C=1), OutputCodeClassifier(BaggingClassifier()),
          OneVsRestClassifier(svm.SVC(kernel='linear'))]

model_names = ["Linear SVM", "OutputCodeClassifier with RBF SVM", "OneVsRestClassifier with Linear SVM"]

# ----------------------------------------------------------------
# Lanzamos cada modelo:
# ----------------------------------------------------------------

# Definir los tamaños de los subconjuntos de entrenamiento en los que evaluaré el modelo
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]


for model, name in zip(models, model_names):
    result = model.fit(data_train, label_train)
    prediction = model.predict(data_test)
    # Imprimimos la precisión
    acc = accuracy_score(label_test, prediction)
    print(confusion_matrix(label_test, prediction))
    plot_confusion_matrix(label_test, prediction, f'Confusion Matrix - {name}')
    print("Accuracy Using", name, ": " + str(acc) + '\n')
    # Calcular la curva de aprendizaje para el modelo actual
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, data_train, label_train, train_sizes=train_sizes, cv=5, scoring='accuracy')
    # Calcular la precisión promedio en cada tamaño de conjunto de entrenamiento
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label=name + ' (train)', linestyle='--')
    plt.plot(train_sizes, test_scores_mean, label=name + ' (test)')
    plt.xlabel('Tamaño del Conjunto de Entrenamiento')
    plt.ylabel('Precisión')
    plt.title(name)
    plt.legend()
    plt.grid(True)

    plt.show()

