import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.optimizers import Adam


def entrenarRegresionLineal(nombreDataset):
    dataset = pd.read_csv(nombreDataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X = transformarDataset(X,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def transformarDataset(x,posicionesDataset):
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),posicionesDataset)], remainder='passthrough')
    x = ct.fit_transform(x)
    return x

def entrenarArbolDecision(nombreDataset, posVariablesIndependientes):
    dataset = pd.read_csv(nombreDataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X = transformarDataset(X, posVariablesIndependientes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    return dtree


def predecirResultadoML(model, datosApredecir):
    resultado = pd.read_csv(datosApredecir, header=None).iloc[:, :].values
    prediccion = model.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        puntoVariacion.append(prediccion[indice])
    return resultado



def guardarPredicciones(dataset, nombreArchivo):
    dataset = sorted(dataset, key=lambda x: x[10])
    with open(nombreArchivo, 'w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerows(dataset)

def aprendizajeSemiAutomatizado(datasetEtiquetado, datasetNoEtiquetado):
    etiquetados = pd.read_csv(datasetEtiquetado)
    noEtiquetados = pd.read_csv(datasetNoEtiquetado).iloc[:, :].values
    X_etiquetado = etiquetados.iloc[:, :-1].values
    y_etiquetado = etiquetados.iloc[:, -1].values
    X_etiquetado = transformarDataset(X_etiquetado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_noetiquetado = transformarDataset(noEtiquetados, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train_labeled, X_test_labeled, y_train_labeled, y_test_labeled = train_test_split(X_etiquetado, y_etiquetado,test_size=1, random_state=42)
    model = LabelPropagation()
    model.fit(X=np.vstack((X_train_labeled, X_noetiquetado)), y=np.concatenate((y_train_labeled, [-1] * len(X_noetiquetado))))
    return model

def entrenarRegresionLinealRegularizadaLasso(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0,1,2,3,4,5,6, 7, 8, 9])
    lasso = LassoCV(cv=5, random_state=0)
    # entrena el modelo con los datos de entrenamiento
    lasso.fit(X,y)
    return lasso

def entrenarArbolesAleatorios(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    rf = RandomForestClassifier(n_estimators=40, random_state=0)
    rf.fit(X_train, y_train)
    return rf

def entrenarArbolesAleatoriosRegresion(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    rf = RandomForestRegressor(n_estimators=40, random_state=0)
    rf.fit(X_train, y_train)
    return rf

def entrenarArbolesAleatoriosRegresion(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    rf = RandomForestRegressor(n_estimators=40, random_state=0)
    rf.fit(X_train, y_train)
    return rf


def entrenarNaiveBayes(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb



def entrenarRedesNeuronales(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)

    def custom_activation(x):
        return K.relu(x, alpha=0.0, max_value=None)
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation=custom_activation))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, epochs=650, batch_size=10)
    return model

def predecirResultadoRedesNeuronales(model, datosApredecir):
    resultado = pd.read_csv(datosApredecir, header=None).iloc[:, :].values
    prediccion = model.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        puntoVariacion.append(prediccion[indice][0])
    return resultado


def entrenarKvecinos(dataset):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn


def arbolesAleatoriosInverso(dataset, dato):
    data = pd.read_csv(dataset)
    #X = data.iloc[:, :-1]  # Variables independientes
    y = data.iloc[:, -1].values  # Variable dependiente
    y = y.reshape(-1, 1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X = data.iloc[:, :-1].apply(lambda x: ",".join(x), axis=1)
    rf.fit(y, X)
    dato = np.array([dato]).reshape(-1, 1)
    return rf.predict(dato).tolist()[0].split(",")


def obtenerJSONPrediccion(data):
    reconfiguracion = {}
    for caracteristica in data:
        estado =False
        if " activada" in caracteristica:
            estado = True
            caracteristica = caracteristica.replace(" activada", "")
        else:
            caracteristica = caracteristica.replace(" desactivada", "")
        print(caracteristica)
        reconfiguracion.update({caracteristica : estado})
    return reconfiguracion


def autoencoder():
    # Definir los parámetros de la red
    input_dim = 10  # Número de características de entrada
    hidden_dim = 256  # Número de neuronas en la capa oculta
    latent_dim = 2  # Número de dimensiones en la representación latente

    # Definir la estructura de la red
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(hidden_dim, activation='relu')(inputs)
    z = tf.keras.layers.Dense(latent_dim, activation='linear')(encoder)
    decoder = tf.keras.layers.Dense(hidden_dim, activation='relu')(z)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(decoder)

    # Definir el modelo
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compilar el modelo
    autoencoder.compile(optimizer='adam', loss='mse')

    # Cargar los datos de entrenamiento etiquetados
    train_labeled_data = pd.read_csv('data/dataset.csv')
    x_train_labeled = train_labeled_data.iloc[:, :-1].values
    x_train_labeled = transformarDataset(x_train_labeled, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_train_labeled = train_labeled_data.iloc[:, -1].values
    y_train_labeled = np.reshape(y_train_labeled, (-1, 1))

    # Cargar los datos de entrenamiento no etiquetados
    train_unlabeled_data = pd.read_csv('data/datos.csv')
    x_train_unlabeled = train_unlabeled_data.iloc[:, :].values
    x_train_unlabeled = transformarDataset(x_train_unlabeled, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Cargar los datos de prueba etiquetados
    test_data = pd.read_csv('data/datos_evaluacionmodelo.csv')
    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Entrenar el modelo
    autoencoder.fit(x=np.concatenate((x_train_labeled, x_train_unlabeled), axis=0),
                    y=np.concatenate((y_train_labeled, np.zeros((x_train_unlabeled.shape[0], 1))), axis=0), batch_size=32, epochs=10,
                    validation_data=(x_test, y_test))
    return autoencoder

def entrenarRedNeuronalConvolucional():
    data = pd.read_csv('data/dataset.csv')
    X = data.iloc[:,:-1].values  # características
    y = data.iloc[:,-1 ].values  # objetivo
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    input_shape = X_train_cnn.shape[1:]

    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(X_test_cnn, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")