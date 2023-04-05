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
from sklearn.neighbors import KNeighborsClassifier
from keras.optimizers import Adam


def regresionLineal(nombreDataset, posVariablesIndependientes):
    dataset = pd.read_csv(nombreDataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X = transformarDataset(X,posVariablesIndependientes)
    regresionLineal = entrenarModeloRegresionLineal(X,y)
    return regresionLineal

def transformarDataset(x,posicionesDataset):
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),posicionesDataset)], remainder='passthrough')
    x = ct.fit_transform(x)
    print(x)
    return x
def entrenarModeloRegresionLineal(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def calcularPrediccionRegresionLineal(regressor, X):
    return regressor.predict(X)

def predecirResultadoRegresionLineal(regressor,nombreDataset):
    df = pd.read_csv(nombreDataset)
    #dataset = df.values
    dataset = df.iloc[:, :].values
    #print(dataset)
    data = transformarDataset(dataset,[0,1,2,3,4,5,6,7,8,9])
    prediccion = calcularPrediccionRegresionLineal(regressor, data)
    dataset = dataset.tolist()
    for indice, puntoVariacion in enumerate(dataset):
        #print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        #print(puntoVariacion)
        #puntoVariacion.add(prediccion)
    print(dataset)
    return dataset

def arbolDecision(nombreDataset, posVariablesIndependientes):
    dataset = pd.read_csv(nombreDataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X = transformarDataset(X, posVariablesIndependientes)
    dtree = entrenarModeloArbolDecision(X,y)
    return dtree

def predecirResultadoArbolDecision(dtree, nombreDataset):
    df = pd.read_csv(nombreDataset)
    #dataset = df.values
    dataset = df.iloc[:, :].values
    data = transformarDataset(dataset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    prediccion = calcularPrediccionArbolDecision(dtree, data)
    dataset = dataset.tolist()
    for indice, puntoVariacion in enumerate(dataset):
        #print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        #print(puntoVariacion)
        #puntoVariacion.add(prediccion)
    #print(dataset)
    return dataset

def calcularPrediccionArbolDecision(dtree, X):
    return dtree.predict(X)

def entrenarModeloArbolDecision(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

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
    prediccion = model.predict(X_noetiquetado)
    noEtiquetados = noEtiquetados.tolist()
    for indice, puntoVariacion in enumerate(noEtiquetados):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return noEtiquetados

def regresionLinealRegularizadaLasso(dataset, datosApredecir):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0,1,2,3,4,5,6, 7, 8, 9])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)

    # crea un modelo de regresión lineal regularizada con Lasso
    #lasso = Lasso(alpha=0.1)
    lasso = LassoCV(cv=5, random_state=0)
    # entrena el modelo con los datos de entrenamiento
    lasso.fit(X,y)
    #lasso.fit(X_train, y_train)
    resultado = pd.read_csv(datosApredecir).iloc[:, :].values
    prediccion = lasso.predict(transformarDataset(resultado,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return resultado

def arbolesAleatorios(dataset, datosApredecir):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    rf = RandomForestClassifier(n_estimators=40, random_state=0)
    rf.fit(X_train, y_train)
    resultado = pd.read_csv(datosApredecir).iloc[:, :].values
    prediccion = rf.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return resultado

def arbolesAleatoriosRegresion(dataset, datosApredecir):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    rf = RandomForestRegressor(n_estimators=40, random_state=0)
    rf.fit(X_train, y_train)
    resultado = pd.read_csv(datosApredecir).iloc[:, :].values
    prediccion = rf.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return resultado

def naiveBayes(dataset, datosApredecir):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    resultado = pd.read_csv(datosApredecir).iloc[:, :].values
    prediccion = gnb.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return resultado

def redesNeuronales(dataset, datosApredecir):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    model = Sequential()
    #model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(1, input_dim=X.shape[1], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, epochs=50, batch_size=10)
    resultado = pd.read_csv(datosApredecir, header=None).iloc[:, :].values
    prediccion = model.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice][0])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return resultado

def kvecinos(dataset, datosApredecir):
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = transformarDataset(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    resultado = pd.read_csv(datosApredecir).iloc[:, :].values
    prediccion = knn.predict(transformarDataset(resultado, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    resultado = resultado.tolist()
    for indice, puntoVariacion in enumerate(resultado):
        # print(prediccion[indice])
        puntoVariacion.append(prediccion[indice])
        # print(puntoVariacion)
        # puntoVariacion.add(prediccion)
    # print(dataset)
    return resultado

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
        estado =""
        if " activada" in caracteristica:
            estado = "activada"
            caracteristica = caracteristica.replace(" activada", "")
        else:
            estado = "desactivada"
            caracteristica = caracteristica.replace(" desactivada", "")
        print(caracteristica)
        reconfiguracion.update({caracteristica : estado})
    return reconfiguracion