import numpy as np
import cv2 as cv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings


def load_dataframe():
    '''
    Carga un dataframe Pandas con las imagenes para entrenamiento de modelo
    '''
    dados = {
        "ARCHIVO": [],
        "ROTULO": [],
        "OBJETO": [],
        "IMAGEN": [],
    }

    com_mascara = os.listdir(f"imagenes{os.sep}conmascara")
    sem_mascara = os.listdir(f"imagenes{os.sep}sinmascara")

    for ARCHIVO in com_mascara:
        dados["ARCHIVO"].append(f"imagenes{os.sep}conmascara{os.sep}{ARCHIVO}")
        dados["ROTULO"].append(f"Com mascara")
        dados["OBJETO"].append(1)
        img = cv.cvtColor(cv.imread(f"imagenes{os.sep}conmascara{os.sep}{ARCHIVO}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEN"].append(img)
        
    for ARCHIVO in sem_mascara:
        dados["ARCHIVO"].append(f"imagenes{os.sep}sinmascara{os.sep}{ARCHIVO}")
        dados["ROTULO"].append(f"Sem mascara")
        dados["OBJETO"].append(0)
        img = cv.cvtColor(cv.imread(f"imagenes{os.sep}sinmascara{os.sep}{ARCHIVO}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEN"].append(img)
        
    dataframe = pd.DataFrame(dados)

    return dataframe


def train_test(dataframe):
    '''
    Divide dataframe en conjunto para prueba
    '''
    X = list(dataframe["IMAGEN"])
    y = list(dataframe["OBJETO"])

    #train_test_split(X, y, train_size=0.40, random_state=13)
    return X, y


def pca_model(X_train):
    '''
    PCA para extraccion de imagenes
    '''
    pca = PCA(n_components=30)
    pca.fit(X_train)
    
    return pca


def knn(X_train, y_train):

    warnings.filterwarnings("ignore")
    '''
    Modelo K-Nearest Neighbors
    '''
    grid_params = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
    }

    knn_model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)

    knn_model.fit(X_train, y_train)

    return knn_model

