import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics


#etapa carregar dados
def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        #carregamento dos dados
        if os.path.exists(caminho_arquivo):

            df= pd.read_csv(caminho_arquivo, encoding="latin1", sep=',')

            print(" o arquivo foi carregado com sucesso")

            return df
        else:
            print("o arquivo nao foi encontrado dentro da pasta!")

            return None
    except Exception as e:
        print("erro inesperado ao carregar o arquivo:", e)

        return None
    #---- chamar a funçao para armazenar o resultado---

dados = carregar_dados()
print(dados)