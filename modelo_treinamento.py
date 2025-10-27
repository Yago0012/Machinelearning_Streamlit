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


#---------- etapa 02: preparação e divisão dos dados -------------
# definição de X (features) e Y(Target)

if dados is not None:
    print(f"\ntotal de registros carregados: {len(dados)}")
    print("Iniciando o pipeline e treinamento ")

    TARGET_COLUMN = "Status_Final"

    #etapa 2.1 definição das features e target
    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]
        print(f"features (x) definidas:{list(X.columns)}")
        print(f" features (y) definidas: {TARGET_COLUMN}")

    except KeyError:

        print(f"\n-------erro critico--------")
        print(f" a coluna {TARGET_COLUMN} nao foi encontrado no CSV")

        print(f" colunas disponiveis:{list(dados.columns)}")
        print(f"por favor, ajuste a variavel 'TARGET_COLUMN' E tente novamente!")
        #se o Target nao for encontrado, ira encerraar o script!
        exit()
    
    #etapa2.2 - divisão entre treino e teste
    print("\n----------Dividindo dados em treino e teste... ---------")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=0.2,    #20% dos dados seram utilizado para teste
        random_state=42, #garantir a reprodutibilidade
        stratify=y       #manter a proporção de aprovados e reprovados
    )

    print(f"dados de treino: {len(X_train)} | dados de teste: {len(X_test)}")

    # Etapa 03: CRIAÇÃO PIPELINE DE ML

    print("\n------ Criando a pipeline de ML....--------- ")
    #scaler -> normalização dos dados (colocando tudo na mesma escala)
    #medel -> aplica o modelo de regressão logistica
    pipeline_model = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    #etapa 04: treinamento e avaliaçao dos dados/modelo

    print(f"\n-------treinamento do modelo...-------")
    #treuna a pipeline com os dados de treino
    pipeline_model.fit(X_train, y_train)

    print("modelo treinado. avaliando com os dados de teste...")
    y_pred = pipeline_model.predict(X_test)

    # avaliaçao de desempenho
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    report = metrics.classification_report(y_test,y_pred)

    print("\n---------- relatorio de avaliaçao geral --------")
    print(f" acuraia geral: {accuracy * 100:.2f}%")
    print("\nrelatorio de classificaçao detalhado:")
    print(report)



    #etapa 05: salvando o modelo
    model_filename = 'modelo_previsao_desempenho.joblib'
    print(f"\n salvando o pipeline treinamento em..{model_filename}")
    joblib.dump(pipeline_model, model_filename)

    print("processo concluido com sucesso!")
    print(f" o arquivo '{model_filename}' esta para ser utilizado!")
else:
    print(" o pipeline nao pode continuar pois os dados nao foram carregados!")
    