import streamlit as st
import pandas as pd
import joblib
import os

#nome_usuario = st.text_input("informe o seu nome", placeholder="digite aqui...")

#if st.button(label="clique aqui:"):
 #   st.success(" seja bem vindo, " + nome_usuario)

 #etapa01 - definiçao das features
FEATURES_NAMES = [
    'Nota_P1',
    'Nota_P2',
    'Media_Trabalhos',
    'Frequencia',
    'Reprovacoes_Anteriores',
    'Acessos_Plataforma_Mes'

]

COLUNAS_HISTORICO = FEATURES_NAMES + ["Previsao_Resultado", 'Prob_Aprovado', 'Prob_Reprovado']


# criar uma sessão st.session
if 'historico_previsoes' not in st.session_state:
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)

#ETAPA02 - CARREGAMENTO DO MODELO PARA O NOSSO FRONT-END
#ST.cache_resource para carregar o modelo apenas uma vez
#otimizando o desempenho do app
@st.cache_resource
def carregar_modelo(caminho_modelo = "modelo_previsao_desempenho.joblib"):
    """
      Carregar o pipeline de ML treinando (scaler + modelo) do arquivo . joblib
    """
    try:
      if os.path.exists(caminho_modelo):
        modelo = joblib.load(caminho_modelo)
        return modelo
      else:
         st.error(f"erro: arquivo do modelo '{caminho_modelo}' nao foi encontrado!")
         st.warning(f"por favor, execute o script 'modelo_treinamento.py' para gerar o modelo")
    except Exception as e:
        st.error(f"erro inesperado ao carregar o modelo:{e}")
    return None

pipeline_modelo = carregar_modelo()


#etapa02: configuração da interface do usuario(streamlit)

st.set_page_config(layout='wide', page_title="previsão de notas")
st.title("🤖previsor de desempenho academico")
st.markdown("""
    essa ferramenta usa inteligencia artificial para prever o status final (aprovado ou reprovado)
    de um aluno com base em seu desempenho parcial.
            
    **preencha os dados do aluno abaixo para obter uma previsao: **
    """)

#etapa03 - FORMULARIO DE ENTRADA

if pipeline_modelo is not None:
    #utilizar o formulario para agrupar as entradas e o botão
    with st.form("formulario_previsao"):
        st.subheader(" insira as notas e metricas do aluno")

        col1, col2 = st.columns(2)

        with col1:
            nota_p1 = st.slider(" nota da p1(0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            media_trabalhos = st.slider("media dos dos trabalhos(0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            reprovacoes_ant = st.number_input(" reprovaçoes anteriores", min_value=0, max_value=10, value=0, step=1)
        with col2:
            nota_p2 = st.slider("nota da p2(0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            frequencia = st.slider("frequencia(%)", min_value=0, max_value=100, value=75, step=5)
            acesso_mes = st.number_input(" media de acessos a plataforma (por mes)", min_value=0, max_value=100, value=10, step=1)

        submitted = st.form_submit_button("realizar previsao")

    if submitted:
        Features_name = [
         'ï»¿Nota_P1',
         'Nota_P2',
         'Media_Trabalhos',
         'Frequencia',
         'Reprovacoes_Anteriores',
         'Acessos_Plataforma_Mes'   
        ]

        #criação de um dataframe a partir dos dados inseridos
        dados_alunos = pd.DataFrame(
            [[nota_p1, nota_p2, media_trabalhos, frequencia, reprovacoes_ant, acesso_mes]],
            columns = Features_name
        )

        st.info(" Processando dados e realizando a previsão...")
        try:
                #realizar a previsão ([0] ou [1])
            previsao = pipeline_modelo.predict(dados_alunos)
                
                #obter a probabilidade
            probabilidade = pipeline_modelo.predict_proba(dados_alunos)

            prob_reprovados = probabilidade[0][0]
            prob_aprovados = probabilidade[0][1]
            resultado_texto = "APROVADO" if previsao[0] == 1 else "REPROVADO"
                
                #EXIBIR OS RESULTADOS NA TELA

            st.subheader("Resultado da previsão")

            if previsao [0] == 1:
                st.success("Previsão: Aprovado")
                st.markdown(f"""
                                Com base nos doados fornecidos, o modelo prevê que o aluno tem:**{prob_aprovados*100:.2f}%** de chance de ser **aprovado**
                                
                                *Chance de aprovação: {prob_aprovados*100:.2f}%)*
                                """)
            else:
                st.success("Previsão: Reprovado (zona de risco)")
                st.markdown(f"""
                                Com base nos doados fornecidos, o modelo prevê que o aluno tem:**{prob_aprovados*100:.2f}%** de chance de ser **reprovado**
                                
                                *Chance de reprovado: {prob_aprovados*100:.2f}%)*
                                """)
                nova_linha_dict = {
                    'Nota_P1': nota_p1,
                    'Nota_P2': nota_p2,
                    'Media_Trabalhos': media_trabalhos,
                    'Frequencia': frequencia,
                    'Reprocoes_Anteriores': reprovacoes_ant,
                    'Acessos_Resultado': resultado_texto,
                    'Prob_Aprovado': prob_aprovados,
                    'Prob_Reprovado': prob_reprovados
                }

                nova_linha_df = pd.DataFrame([nova_linha_dict], columns=COLUNAS_HISTORICO)

                st.session_state.historico_previsoes = pd.concat(
                    [st.session_state.historico_previsoes, nova_linha_df],
                    ignore_index=True
                )
        except Exception as e:
            st.error(f" erro ao fazer a revisão: {e}")
            st.error("verifique se os nomes das colunas correspondem aos nomes das colunas utilizados no treino")
    st.subheader(" Historico de previsões realizadas ma sessão:")
    if st.session_state.historico_previsoes.empty:
        st.write("nenhuma previsão foi realizada ainda")
    else:
        st.dataframe(st.session_state.historico_previsoes, use_container_width=True)

        if st.button(" Limpar historico "):
            st.session_state.historico_previsoes = pd.DataFrame(columns = COLUNAS_HISTORICO)
            st.rerun()
else:
    st.warning(" o aplicativo  nao pode fazer previsões porque o modelo nao foi carregado")