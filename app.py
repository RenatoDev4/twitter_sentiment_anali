import streamlit as st
from PIL import Image

from clean_text import classify_msg

st.set_page_config(page_title="Análise de Frases para Prevenção de Suicídios", page_icon="🌼")


image_main = Image.open("imgs/yellow_set_main.png")
st.sidebar.image(image_main)

st.sidebar.markdown("**Projeto de Análise de Sentimentos para Prevenção de Suicídios**")
nome = "Renato Moraes"
titulo = "Cientista de Dados"
linkedin_url = 'https://linkedin.com/in/renato-moraes-11b546272'
github_url = 'https://github.com/RenatoDev4'


st.sidebar.markdown("**Sobre o Autor:**")
st.sidebar.text(f'Nome: {nome}')
st.sidebar.text(f'Cargo: {titulo}')
st.sidebar.markdown(f'**[LinkedIn]({linkedin_url})** | **[GitHub]({github_url})**')

st.sidebar.markdown("----")
st.sidebar.markdown("Contate o **Centro de Valorização da Vida**:")
st.sidebar.markdown("Disponível 24 horas por telefone")
st.sidebar.markdown("Ligue **188**")

st.sidebar.markdown("----")
st.sidebar.markdown("""Este projeto representa apenas uma pequena parte das possibilidades de um modelo de machine learning no auxílio às pessoas. Com o acesso a mais dados, integrações com APIs e recebimento de dados em tempo real, o potencial e a variedade de serviços disponíveis aumentam significativamente.""")

st.header("Projeto de conscientização e prevenção de suicídios")

st.markdown(
    """
    Este modelo de machine learning foi treinado para detectar potenciais tendências suicidas em frases, usando uma base de dados de textos publicados no Twitter. 
    A sua missão é contribuir ativamente para a conscientização da importância da vida e a prevenção do suicídio.
    """)

image = Image.open("imgs/yellow_set.png")
st.image(image, caption="Yellow Setember")


input_msg = st.text_input("Escreva uma mensagem para detecção de sentimento:", "O modelo foi treinado com textos em inglês, portanto deve-se escrever mensagens em inglês.")

if st.button("Classificar"):
    if input_msg:
        prediction = classify_msg(input_msg)
        if prediction == 0:
            st.success("Esta NÃO é uma mensagem com tendências suicidas ✅")
        else:
            st.error("Esta é uma mensagem com tendências suicidas 🚨")
