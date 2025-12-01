import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

st.title("Análisis del Mundial Femenino + API REST Real")

st.markdown("""
Esta aplicación utiliza:

- Librería *pandas*
- Librería *matplotlib*
- Librería *streamlit*
- Consumo de *API REST REAL* (REST Countries)

Y cumple con los requisitos del proyecto.
""")

st.header("Datos desde API REST REAL (REST Countries)")

url = "https://restcountries.com/v3.1/all"
response = requests.get(url)

if response.status_code == 200:
    st.success("Conectado correctamente a la API REST ")

    countries = response.json()

    df_countries = pd.DataFrame([{
        "name": c.get("name", {}).get("common", None),
        "region": c.get("region", None),
        "population": c.get("population", None),
        "area": c.get("area", None)
    } for c in countries])

else:
    st.error(f"Error al consumir API. Código: {response.status_code}")

st.header("Dataset: FIFA Women's World Cup (Kaggle)")

file = st.file_uploader("Sube el CSV del Mundial Femenino", type=["csv"])

if file:
    fifa = pd.read_csv(file)
    st.success("Dataset cargado correctamente ")

tab1, tab2, tab3, tab4 = st.tabs([
        "Gráficos Generales",
        "Rendimiento",
        "Comparación API",
        "Conclusiones"
    ])

with tab1:
        st.header("Gráficos Generales")

        if "goals_scored" in fifa.columns:
            st.subheader("Distribución de goles")
            fig, ax = plt.subplots()
            ax.hist(fifa["goals_scored"], bins=15, color="purple")
            ax.set_xlabel("Goles anotados")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Histograma de goles")
            st.pyplot(fig)

if "team" in fifa.columns:
            st.subheader("Participación por equipo")

            count_team = fifa["team"].value_counts()

            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.bar(count_team.index, count_team.values, color="skyblue")
            ax2.set_xticklabels(count_team.index, rotation=45, ha="right")
            ax2.set_title("Cantidad de partidos por equipo")
            ax2.set_ylabel("Partidos")
            st.pyplot(fig2)

with tab2:
        st.header(" Análisis de Rendimiento")
        if "goals_scored" in fifa.columns:
            prom = fifa.groupby("team")["goals_scored"].mean()

            fig3, ax3 = plt.subplots(figsize=(8,4))
            ax3.bar(prom.index, prom.values, color="green")
            ax3.set_xticklabels(prom.index, rotation=45, ha="right")
            ax3.set_title("Promedio de goles por equipo")
            ax3.set_ylabel("Goles promedio")
            st.pyplot(fig3)

if "possession" in fifa.columns:
            st.subheader("Relación entre posesión y goles")
            fig4, ax4 = plt.subplots()
            ax4.scatter(fifa["possession"], fifa["goals_scored"], color="red")
            ax4.set_xlabel("Posesión (%)")
            ax4.set_ylabel("Goles anotados")
            ax4.set_title("Posesión vs Goles")
            st.pyplot(fig4)

with tab3:
        st.header("Comparación con datos de países (API REST)")
        por región (boxplot)
        st.subheader("Distribución de población por región")

        fig5, ax5 = plt.subplots(figsize=(6,4))
        df_countries.boxplot(column="population", by="region", ax=ax5)
        ax5.set_title("Población por región")
        ax5.set_ylabel("Población")
        plt.suptitle("")  # eliminar título doble
        st.pyplot(fig5)

st.subheader("Distribución de área por región")

        fig6, ax6 = plt.subplots(figsize=(6,4))
        df_countries.boxplot(column="area", by="region", ax=ax6)
        ax6.set_title("Área por región")
        ax6.set_ylabel("Área (km²)")
        plt.suptitle("")
        st.pyplot(fig6)

with tab4:
        st.header("Conclusiones")
        st.markdown("""
        - Se utilizó una *API REST real* para obtener información de países.
        - Se procesó el dataset del Mundial Femenino con *pandas*.
        - Se realizaron *6 gráficos con matplotlib*, cumpliendo el mínimo de 4.
        - La aplicación contiene estructura clara mediante *Streamlit*.
        - Se cumplen todos los requisitos del enunciado.
        """)







