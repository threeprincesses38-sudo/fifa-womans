import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

st.set_page_config(
    page_title="Mundial Femenino - Análisis de Datos",
    layout="wide",
)

st.title("Análisis de Datos del Mundial Femenino")
st.caption("Aplicación desarrollada para la Prueba Solemne N°3 - Taller de Programación II")
st.markdown("*Integrantes:* Aylin Mella · Luis Torres · Franciska Zúñiga")


@st.cache_data
def load_world_cup_data() -> pd.DataFrame:
    # El archivo womens-world-cup.csv debe estar en la raíz del repositorio.
    df = pd.read_csv("womens-world-cup.csv")
    return df

df = load_world_cup_data()


st.sidebar.header("Filtros generales")

years = sorted(df["year"].unique())
selected_year = st.sidebar.selectbox("Año del Mundial", years, index=len(years) - 1)

min_matches = st.sidebar.slider(
    "Mínimo de partidos jugados",
    min_value=int(df["matches_played"].min()),
    max_value=int(df["matches_played"].max()),
    value=int(df["matches_played"].min()),
)

selected_teams = st.sidebar.multiselect(
    "Seleccionar equipos (opcional)",
    sorted(df["squad"].unique()),
)

show_raw_data = st.sidebar.checkbox("Mostrar tabla de datos filtrados")

# Aplicar filtros
filtered = df[df["year"] == selected_year].copy()
filtered = filtered[filtered["matches_played"] >= min_matches]

if selected_teams:
    filtered = filtered[filtered["squad"].isin(selected_teams)]

COUNTRY_NAME_MAP = {
    "USA": "United States",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Equ. Guinea": "Equatorial Guinea",
    "Korea DPR": "North Korea",
    "Korea Rep": "South Korea",
}

@st.cache_data
def fetch_country_info(country_name: str):
    """
    Consulta la API pública RestCountries (https://restcountries.com/)
    para obtener información demográfica básica de un país.
    No requiere credenciales.
    """
    query = COUNTRY_NAME_MAP.get(country_name, country_name)
    url = f"https://restcountries.com/v3.1/name/{query}"

    try:
        response = requests.get(url, timeout=10)
    except Exception:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except Exception:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    country = data[0]
    return {
        "official_name": country.get("name", {}).get("official", query),
        "population": country.get("population", None),
        "region": country.get("region", None),
        "subregion": country.get("subregion", None),
        "capital": (country.get("capital") or ["Desconocida"])[0],
    }

@st.cache_data
def build_country_demographics(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla con población y región por selección,
    usando la API RestCountries.
    """
    rows = []
    squads = sorted(df_year["squad"].unique())

    for squad in squads:
        info = fetch_country_info(squad)
        if info is None:
            rows.append(
                {
                    "squad": squad,
                    "official_name": squad,
                    "population": None,
                    "region": None,
                    "subregion": None,
                    "capital": None,
                }
            )
        else:
            rows.append(
                {
                    "squad": squad,
                    "official_name": info["official_name"],
                    "population": info["population"],
                    "region": info["region"],
                    "subregion": info["subregion"],
                    "capital": info["capital"],
                }
            )

    return pd.DataFrame(rows)

tab_general, tab_comparacion, tab_demografia, tab_descarga = st.tabs(
    [
        "Visión general",
        "Comparar equipos",
        "API de países (REST)",
        "Datos y descarga",
    ]
)


with tab_general:
    st.subheader(f"Resumen general - Mundial {selected_year}")

    col1, col2, col3 = st.columns(3)


    col1.metric("Equipos analizados", len(filtered["squad"].unique()))
    col2.metric("Goles totales", int(filtered["goals"].sum()))
    col3.metric(
        "Edad promedio de las jugadoras",
        f"{filtered['age'].mean():.1f} años",
    )

    if filtered.empty:
        st.warning("No hay datos que cumplan con los filtros seleccionados.")
    else:
        # ---- Gráfico 1: Goles por equipo (barras) ----
        st.markdown("### Goles totales por equipo")

        goals_by_team = (
            filtered.groupby("squad")["goals"].sum().sort_values(ascending=False)
        )

        fig1, ax1 = plt.subplots()
        goals_by_team.plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Equipo")
        ax1.set_ylabel("Goles")
        ax1.set_title("Goles totales por equipo")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig1)

        st.write(
            "Este gráfico permite identificar rápidamente qué selecciones fueron más "
            "ofensivas en el torneo según el número total de goles anotados."
        )


        st.markdown("### Relación entre posesión del balón y goles")

        fig2, ax2 = plt.subplots()
        ax2.scatter(filtered["possesion"], filtered["goals"])
        for _, row in filtered.iterrows():
            ax2.annotate(row["squad"], (row["possesion"], row["goals"]), fontsize=6)

        ax2.set_xlabel("Posesión promedio (%)")
        ax2.set_ylabel("Goles")
        ax2.set_title("Posesión vs. goles anotados")
        st.pyplot(fig2)

        st.write(
            "En este diagrama de dispersión se observa si una mayor posesión del balón "
            "se traduce o no en más goles anotados por cada selección."
        )

        st.markdown("### Juego físico: tarjetas amarillas y rojas")

        cards = filtered[["squad", "yellow_cards", "red_cards"]].set_index("squad")

        fig3, ax3 = plt.subplots()
        cards.sort_values("yellow_cards", ascending=False).plot(
            kind="bar", ax=ax3
        )
        ax3.set_xlabel("Equipo")
        ax3.set_ylabel("Tarjetas")
        ax3.set_title("Tarjetas amarillas y rojas por equipo")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig3)

        st.write(
            "Este gráfico resume el comportamiento disciplinario de cada equipo, "
            "considerando la cantidad de tarjetas amarillas y rojas recibidas."
        )


with tab_comparacion:
    st.subheader("Comparador interactivo de equipos")

    team_options = sorted(filtered["squad"].unique())
    selected_for_compare = st.multiselect(
        "Elige uno o más equipos para comparar",
        team_options,
        default=team_options[:3] if len(team_options) >= 3 else team_options,
    )

    metric = st.radio(
        "Métrica a comparar",
        ["goals", "assists", "goals_per_90", "assists_per_90"],
        index=0,
        format_func=lambda x: {
            "goals": "Goles totales",
            "assists": "Asistencias totales",
            "goals_per_90": "Goles por 90 minutos",
            "assists_per_90": "Asistencias por 90 minutos",
        }[x],
    )

    if not selected_for_compare:
        st.info("Selecciona al menos un equipo para mostrar la comparación.")
    else:
        df_comp = filtered[filtered["squad"].isin(selected_for_compare)]

        fig4, ax4 = plt.subplots()
        df_comp.plot(
            x="squad",
            y=metric,
            kind="bar",
            ax=ax4,
        )
        ax4.set_xlabel("Equipo")
        ax4.set_ylabel(metric)
        ax4.set_title(f"Comparación de equipos según: {metric}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig4)

        st.write(
            "El gráfico permite comparar de manera directa el desempeño ofensivo de "
            "las selecciones según la métrica elegida."
        )


with tab_demografia:
    st.subheader("Integración con API REST: RestCountries")

    st.write(
        "En esta sección se consulta la API pública *RestCountries* para obtener "
        "datos demográficos (población, región, capital) de los países que "
        "participan en el Mundial seleccionado. Esta API es pública y no requiere "
        "credenciales ni pagos."
    )

    if st.button("Actualizar datos desde la API"):
        # Limpia la caché para forzar una nueva consulta
        build_country_demographics.clear()
        fetch_country_info.clear()
        st.experimental_rerun()

    demo_df = build_country_demographics(filtered)

    st.markdown("#### Tabla de países y población")
    st.dataframe(demo_df)

    
    demo_merge = filtered.merge(demo_df, on="squad", how="left")
    demo_merge["goals_per_million"] = demo_merge.apply(
        lambda row: (row["goals"] / row["population"] * 1_000_000)
        if pd.notnull(row["population"]) and row["population"] > 0
        else None,
        axis=1,
    )

    valid = demo_merge.dropna(subset=["goals_per_million"]).sort_values(
        "goals_per_million", ascending=False
    )

    if valid.empty:
        st.warning(
            "No se pudo calcular la métrica de goles por millón de habitantes "
            "porque faltan datos de población desde la API."
        )
    else:
        st.markdown("### Goles por millón de habitantes (API + dataset)")

        fig5, ax5 = plt.subplots()
        ax5.bar(valid["squad"], valid["goals_per_million"])
        ax5.set_xlabel("Equipo")
        ax5.set_ylabel("Goles por millón de habitantes")
        ax5.set_title(
            "Relación entre rendimiento ofensivo y tamaño de la población del país"
        )
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig5)

        st.write(
            "Este gráfico integra la información del dataset de fútbol con la "
            "información demográfica de la API, permitiendo analizar qué selecciones "
            "tienen un rendimiento goleador alto en relación con el tamaño de su país."
        )

with tab_descarga:
    st.subheader("Datos filtrados y descarga reproducible")

    st.write(
        "Aquí se muestran los datos del Mundial ya filtrados según tus elecciones en "
        "la barra lateral. Además, puedes descargar un archivo CSV para reproducir "
        "el análisis fuera de la aplicación."
    )

    st.dataframe(filtered)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar datos filtrados (CSV)",
        data=csv_bytes,
        file_name=f"womens_world_cup_{selected_year}_filtrado.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        *Nota:* La descarga de datos permite que el análisis sea reproducible,
        ya que cualquier persona puede volver a cargar este archivo y repetir los
        cálculos realizados en la aplicación.
        """
    )

if show_raw_data:
    st.markdown("---")
    st.markdown("### Vista rápida de los datos crudos (después de los filtros)")
    st.dataframe(filtered)







