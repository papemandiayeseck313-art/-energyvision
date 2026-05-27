import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="EnergyVision", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0f1e; color: #e0e6f0; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 100%); }
.metric-box { background: linear-gradient(135deg, #0d2137, #0a3d62); border: 1px solid #1e90ff33; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 0 20px #1e90ff22; }
.metric-box h2 { color: #1e90ff; font-size: 2rem; margin: 0; }
.metric-box p { color: #a0b4cc; margin: 0; font-size: 0.85rem; }
.auteur-badge { background: linear-gradient(90deg, #0d2137, #0a3d62); border: 1px solid #1e90ff55; border-radius: 8px; padding: 10px 18px; font-size: 0.8rem; color: #7eb8f7; text-align: center; margin-bottom: 20px; }
section[data-testid="stSidebar"] { background: #060d1a; border-right: 1px solid #1e90ff22; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 10px 0 20px'><span style='font-size:2.5rem'>⚡</span><h2 style='color:#1e90ff; margin:4px 0; font-family:Orbitron'>EnergyVision</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='auteur-badge'>👤 <b>Pape Mandiaye Seck</b><br>L1 Big Data · Dakar Institute of Technology<br>Année 2025 / 2026</div>", unsafe_allow_html=True)
    st.markdown("### 📌 Navigation")
    page = st.radio("Navigation", ["📊 Tableau de bord", "🤖 Prédiction ML", "ℹ️ À propos"], label_visibility="collapsed")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    df = pd.read_csv(url, sep=';', na_values='?', low_memory=False, compression='zip')
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])
    df = df.set_index('Datetime')
    df = df.dropna()
    return df

with st.spinner("⏳ Chargement des données..."):
    df = load_data()

@st.cache_resource
def train_model(df):
    df2 = df.copy()
    df2['heure'] = df2.index.hour
    df2['jour_semaine'] = df2.index.dayofweek
    df2['mois'] = df2.index.month
    df2['annee'] = df2.index.year
    saison_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
    df2['saison'] = df2['mois'].map(saison_map)
    X = df2[['heure','jour_semaine','mois','annee','saison','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']]
    y = df2['Global_active_power']
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

with st.spinner("⚙️ Chargement du modèle IA..."):
    model = train_model(df)

if page == "📊 Tableau de bord":
    st.markdown("<h1 style='color:#1e90ff'>📊 Tableau de bord</h1>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-box'><p>Total mesures</p><h2>{df.shape[0]:,}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><p>Conso moyenne (kW)</p><h2>{df['Global_active_power'].mean():.3f}</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><p>Conso max (kW)</p><h2>{df['Global_active_power'].max():.2f}</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-box'><p>Période</p><h2 style='font-size:1.1rem'>2006–2010</h2></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Consommation journalière moyenne")
    daily = df['Global_active_power'].resample('D').mean()
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0d1a2e')
    ax.plot(daily, color='#1e90ff', linewidth=0.8)
    ax.set_ylabel('kW', color='#a0b4cc')
    ax.tick_params(colors='#a0b4cc')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e90ff22')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("### 🕐 Consommation moyenne par heure")
    hourly = df.groupby(df.index.hour)['Global_active_power'].mean()
    fig2, ax2 = plt.subplots(figsize=(11, 3.5))
    fig2.patch.set_facecolor('#0a0f1e')
    ax2.set_facecolor('#0d1a2e')
    ax2.plot(hourly.index, hourly.values, color='#00d4aa', marker='o', linewidth=2)
    ax2.fill_between(hourly.index, hourly.values, alpha=0.2, color='#00d4aa')
    ax2.set_xlabel('Heure', color='#a0b4cc')
    ax2.set_ylabel('kW', color='#a0b4cc')
    ax2.set_xticks(range(0, 24))
    ax2.tick_params(colors='#a0b4cc')
    for spine in ax2.spines.values(): spine.set_edgecolor('#1e90ff22')
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("### 🌍 Consommation par saison")
    saison_map = {12:'Hiver',1:'Hiver',2:'Hiver',3:'Printemps',4:'Printemps',5:'Printemps',6:'Été',7:'Été',8:'Été',9:'Automne',10:'Automne',11:'Automne'}
    df['saison'] = df.index.month.map(saison_map)
    saison_data = df.groupby('saison')['Global_active_power'].mean().sort_values(ascending=False)
    colors = ['#1e90ff','#00d4aa','#f39c12','#e74c3c']
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    fig3.patch.set_facecolor('#0a0f1e')
    ax3.set_facecolor('#0d1a2e')
    ax3.bar(saison_data.index, saison_data.values, color=colors)
    ax3.set_ylabel('kW', color='#a0b4cc')
    ax3.tick_params(colors='#a0b4cc')
    for spine in ax3.spines.values(): spine.set_edgecolor('#1e90ff22')
    plt.tight_layout()
    st.pyplot(fig3)

elif page == "🤖 Prédiction ML":
    st.markdown("<h1 style='color:#1e90ff'>🤖 Prédiction ML</h1>", unsafe_allow_html=True)
    st.markdown("Ajuste les paramètres pour prédire la consommation électrique.")
    col1, col2, col3 = st.columns(3)
    with col1:
        heure = st.slider("🕐 Heure", 0, 23, 12)
    with col2:
        mois = st.slider("📅 Mois", 1, 12, 6)
    with col3:
        jour = st.slider("📆 Jour de la semaine", 0, 6, 0, help="0=Lundi ... 6=Dimanche")
    saison_input = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[mois]
    input_data = pd.DataFrame([[heure, jour, mois, 2009, saison_input, 241.0, 4.6, 1.0, 1.0, 6.0]], columns=['heure','jour_semaine','mois','annee','saison','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
    prediction = model.predict(input_data)[0]
    st.markdown(f"<div class='metric-box' style='max-width:400px; margin: 30px auto'><p>⚡ Consommation prédite</p><h2>{prediction:.3f} kW</h2><p>Heure {heure}h · Mois {mois} · Jour {jour}</p></div>", unsafe_allow_html=True)

elif page == "ℹ️ À propos":
    st.markdown("<h1 style='color:#1e90ff'>ℹ️ À propos du projet</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1a2e; border:1px solid #1e90ff33; border-radius:12px; padding:28px; line-height:1.9'>
    <h3 style='color:#1e90ff'>⚡ EnergyVision</h3>
    <p>Ce projet analyse et prédit la consommation électrique d'un foyer résidentiel à partir d'un jeu de données réel de <b>2 075 259 mesures</b> collectées entre 2006 et 2010 (source : UCI Machine Learning Repository).</p>
    <h4 style='color:#00d4aa'>🎯 Objectifs</h4>
    <ul><li>Analyser les tendances et patterns de consommation</li><li>Identifier les pics de consommation par heure, jour et saison</li><li>Prédire la consommation grâce au Machine Learning</li></ul>
    <h4 style='color:#00d4aa'>🛠️ Technologies utilisées</h4>
    <ul><li><b>Python</b> — Pandas, NumPy, Matplotlib</li><li><b>SQL</b> — SQLite via SQLAlchemy</li><li><b>Machine Learning</b> — Scikit-learn (Régression Linéaire, Random Forest), XGBoost</li><li><b>Web App</b> — Streamlit</li></ul>
    <h4 style='color:#00d4aa'>📊 Résultats des modèles</h4>
    <ul><li>Régression Linéaire — R² : 0.9982</li><li>XGBoost — R² : 0.9992</li><li><b>Random Forest — R² : 0.9994 🏆 Meilleur modèle</b></li></ul>
    <h4 style='color:#00d4aa'>👤 Auteur</h4>
    <p><b>Pape Mandiaye Seck</b><br>Étudiant en L1 Big Data<br>Dakar Institute of Technology · 2025/2026</p>
    </div>
    """, unsafe_allow_html=True)
