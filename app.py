import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="EnergyVision", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0f1e; color: #e0e6f0; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 100%); }
.metric-box { background: linear-gradient(135deg, #0d2137, #0a3d62); border: 1px solid #1e90ff33; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 0 20px #1e90ff22; margin-bottom: 10px; }
.metric-box h2 { color: #1e90ff; font-size: 2rem; margin: 0; }
.metric-box p { color: #a0b4cc; margin: 0; font-size: 0.85rem; }
.insight-box { background: #0d1a2e; border-left: 4px solid #1e90ff; border-radius: 6px; padding: 12px 16px; margin: 8px 0; color: #e0e6f0; font-size: 0.95rem; }
.prediction-box { background: linear-gradient(135deg, #0d2137, #0a3d62); border: 2px solid #1e90ff55; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 0 40px #1e90ff22; max-width: 500px; margin: 20px auto; }
.prediction-box h1 { color: #1e90ff; font-size: 3.5rem; margin: 10px 0; }
.prediction-box p { color: #a0b4cc; font-size: 0.95rem; }
.auteur-badge { background: linear-gradient(90deg, #0d2137, #0a3d62); border: 1px solid #1e90ff55; border-radius: 8px; padding: 10px 18px; font-size: 0.8rem; color: #7eb8f7; text-align: center; margin-bottom: 20px; }
section[data-testid="stSidebar"] { background: #060d1a; border-right: 1px solid #1e90ff22; }
.footer { text-align: center; color: #3a4f6a; font-size: 0.75rem; padding: 20px 0 5px; border-top: 1px solid #1e90ff11; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 15px 0 20px'><span style='font-size:3rem'>⚡</span><h2 style='color:#1e90ff; margin:6px 0 2px; font-family:Orbitron; font-size:1.4rem'>EnergyVision</h2><p style='color:#a0b4cc; font-size:0.75rem; margin:0'>Analyse & Prediction Energetique</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='auteur-badge'>👤 <b>Pape Mandiaye Seck</b><br>L1 Big Data · Dakar Institute of Technology<br>Annee 2025 / 2026</div>", unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio("Navigation", ["Tableau de bord", "Analyse mensuelle", "Prediction ML", "A propos"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color:#3a4f6a; font-size:0.7rem; text-align:center'>Source : UCI ML Repository<br>Dataset : 2006-2010</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/papemandiayeseck313-art/-energyvision/main/data.csv"
    df = pd.read_csv(url, sep=';', na_values='?', low_memory=False)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])
    df = df.set_index('Datetime')
    df = df.dropna()
    return df

with st.spinner("Chargement des donnees UCI..."):
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

with st.spinner("Initialisation du modele IA..."):
    model = train_model(df)

def make_fig(w=13, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0d1a2e')
    ax.tick_params(colors='#a0b4cc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e90ff22')
    return fig, ax

if page == "Tableau de bord":
    st.markdown("<h1 style='color:#1e90ff'>Tableau de bord</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0b4cc'>Vue d'ensemble de la consommation electrique d'un foyer residentiel (2006-2010)</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-box'><p>Total mesures</p><h2>{df.shape[0]:,}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><p>Conso moyenne</p><h2>{df['Global_active_power'].mean():.3f} kW</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><p>Conso maximale</p><h2>{df['Global_active_power'].max():.2f} kW</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-box'><p>Periode couverte</p><h2 style='font-size:1.2rem'>2006-2010</h2></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Consommation journaliere moyenne")
    daily = df['Global_active_power'].resample('D').mean()
    fig, ax = make_fig(13, 3.5)
    ax.plot(daily, color='#1e90ff', linewidth=0.8)
    ax.fill_between(daily.index, daily.values, alpha=0.1, color='#1e90ff')
    ax.set_ylabel('kW', color='#a0b4cc')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("<div class='insight-box'>La consommation presente une saisonnalite claire - les pics hivernaux sont nettement plus eleves que les creux estivaux.</div>", unsafe_allow_html=True)
    st.markdown("### Consommation moyenne par heure de la journee")
    hourly = df.groupby(df.index.hour)['Global_active_power'].mean()
    fig2, ax2 = make_fig(11, 3.5)
    ax2.plot(hourly.index, hourly.values, color='#00d4aa', marker='o', linewidth=2, markersize=5)
    ax2.fill_between(hourly.index, hourly.values, alpha=0.2, color='#00d4aa')
    ax2.set_xlabel('Heure', color='#a0b4cc')
    ax2.set_ylabel('kW', color='#a0b4cc')
    ax2.set_xticks(range(0, 24))
    ax2.grid(axis='y', color='#1e90ff11', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("<div class='insight-box'>Deux pics de consommation : le matin (7h-9h) et surtout le soir (18h-22h), correspondant aux heures de presence au foyer.</div>", unsafe_allow_html=True)
    st.markdown("### Consommation moyenne par saison")
    saison_map2 = {12:'Hiver',1:'Hiver',2:'Hiver',3:'Printemps',4:'Printemps',5:'Printemps',6:'Ete',7:'Ete',8:'Ete',9:'Automne',10:'Automne',11:'Automne'}
    df2 = df.copy()
    df2['saison'] = df2.index.month.map(saison_map2)
    saison_data = df2.groupby('saison')['Global_active_power'].mean().sort_values(ascending=False)
    colors_s = ['#1e90ff','#00d4aa','#f39c12','#e74c3c']
    fig3, ax3 = make_fig(8, 3.5)
    bars = ax3.bar(saison_data.index, saison_data.values, color=colors_s[:len(saison_data)], width=0.5)
    for bar, val in zip(bars, saison_data.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f} kW', ha='center', va='bottom', color='#e0e6f0', fontsize=11)
    ax3.set_ylabel('kW', color='#a0b4cc')
    ax3.grid(axis='y', color='#1e90ff11', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown("<div class='insight-box'>L'hiver concentre la consommation la plus elevee. L'ete enregistre les niveaux les plus bas.</div>", unsafe_allow_html=True)
    st.markdown("### Consommation moyenne par jour de la semaine")
    jours = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']
    weekly = df.groupby(df.index.dayofweek)['Global_active_power'].mean()
    fig4, ax4 = make_fig(10, 3)
    colors_w = ['#1e90ff' if i < 5 else '#f39c12' for i in range(7)]
    bars2 = ax4.bar(jours, weekly.values, color=colors_w, width=0.6)
    for bar, val in zip(bars2, weekly.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', color='#e0e6f0', fontsize=10)
    ax4.set_ylabel('kW', color='#a0b4cc')
    ax4.grid(axis='y', color='#1e90ff11', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown("<div class='insight-box'>Le weekend (orange) affiche une consommation plus elevee - les occupants restent davantage a domicile.</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>EnergyVision - Pape Mandiaye Seck - L1 Big Data - Dakar Institute of Technology - 2025/2026</div>", unsafe_allow_html=True)

elif page == "Analyse mensuelle":
    st.markdown("<h1 style='color:#1e90ff'>Analyse Mensuelle</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0b4cc'>Evolution de la consommation mois par mois sur toute la periode</p>", unsafe_allow_html=True)
    monthly = df['Global_active_power'].resample('ME').mean()
    fig, ax = make_fig(13, 4)
    ax.bar(range(len(monthly)), monthly.values, color='#1e90ff', width=0.7, alpha=0.8)
    ax.plot(range(len(monthly)), monthly.values, color='#00d4aa', linewidth=2, marker='o', markersize=4)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([d.strftime('%b %Y') for d in monthly.index], rotation=90, fontsize=7, color='#a0b4cc')
    ax.set_ylabel('kW', color='#a0b4cc')
    ax.grid(axis='y', color='#1e90ff11', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("### Consommation moyenne par mois de l'annee")
    mois_noms = ['Jan','Fev','Mar','Avr','Mai','Jun','Jul','Aou','Sep','Oct','Nov','Dec']
    monthly_avg = df.groupby(df.index.month)['Global_active_power'].mean()
    colors_m = ['#1e90ff' if v >= monthly_avg.mean() else '#00d4aa' for v in monthly_avg.values]
    fig2, ax2 = make_fig(11, 3.5)
    ax2.bar(mois_noms, monthly_avg.values, color=colors_m, width=0.6)
    ax2.axhline(y=monthly_avg.mean(), color='#f39c12', linewidth=1.5, linestyle='--', label='Moyenne annuelle')
    ax2.set_ylabel('kW', color='#a0b4cc')
    ax2.legend(facecolor='#0d1a2e', edgecolor='#1e90ff33', labelcolor='#a0b4cc')
    ax2.grid(axis='y', color='#1e90ff11', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("<div class='insight-box'>Bleu = au-dessus de la moyenne | Vert = en-dessous. Decembre-Janvier sont les mois les plus energivores.</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>EnergyVision - Pape Mandiaye Seck - L1 Big Data - Dakar Institute of Technology - 2025/2026</div>", unsafe_allow_html=True)

elif page == "Prediction ML":
    st.markdown("<h1 style='color:#1e90ff'>Prediction ML</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0b4cc'>Utilisez le modele Random Forest (R2 = 0.9994) pour predire la consommation electrique en temps reel.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        heure = st.slider("Heure de la journee", 0, 23, 12)
    with col2:
        mois = st.slider("Mois", 1, 12, 6)
    with col3:
        jour = st.slider("Jour de la semaine", 0, 6, 0, help="0=Lundi, 6=Dimanche")
    mois_noms = {1:'Janvier',2:'Fevrier',3:'Mars',4:'Avril',5:'Mai',6:'Juin',7:'Juillet',8:'Aout',9:'Septembre',10:'Octobre',11:'Novembre',12:'Decembre'}
    jour_noms = {0:'Lundi',1:'Mardi',2:'Mercredi',3:'Jeudi',4:'Vendredi',5:'Samedi',6:'Dimanche'}
    saison_noms = {0:'Hiver',1:'Printemps',2:'Ete',3:'Automne'}
    saison_input = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[mois]
    input_data = pd.DataFrame([[heure, jour, mois, 2009, saison_input, 241.0, 4.6, 1.0, 1.0, 6.0]], columns=['heure','jour_semaine','mois','annee','saison','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
    prediction = model.predict(input_data)[0]
    if prediction < 0.5:
        niveau = "Faible"
        couleur = "#00d4aa"
    elif prediction < 1.2:
        niveau = "Moderee"
        couleur = "#f39c12"
    else:
        niveau = "Elevee"
        couleur = "#e74c3c"
    st.markdown(f"<div class='prediction-box'><p style='color:#a0b4cc; font-size:0.9rem; margin-bottom:5px'>Consommation predite</p><h1 style='color:{couleur}'>{prediction:.3f} kW</h1><p style='font-size:1.1rem; color:{couleur}; font-weight:bold'>{niveau}</p><hr style='border-color:#1e90ff22; margin:15px 0'><p>{heure}h00 - {mois_noms[mois]} - {jour_noms[jour]}</p><p>Saison : {saison_noms[saison_input]}</p></div>", unsafe_allow_html=True)
    st.markdown("### Prediction sur 24h pour ce jour")
    predictions_24h = []
    for h in range(24):
        inp = pd.DataFrame([[h, jour, mois, 2009, saison_input, 241.0, 4.6, 1.0, 1.0, 6.0]], columns=['heure','jour_semaine','mois','annee','saison','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'])
        predictions_24h.append(model.predict(inp)[0])
    fig, ax = make_fig(11, 3)
    ax.plot(range(24), predictions_24h, color='#1e90ff', linewidth=2, marker='o', markersize=4)
    ax.fill_between(range(24), predictions_24h, alpha=0.2, color='#1e90ff')
    ax.axvline(x=heure, color='#f39c12', linewidth=2, linestyle='--', label=f'Heure selectionnee ({heure}h)')
    ax.set_xlabel('Heure', color='#a0b4cc')
    ax.set_ylabel('kW', color='#a0b4cc')
    ax.set_xticks(range(0, 24))
    ax.legend(facecolor='#0d1a2e', edgecolor='#1e90ff33', labelcolor='#a0b4cc')
    ax.grid(axis='y', color='#1e90ff11', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("### Performances des modeles ML")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-box'><p>Regression Lineaire</p><h2 style='font-size:1.3rem'>R2 = 0.9982</h2><p>RMSE = 0.0444</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'><p>XGBoost</p><h2 style='font-size:1.3rem'>R2 = 0.9992</h2><p>RMSE = 0.0303</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box' style='border-color:#00d4aa55'><p>Random Forest</p><h2 style='font-size:1.3rem; color:#00d4aa'>R2 = 0.9994</h2><p>RMSE = 0.0265</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>EnergyVision - Pape Mandiaye Seck - L1 Big Data - Dakar Institute of Technology - 2025/2026</div>", unsafe_allow_html=True)

elif page == "A propos":
    st.markdown("<h1 style='color:#1e90ff'>A propos du projet</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1a2e; border:1px solid #1e90ff33; border-radius:12px; padding:28px; line-height:1.9'>
    <h3 style='color:#1e90ff'>EnergyVision</h3>
    <p>Ce projet analyse et predit la consommation electrique d un foyer residentiel a partir d un jeu de donnees reel de <b>2 075 259 mesures</b> collectees entre 2006 et 2010.</p>
    <h4 style='color:#00d4aa'>Objectifs</h4>
    <ul><li>Analyser les tendances et patterns de consommation electrique</li><li>Identifier les pics de consommation par heure, jour et saison</li><li>Predire la consommation future grace au Machine Learning</li><li>Deployer une application web accessible en ligne</li></ul>
    <h4 style='color:#00d4aa'>Source des donnees</h4>
    <p><b>UCI Machine Learning Repository</b><br>Individual Household Electric Power Consumption Dataset<br>Auteurs : Georges Hebrail et Alice Berard (EDF R&D, France)</p>
    <h4 style='color:#00d4aa'>Technologies utilisees</h4>
    <ul><li><b>Python</b> - Pandas, NumPy, Matplotlib</li><li><b>SQL</b> - SQLite via SQLAlchemy</li><li><b>Machine Learning</b> - Scikit-learn (Regression Lineaire, Random Forest), XGBoost</li><li><b>Web App</b> - Streamlit deploye sur Streamlit Cloud</li><li><b>Versioning</b> - GitHub</li></ul>
    <h4 style='color:#00d4aa'>Resultats des modeles</h4>
    <ul><li>Regression Lineaire - R2 : 0.9982 | RMSE : 0.0444</li><li>XGBoost - R2 : 0.9992 | RMSE : 0.0303</li><li><b style='color:#00d4aa'>Random Forest - R2 : 0.9994 | RMSE : 0.0265 - Meilleur modele</b></li></ul>
    <h4 style='color:#00d4aa'>Perspectives</h4>
    <ul><li>Integrer des donnees reelles de foyers senegalais (SENELEC)</li><li>Deployer des capteurs IoT pour la collecte en temps reel</li><li>Developper des alertes automatiques en cas de surconsommation</li></ul>
    <h4 style='color:#00d4aa'>Auteur</h4>
    <p><b>Pape Mandiaye Seck</b><br>Etudiant en L1 Big Data<br>Dakar Institute of Technology - 2025/2026</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='footer'>EnergyVision - Pape Mandiaye Seck - L1 Big Data - Dakar Institute of Technology - 2025/2026</div>", unsafe_allow_html=True)