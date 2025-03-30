import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Prédiction de Déviations de Forages Miniers", layout="wide")

# Titre et introduction
st.title("Prédiction des déviations de forages d'exploration minière")
st.markdown("""
Cette application utilise le machine learning pour prédire les déviations d'azimuth et d'inclinaison des forages
en fonction des paramètres initiaux et des caractéristiques géologiques.
""")

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Sidebar pour les options
st.sidebar.header("Configuration")

# Option pour uploader les données ou utiliser des données de démonstration
data_option = st.sidebar.radio(
    "Source des données",
    ["Charger mes données", "Utiliser données démo"]
)

df = None

if data_option == "Charger mes données":
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
else:
    # Données de démonstration
    st.sidebar.info("Utilisation de données de démonstration")
    # Créer des données synthétiques pour la démonstration
    np.random.seed(42)
    n_samples = 1000
    
    prof_finale = np.random.uniform(100, 1000, n_samples)
    azimuth_initial = np.random.uniform(0, 360, n_samples)
    inclinaison_initiale = np.random.uniform(-90, 0, n_samples)
    vitesse_rotation = np.random.uniform(50, 200, n_samples)
    
    # Lithologies possibles
    lithologies = ['Granite', 'Schiste', 'Gneiss', 'Calcaire', 'Basalte']
    lithologie = np.random.choice(lithologies, n_samples)
    
    # Créer une relation entre les entrées et les déviations (simplifiée)
    azimuth_deviation = (
        0.05 * prof_finale 
        + 0.02 * azimuth_initial 
        + 0.1 * inclinaison_initiale 
        + 0.03 * vitesse_rotation 
        + np.random.normal(0, 10, n_samples)
    )
    
    inclinaison_deviation = (
        0.03 * prof_finale 
        - 0.01 * azimuth_initial 
        + 0.05 * inclinaison_initiale 
        + 0.02 * vitesse_rotation 
        + np.random.normal(0, 5, n_samples)
    )
    
    # Ajouter un effet de la lithologie (différent pour chaque type)
    lithology_effect = {
        'Granite': (2.0, 1.0),
        'Schiste': (-1.5, 3.0),
        'Gneiss': (0.5, -2.0),
        'Calcaire': (-1.0, -1.5),
        'Basalte': (3.0, 2.5)
    }
    
    for i, lith in enumerate(lithologie):
        effect_az, effect_inc = lithology_effect[lith]
        azimuth_deviation[i] += effect_az
        inclinaison_deviation[i] += effect_inc
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'profondeur_finale': prof_finale,
        'azimuth_initial': azimuth_initial,
        'inclinaison_initiale': inclinaison_initiale,
        'lithologie': lithologie,
        'vitesse_rotation': vitesse_rotation,
        'deviation_azimuth': azimuth_deviation,
        'deviation_inclinaison': inclinaison_deviation
    })

# Si des données sont disponibles, afficher l'application principale
if df is not None:
    st.write("## Aperçu des données")
    st.dataframe(df.head())
    
    st.write("## Statistiques descriptives")
    st.dataframe(df.describe())
    
    st.write("## Distribution des lithologies")
    fig_litho = px.histogram(df, x='lithologie', color='lithologie')
    st.plotly_chart(fig_litho)
    
    # Analyse exploratoire
    st.write("## Analyse exploratoire des données")
    
    tab1, tab2, tab3 = st.tabs(["Corrélations", "Déviations par lithologie", "Déviations vs paramètres"])
    
    with tab1:
        # Matrice de corrélation
        numeric_cols = df.select_dtypes(include=np.number).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True, 
                            color_continuous_scale='RdBu_r',
                            title="Matrice de corrélation")
        st.plotly_chart(fig_corr)
    
    with tab2:
        # Boîtes à moustaches par lithologie
        fig_box1 = px.box(df, x='lithologie', y='deviation_azimuth', 
                        title="Déviation d'azimuth par lithologie", color='lithologie')
        st.plotly_chart(fig_box1)
        
        fig_box2 = px.box(df, x='lithologie', y='deviation_inclinaison', 
                        title="Déviation d'inclinaison par lithologie", color='lithologie')
        st.plotly_chart(fig_box2)
    
    with tab3:
        # Nuages de points des déviations vs paramètres
        col1, col2 = st.columns(2)
        
        features = ['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'vitesse_rotation']
        for i, feature in enumerate(features):
            fig_scatter1 = px.scatter(df, x=feature, y='deviation_azimuth', 
                                    color='lithologie', opacity=0.7,
                                    title=f"Déviation d'azimuth vs {feature}")
            
            fig_scatter2 = px.scatter(df, x=feature, y='deviation_inclinaison', 
                                    color='lithologie', opacity=0.7,
                                    title=f"Déviation d'inclinaison vs {feature}")
            
            if i % 2 == 0:
                col1.plotly_chart(fig_scatter1, use_container_width=True)
                col1.plotly_chart(fig_scatter2, use_container_width=True)
            else:
                col2.plotly_chart(fig_scatter1, use_container_width=True)
                col2.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Modélisation
    st.write("## Modélisation")
    
    # Définir les caractéristiques et cibles
    X = df[['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'lithologie', 'vitesse_rotation']]
    y_azimuth = df['deviation_azimuth']
    y_inclinaison = df['deviation_inclinaison']
    
    # Préparation pour la modélisation
    numeric_features = ['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'vitesse_rotation']
    categorical_features = ['lithologie']
    
    # Préprocesseurs
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Sélection du modèle
    model_option = st.sidebar.selectbox(
        "Choisir un modèle",
        ["Random Forest", "SVM", "Régression Linéaire", "Réseau de Neurones"]
    )
    
    # Choix du modèle
    if model_option == "Random Forest":
        model_azimuth = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model_inclinaison = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
    elif model_option == "SVM":
        model_azimuth = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR())
        ])
        
        model_inclinaison = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR())
        ])
        
    elif model_option == "Régression Linéaire":
        model_azimuth = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        model_inclinaison = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
    else:  # Réseau de Neurones
        model_azimuth = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))
        ])
        
        model_inclinaison = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))
        ])
    
    # Diviser les données
    X_train, X_test, y_azimuth_train, y_azimuth_test, y_inclinaison_train, y_inclinaison_test = train_test_split(
        X, y_azimuth, y_inclinaison, test_size=0.2, random_state=42
    )
    
    # Entraîner les modèles si l'utilisateur clique sur le bouton
    if st.sidebar.button("Entraîner le modèle"):
        with st.spinner(f"Entraînement du modèle {model_option} en cours..."):
            # Entraînement pour la déviation d'azimuth
            model_azimuth.fit(X_train, y_azimuth_train)
            y_azimuth_pred = model_azimuth.predict(X_test)
            
            # Entraînement pour la déviation d'inclinaison
            model_inclinaison.fit(X_train, y_inclinaison_train)
            y_inclinaison_pred = model_inclinaison.predict(X_test)
            
            # Métriques de performance
            azimuth_rmse = np.sqrt(mean_squared_error(y_azimuth_test, y_azimuth_pred))
            azimuth_r2 = r2_score(y_azimuth_test, y_azimuth_pred)
            
            inclinaison_rmse = np.sqrt(mean_squared_error(y_inclinaison_test, y_inclinaison_pred))
            inclinaison_r2 = r2_score(y_inclinaison_test, y_inclinaison_pred)
        
        # Affichage des résultats
        st.write("### Résultats de l'entraînement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Déviation d'azimuth**")
            st.write(f"RMSE: {azimuth_rmse:.4f}")
            st.write(f"R²: {azimuth_r2:.4f}")
            
            # Graphique des prédictions vs réelles
            fig_pred_az = px.scatter(x=y_azimuth_test, y=y_azimuth_pred, 
                                    labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                                    title="Prédictions vs Réelles - Déviation d'azimuth")
            fig_pred_az.add_shape(type='line', line=dict(dash='dash'),
                                x0=y_azimuth_test.min(), y0=y_azimuth_test.min(),
                                x1=y_azimuth_test.max(), y1=y_azimuth_test.max())
            st.plotly_chart(fig_pred_az)
        
        with col2:
            st.write("**Déviation d'inclinaison**")
            st.write(f"RMSE: {inclinaison_rmse:.4f}")
            st.write(f"R²: {inclinaison_r2:.4f}")
            
            # Graphique des prédictions vs réelles
            fig_pred_inc = px.scatter(x=y_inclinaison_test, y=y_inclinaison_pred, 
                                    labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                                    title="Prédictions vs Réelles - Déviation d'inclinaison")
            fig_pred_inc.add_shape(type='line', line=dict(dash='dash'),
                                x0=y_inclinaison_test.min(), y0=y_inclinaison_test.min(),
                                x1=y_inclinaison_test.max(), y1=y_inclinaison_test.max())
            st.plotly_chart(fig_pred_inc)
    
    # Interface de prédiction pour nouveaux forages
    st.write("## Prédiction pour un nouveau forage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prof_finale_input = st.number_input("Profondeur finale (m)", min_value=50.0, max_value=2000.0, value=500.0)
        azimuth_initial_input = st.number_input("Azimuth initial (degrés)", min_value=0.0, max_value=360.0, value=90.0)
    
    with col2:
        inclinaison_initiale_input = st.number_input("Inclinaison initiale (degrés)", min_value=-90.0, max_value=0.0, value=-45.0)
        vitesse_rotation_input = st.number_input("Vitesse de rotation (tr/min)", min_value=20.0, max_value=300.0, value=120.0)
    
    lithologies_uniques = df['lithologie'].unique().tolist()
    lithologie_input = st.selectbox("Lithologie", lithologies_uniques)
    
    if st.button("Prédire les déviations"):
        # Vérifier si les modèles ont été entraînés
        try:
            # Créer un dataframe avec les données d'entrée
            input_data = pd.DataFrame({
                'profondeur_finale': [prof_finale_input],
                'azimuth_initial': [azimuth_initial_input],
                'inclinaison_initiale': [inclinaison_initiale_input],
                'lithologie': [lithologie_input],
                'vitesse_rotation': [vitesse_rotation_input]
            })
            
            # Faire les prédictions
            predicted_azimuth = model_azimuth.predict(input_data)[0]
            predicted_inclinaison = model_inclinaison.predict(input_data)[0]
            
            # Afficher les résultats
            st.write("### Résultats de la prédiction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Déviation d'azimuth prédite", f"{predicted_azimuth:.2f}°")
            with col2:
                st.metric("Déviation d'inclinaison prédite", f"{predicted_inclinaison:.2f}°")
            
            # Calculer les valeurs finales
            azimuth_final = azimuth_initial_input + predicted_azimuth
            inclinaison_final = inclinaison_initiale_input + predicted_inclinaison
            
            # Normalization pour azimuth (0-360°)
            azimuth_final = azimuth_final % 360
            
            # Contraindre l'inclinaison entre -90 et 0
            inclinaison_final = max(-90, min(0, inclinaison_final))
            
            st.write("### Orientation finale du forage")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Azimuth final", f"{azimuth_final:.2f}°")
            with col2:
                st.metric("Inclinaison finale", f"{inclinaison_final:.2f}°")
            
            # Visualisation 3D de la trajectoire du forage
            st.write("### Visualisation de la trajectoire")
            
            # Simplification: trajectoire linéaire entre le point de départ et le point final
            # Dans un cas réel, il faudrait modéliser la trajectoire de manière plus précise
            
            # Conversion des coordonnées polaires en coordonnées cartésiennes
            def sph_to_cart(depth, azimuth, inclination):
                # Conversion des degrés en radians
                azimuth_rad = np.radians(azimuth)
                inclination_rad = np.radians(inclination)
                
                # x pointe vers l'est, y vers le nord, z vers le haut
                x = depth * np.cos(inclination_rad) * np.sin(azimuth_rad)
                y = depth * np.cos(inclination_rad) * np.cos(azimuth_rad)
                z = depth * np.sin(inclination_rad)  # z est négatif car inclination est négative
                
                return x, y, z
            
            # Calculer plusieurs points le long de la trajectoire
            num_points = 100
            depths = np.linspace(0, prof_finale_input, num_points)
            
            # Interpolation linéaire entre les angles initiaux et finaux
            azimuth_interp = np.linspace(azimuth_initial_input, azimuth_final, num_points)
            inclination_interp = np.linspace(inclinaison_initiale_input, inclinaison_final, num_points)
            
            # Calculer les coordonnées cartésiennes pour chaque point
            x_coords, y_coords, z_coords = [], [], []
            for i in range(num_points):
                x, y, z = sph_to_cart(depths[i], azimuth_interp[i], inclination_interp[i])
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
            
            # Créer la visualisation 3D
            fig = go.Figure(data=[go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(
                    color='blue',
                    width=6
                )
            )])
            
            # Ajouter la position de départ
            fig.add_trace(go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(
                    size=8,
                    color='green'
                ),
                name='Départ'
            ))
            
            # Ajouter la position finale
            fig.add_trace(go.Scatter3d(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                z=[z_coords[-1]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red'
                ),
                name='Arrivée'
            ))
            
            fig.update_layout(
                title="Trajectoire du forage",
                scene=dict(
                    xaxis_title='Est (m)',
                    yaxis_title='Nord (m)',
                    zaxis_title='Profondeur (m)',
                    aspectmode='data'
                ),
                width=700,
                height=700
            )
            
            st.plotly_chart(fig)
            
        except NameError:
            st.error("Veuillez d'abord entraîner le modèle en cliquant sur le bouton 'Entraîner le modèle' dans la barre latérale.")

else:
    # Message pour guider l'utilisateur si aucune donnée n'est encore chargée
    if data_option == "Charger mes données":
        st.info("Veuillez charger un fichier CSV depuis la barre latérale pour commencer.")