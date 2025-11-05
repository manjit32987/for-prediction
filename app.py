import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Bacteria Self-Healing Concrete Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2E86AB;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    </style>
    """, unsafe_allow_html=True)

class BacteriaHealingPredictor:
    def __init__(self):
        self.healing_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.strength_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler_X = StandardScaler()
        self.is_trained = False
        
    def load_data(self, cereus_file, subtilis_file):
        """Load data from Excel files"""
        try:
            cereus_df = pd.read_excel(cereus_file)
            subtilis_df = pd.read_excel(subtilis_file)
            
            # Add species column
            cereus_df['Species_Encoded'] = 0  # B. cereus = 0
            subtilis_df['Species_Encoded'] = 1  # B. subtilis = 1
            
            # Combine datasets
            self.data = pd.concat([cereus_df, subtilis_df], ignore_index=True)
            return True, len(cereus_df), len(subtilis_df)
        except Exception as e:
            return False, str(e), None
    
    def train_model(self):
        """Train the model on loaded data"""
        # Features: Species, Concentration, Age, Initial Crack Width, Final Crack Width
        X = self.data[['Species_Encoded', 'Concentration (cells/ml)', 'Age (days)', 
                        'Initial Crack Width (mm)', 'Final Crack Width (mm)']]
        
        # Targets
        y_healing = self.data['Healing Efficiency (%)']
        y_strength = self.data['Compressive Strength Gain (%)']
        
        # Split data
        X_train, X_test, y_heal_train, y_heal_test, y_str_train, y_str_test = train_test_split(
            X, y_healing, y_strength, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Train healing efficiency model
        self.healing_model.fit(X_train_scaled, y_heal_train)
        heal_score = self.healing_model.score(X_test_scaled, y_heal_test)
        
        # Train strength gain model
        self.strength_model.fit(X_train_scaled, y_str_train)
        strength_score = self.strength_model.score(X_test_scaled, y_str_test)
        
        self.is_trained = True
        return heal_score, strength_score
        
    def predict_timeline(self, species, concentration, initial_crack, final_crack):
        """Generate predictions for days 1-30"""
        if not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")
        
        # Encode species
        species_encoded = 0 if species == 'B. cereus' else 1
        
        # Generate predictions for each day
        days = np.arange(1, 31)
        healing_predictions = []
        strength_predictions = []
        
        for day in days:
            # Create feature vector
            features = np.array([[species_encoded, concentration, day, initial_crack, final_crack]])
            features_scaled = self.scaler_X.transform(features)
            
            # Predict
            healing_pred = self.healing_model.predict(features_scaled)[0]
            strength_pred = self.strength_model.predict(features_scaled)[0]
            
            healing_predictions.append(max(0, min(100, healing_pred)))
            strength_predictions.append(max(0, strength_pred))
        
        return days, healing_predictions, strength_predictions

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = BacteriaHealingPredictor()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'cereus_count' not in st.session_state:
    st.session_state.cereus_count = 0
if 'subtilis_count' not in st.session_state:
    st.session_state.subtilis_count = 0
if 'heal_score' not in st.session_state:
    st.session_state.heal_score = 0.0
if 'strength_score' not in st.session_state:
    st.session_state.strength_score = 0.0

# Title and description
st.title("🦠 Bacteria Self-Healing Concrete Predictor")
st.markdown("### Machine Learning Model for Predicting Healing Efficiency and Compressive Strength Gain")
st.markdown("---")

# Sidebar for model training
with st.sidebar:
    st.header("📊 Model Training")
    
    # Show model status
    if st.session_state.model_trained:
        st.success("✅ Model is trained and ready!")
        st.info(f"📦 B. cereus samples: {st.session_state.cereus_count}")
        st.info(f"📦 B. subtilis samples: {st.session_state.subtilis_count}")
        st.metric("Healing Efficiency R²", f"{st.session_state.heal_score:.4f}")
        st.metric("Strength Gain R²", f"{st.session_state.strength_score:.4f}")
        
        if st.button("🔄 Retrain Model"):
            st.session_state.model_trained = False
            st.session_state.predictor = BacteriaHealingPredictor()
            st.rerun()
    else:
        st.markdown("Upload your Excel files to train the model")
        
        cereus_file = st.file_uploader("Upload B. cereus Excel file", type=['xlsx', 'xls'], key='cereus')
        subtilis_file = st.file_uploader("Upload B. subtilis Excel file", type=['xlsx', 'xls'], key='subtilis')
        
        if st.button("🚀 Train Model"):
            if cereus_file and subtilis_file:
                with st.spinner("Loading data..."):
                    success, cereus_count, subtilis_count = st.session_state.predictor.load_data(cereus_file, subtilis_file)
                
                if success:
                    st.session_state.cereus_count = cereus_count
                    st.session_state.subtilis_count = subtilis_count
                    st.success(f"✓ Loaded {cereus_count} B. cereus samples")
                    st.success(f"✓ Loaded {subtilis_count} B. subtilis samples")
                    
                    with st.spinner("Training model..."):
                        heal_score, strength_score = st.session_state.predictor.train_model()
                    
                    st.session_state.model_trained = True
                    st.session_state.heal_score = heal_score
                    st.session_state.strength_score = strength_score
                    st.success("✓ Model trained successfully!")
                    st.rerun()
                else:
                    st.error(f"Error loading data: {cereus_count}")
            else:
                st.warning("Please upload both Excel files")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Upload both Excel files
    2. Click 'Train Model'
    3. Enter parameters below
    4. Generate predictions
    """)

# Main content area
if st.session_state.model_trained:
    st.success("✓ Model is ready for predictions!")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Input Parameters")
        
        species = st.selectbox(
            "Select Bacterial Species",
            options=['B. cereus', 'B. subtilis'],
            help="Choose the bacterial species for prediction"
        )
        
        concentration = st.number_input(
            "Concentration (cells/ml)",
            min_value=0.0,
            value=10000000.0,
            step=100000.0,
            format="%.2f",
            help="Enter the bacterial concentration in cells per milliliter"
        )
        
    with col2:
        st.subheader("🔧 Crack Parameters")
        
        initial_crack = st.number_input(
            "Initial Crack Width (mm)",
            min_value=0.0,
            max_value=10.0,
            value=0.50,
            step=0.01,
            format="%.3f",
            help="Enter the initial crack width in millimeters"
        )
        
        final_crack = st.number_input(
            "Final Crack Width (mm)",
            min_value=0.0,
            max_value=10.0,
            value=0.10,
            step=0.01,
            format="%.3f",
            help="Enter the final crack width in millimeters"
        )
    
    # Generate predictions button
    st.markdown("---")
    if st.button("📈 Generate Predictions", key="predict"):
        with st.spinner("Generating predictions..."):
            days, healing_preds, strength_preds = st.session_state.predictor.predict_timeline(
                species, concentration, initial_crack, final_crack
            )
        
        # Display summary metrics
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Healing Efficiency", f"{np.mean(healing_preds):.2f}%")
        with col2:
            st.metric("Max Healing Efficiency", f"{np.max(healing_preds):.2f}%")
        with col3:
            st.metric("Average Strength Gain", f"{np.mean(strength_preds):.2f}%")
        with col4:
            st.metric("Max Strength Gain", f"{np.max(strength_preds):.2f}%")
        
        st.markdown("---")
        
        # Display input parameters used for prediction
        st.markdown("### 🔍 Input Parameters Used")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)
        
        with param_col1:
            st.info(f"**Species**\n\n{species}")
        with param_col2:
            st.info(f"**Concentration**\n\n{concentration:,.2f} cells/ml")
        with param_col3:
            st.info(f"**Initial Crack**\n\n{initial_crack:.3f} mm")
        with param_col4:
            st.info(f"**Final Crack**\n\n{final_crack:.3f} mm")
        
        st.markdown("---")
        
        # Create interactive plots using Plotly
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Healing Efficiency Over Time', 'Compressive Strength Gain Over Time'),
            horizontal_spacing=0.12
        )
        
        # Healing Efficiency plot
        fig.add_trace(
            go.Scatter(
                x=days,
                y=healing_preds,
                mode='lines+markers',
                name='Healing Efficiency',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Compressive Strength Gain plot
        fig.add_trace(
            go.Scatter(
                x=days,
                y=strength_preds,
                mode='lines+markers',
                name='Strength Gain',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Age (days)", row=1, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Age (days)", row=1, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Healing Efficiency (%)", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Compressive Strength Gain (%)", row=1, col=2, gridcolor='lightgray')
        
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("### 📋 Detailed Predictions")
        predictions_df = pd.DataFrame({
            'Day': days,
            'Healing Efficiency (%)': [f"{h:.2f}" for h in healing_preds],
            'Compressive Strength Gain (%)': [f"{s:.2f}" for s in strength_preds]
        })
        st.dataframe(predictions_df, use_container_width=True, height=400)
        
        # Download button
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{species.replace('. ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Please upload the Excel files and train the model using the sidebar to get started.")
    
    # Display example data format
    st.markdown("### 📝 Expected Excel File Format")
    st.markdown("Your Excel files should contain the following columns:")
    
    example_df = pd.DataFrame({
        'Species': ['B. cereus', 'B. cereus'],
        'Concentration (cells/ml)': [10068378.64, 132268.62],
        'Age (days)': [6.84, 6.90],
        'Initial Crack Width (mm)': [0.499, 0.490],
        'Final Crack Width (mm)': [0.054, 0.329],
        'Healing Efficiency (%)': [89.78, 23.67],
        'Compressive Strength Gain (%)': [10.86, 12.31]
    })
    
    st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🦠 Bacteria Self-Healing Concrete Predictor | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)