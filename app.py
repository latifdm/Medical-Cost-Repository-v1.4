import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. Streamlit config
st.set_page_config(
    page_title="Medical Cost Prediction", 
    page_icon="💊", 
    layout="centered"
)

# 2. Load model with error handling
@st.cache_resource
def load_model():
    try:
        with open('gradient_boosting_regressor_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'gradient_boosting_regressor_model.pkl' tidak ditemukan!")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("💡 Tip: Model mungkin dilatih dengan versi scikit-learn yang berbeda. Coba retrain model dengan versi scikit-learn yang sama.")
        return None

# Load model once
model = load_model()

# --- Utility Functions ---
def calculate_bmi(height, weight):
    """Calculate BMI with input validation"""
    if height <= 0 or weight <= 0:
        return 0
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

def get_bmi_category(bmi):
    """Get BMI category for better user understanding"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def preprocess_input(age, bmi, children, sex, smoker, region) -> pd.DataFrame:
    """Preprocess input data for model prediction"""
    cols = [
        "age",
        "bmi", 
        "children",
        "sex_male",
        "smoker_yes",
        "region_northeast",
        "region_northwest", 
        "region_southeast",
        "region_southwest",
    ]

    data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "Pria" else 0,
        "smoker_yes": 1 if smoker == "Ya" else 0,
        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    input_data_for_df = {col: data[col] for col in cols}
    return pd.DataFrame([input_data_for_df])[cols]

# --- Sidebar navigation ---
with st.sidebar:
    st.markdown("### 🧭 Menu")
    page = st.selectbox(
        label="Navigasi",
        options=["Home", "Machine Learning App", "Dashboard"],
        index=0,
        label_visibility="collapsed"
    )

# ----------------------------------------------------------------------
# 🏠 PAGE — Home
# ----------------------------------------------------------------------
if page == "Home":
    st.title("💊 Medical Cost Predictor App")
    
    st.markdown("""
    ### 🎯 Tentang Aplikasi
    Aplikasi Machine Learning ini dibuat untuk memprediksi biaya medis tahunan pasien 
    berdasarkan model **Gradient Boosting Regressor** yang telah dilatih sebelumnya.
    
    ### 📊 Fitur Utama:
    - **Prediksi Biaya Medis**: Input data pasien untuk mendapatkan estimasi biaya
    - **Dashboard Analitik**: Visualisasi data dan insights statistik
    - **BMI Calculator**: Kalkulasi otomatis Body Mass Index
    """)
    
    st.info("📖 **Data Source**: [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)")

    st.subheader("👨‍⚕️ Delta Seekers Team")
    
    # Team members with better layout
    members = [
        {"name": "Ahmad Azhar Naufal Farizky", "role": "ML Engineer"},
        {"name": "Kristina Sarah Yuliana", "role": "Data Analyst"}, 
        {"name": "Latif Dwi Mardani", "role": "Backend Developer"},
        {"name": "Jalu Prayoga", "role": "Frontend Developer"},
        {"name": "Ayasha Naila Ismunandar", "role": "UI/UX Designer"},
    ]

    for member in members:
        st.markdown(f"**{member['name']}** - _{member['role']}_")

# ----------------------------------------------------------------------
# 🤖 PAGE — Machine Learning App
# ----------------------------------------------------------------------
elif page == "Machine Learning App":
    st.title("💊 Medical Cost Predictor App")
    
    # Check if model is loaded
    if model is None:
        st.error("❌ Model tidak tersedia. Silakan periksa file model.")
        st.stop()
    
    st.markdown("### 📝 Masukkan informasi pasien untuk memprediksi **biaya medis tahunan**")

    # Better input layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Informasi Pribadi")
        age = st.slider("Usia", 18, 100, 30, help="Usia pasien dalam tahun")
        sex = st.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
        children = st.selectbox("Jumlah Anak", list(range(0, 6)), index=0, 
                               help="Jumlah anak yang ditanggung asuransi")
    
    with col2:
        st.subheader("🏥 Informasi Kesehatan")
        smoker = st.selectbox('Apakah Merokok?', ('Tidak', 'Ya'))
        height = st.number_input('Tinggi Badan (cm)', min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input('Berat Badan (kg)', min_value=30.0, max_value=200.0, value=70.0)
    
    st.subheader("📍 Lokasi")
    region = st.selectbox('Lokasi Tinggal', 
                         ("northeast", "northwest", "southeast", "southwest"),
                         help="Region tempat tinggal di Amerika Serikat")

    # Calculate and display BMI
    bmi = calculate_bmi(height, weight)
    bmi_category = get_bmi_category(bmi)
    
    col_bmi1, col_bmi2 = st.columns(2)
    with col_bmi1:
        st.metric("BMI", f"{bmi:.1f}")
    with col_bmi2:
        st.metric("Kategori BMI", bmi_category)

    # Prediction button
    if st.button("🔮 Predict Medical Cost", type="primary", use_container_width=True):
        
        # Input validation
        if height <= 0 or weight <= 0:
            st.error("⚠️ Tinggi dan berat badan harus lebih dari 0!")
            st.stop()
            
        if bmi == 0:
            st.error("⚠️ BMI tidak valid!")
            st.stop()

        try:
            input_df = preprocess_input(age, bmi, children, sex, smoker, region)

            with st.spinner("🔄 Menghitung prediksi..."):
                prediction = model.predict(input_df)[0]

            # Results section
            st.success("✅ Prediksi berhasil!")
            
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.metric("💵 Estimasi Biaya Medis Tahunan", f"${prediction:,.2f}")
            
            with col_result2:
                # Risk assessment based on prediction
                if prediction > 20000:
                    risk_level = "🔴 Tinggi"
                    risk_color = "red"
                elif prediction > 10000:
                    risk_level = "🟡 Sedang"
                    risk_color = "orange"
                else:
                    risk_level = "🟢 Rendah"
                    risk_color = "green"
                
                st.metric("📊 Level Risiko Biaya", risk_level)

            # Additional insights
            st.subheader("💡 Insights")
            insights = []
            
            if smoker == "Ya":
                insights.append("🚬 Status perokok significantly increases medical costs")
            if bmi >= 30:
                insights.append("⚖️ BMI dalam kategori obesitas dapat meningkatkan biaya medis")
            if age >= 50:
                insights.append("👴 Usia di atas 50 tahun umumnya memiliki biaya medis lebih tinggi")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.success("✅ Profil risiko relatif rendah!")

            # Show input details
            with st.expander("📋 Detail Input Data"):
                input_display = input_df.copy()
                input_display.columns = [
                    "Usia", "BMI", "Jumlah Anak", "Pria", "Perokok", 
                    "Northeast", "Northwest", "Southeast", "Southwest"
                ]
                st.dataframe(input_display, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {str(e)}")
            st.info("💡 Pastikan semua input sudah benar dan model telah diload dengan proper.")

# ----------------------------------------------------------------------
# 📊 PAGE — Dashboard  
# ----------------------------------------------------------------------
elif page == "Dashboard":
    st.title("📊 Medical Cost Dashboard")
    st.markdown("### 📈 Analisis data dan visualisasi statistik pasien")

    try:
        df = pd.read_csv("insurance.csv")
        
        # Basic info
        st.subheader("ℹ️ Informasi Dataset")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Total Pasien", len(df))
        with col_info2:
            st.metric("Rata-rata Biaya", f"${df['charges'].mean():,.0f}")
        with col_info3:
            st.metric("Fitur", len(df.columns))

        # Statistics summary
        st.subheader("📊 Ringkasan Statistik")
        st.dataframe(df.describe(), use_container_width=True)

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # BMI Distribution
        if 'bmi' in df.columns:
            sns.histplot(df["bmi"], kde=True, ax=axes[0,0], color='skyblue')
            axes[0,0].set_title("Distribusi BMI")
            axes[0,0].set_xlabel("BMI")
        
        # Smoker comparison
        if 'smoker' in df.columns:
            smoker_counts = df["smoker"].value_counts()
            axes[0,1].bar(smoker_counts.index, smoker_counts.values, color=['lightgreen', 'salmon'])
            axes[0,1].set_title('Perokok vs Non-Perokok')
            axes[0,1].set_ylabel('Jumlah')
        
        # Age vs Charges
        if 'age' in df.columns and 'charges' in df.columns:
            sns.scatterplot(x="age", y="charges", data=df, ax=axes[1,0], alpha=0.6)
            axes[1,0].set_title("Biaya Medis berdasarkan Usia")
        
        # Children per region
        if 'region' in df.columns and 'children' in df.columns:
            children_region = df.groupby("region")["children"].sum()
            axes[1,1].bar(children_region.index, children_region.values, color='lightcoral')
            axes[1,1].set_title('Total Anak per Region')
            axes[1,1].set_ylabel('Jumlah Anak')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

        # Additional visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("🔥 Korelasi Fitur Numerik")
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax_corr, center=0)
            st.pyplot(fig_corr)
        
        with col_viz2:
            st.subheader("💰 Sebaran Biaya per Region")
            fig_box, ax_box = plt.subplots(figsize=(8, 6))
            sns.boxplot(x="region", y="charges", data=df, ax=ax_box)
            ax_box.tick_params(axis='x', rotation=45)
            st.pyplot(fig_box)

        # Age distribution
        st.subheader("📈 Distribusi Usia Pasien")
        fig_age, ax_age = plt.subplots(figsize=(10, 5))
        sns.histplot(df["age"], bins=15, kde=True, ax=ax_age, color="skyblue", alpha=0.7)
        ax_age.set_title("Distribusi Usia")
        ax_age.set_xlabel("Usia")
        st.pyplot(fig_age)

        # Region proportion pie chart
        st.subheader("🗺️ Proporsi Pasien per Region")
        region_counts = df["region"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        ax_pie.pie(region_counts, labels=region_counts.index, autopct="%1.1f%%", 
                  startangle=90, colors=colors)
        ax_pie.axis("equal")
        st.pyplot(fig_pie)
        
    except FileNotFoundError:
        st.error("⚠️ File 'insurance.csv' tidak ditemukan!")
        st.info("💡 Pastikan file dataset berada di folder yang sama dengan aplikasi.")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "💊 Medical Cost Predictor App | Built with Streamlit | Delta Seekers Team"
    "</div>", 
    unsafe_allow_html=True
)
