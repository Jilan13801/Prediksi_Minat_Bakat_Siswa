import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

st.set_page_config(page_title="Prediksi Minat & Bakat", page_icon="🎓", layout="wide")

# --- Load & Encode Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data_minat_bakat_100.csv")
    df_no_nama = df.drop("Nama", axis=1)
    le_dict = {}
    for col in df_no_nama.columns:
        le = LabelEncoder()
        df_no_nama[col] = le.fit_transform(df_no_nama[col])
        le_dict[col] = le
    return df, df_no_nama, le_dict

df_raw, df_encoded, le_dict = load_data()

# Train model
X = df_encoded.drop("Minat_Bakat", axis=1)
y = df_encoded["Minat_Bakat"]
model = CategoricalNB()
model.fit(X, y)

# --- Sidebar ---
st.sidebar.title("🎯 Menu")
menu = st.sidebar.radio("Pilih Halaman", ["📊 Analisis Data", "🧠 Prediksi Siswa"])

# --- 📊 ANALISIS DATA ---
if menu == "📊 Analisis Data":
    st.title("📊 Analisis Data Minat & Bakat Siswa")
    st.markdown("Visualisasi distribusi berdasarkan data input yang tersedia.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Jumlah Siswa per Minat & Bakat")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Minat_Bakat", data=df_raw, palette="pastel", ax=ax1)
        plt.xticks(rotation=15)
        st.pyplot(fig1)

    with col2:
        st.subheader("Distribusi Gaya Belajar")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Gaya_Belajar", data=df_raw, palette="muted", ax=ax2)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Data Mentah")
    st.dataframe(df_raw)

# --- 🧠 PREDIKSI ---
elif menu == "🧠 Prediksi Siswa":
    st.title("🧠 Prediksi Minat & Bakat Siswa")
    st.markdown("Masukkan data siswa untuk memprediksi kejuruan yang sesuai.")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            gaya_belajar = st.selectbox("Gaya Belajar", le_dict["Gaya_Belajar"].classes_)
            kegiatan = st.selectbox("Kegiatan Favorit", le_dict["Kegiatan_Favorit"].classes_)

        with col2:
            nilai_rata2 = st.selectbox("Nilai Rata-rata", le_dict["Nilai_Rata2"].classes_)
            kepribadian = st.selectbox("Tes Kepribadian", le_dict["Tes_Kepribadian"].classes_)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        input_df = pd.DataFrame([{
            "Gaya_Belajar": gaya_belajar,
            "Kegiatan_Favorit": kegiatan,
            "Nilai_Rata2": nilai_rata2,
            "Tes_Kepribadian": kepribadian
        }])

        for col in input_df.columns:
            input_df[col] = le_dict[col].transform(input_df[col])

        pred = model.predict(input_df)
        label = le_dict["Minat_Bakat"].inverse_transform(pred)

        st.success(f"🎓 **Rekomendasi Minat & Bakat:** {label[0]}")