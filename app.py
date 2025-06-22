import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Prediksi Minat & Bakat", page_icon="🎓", layout="wide")

# --- (Opsional) CSS pelembut padding layout ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# -- Render fix (untuk sidebar toggle) --
st.empty()

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

# Split data
X = df_encoded.drop("Minat_Bakat", axis=1)
y = df_encoded["Minat_Bakat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CategoricalNB()
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
laporan_klasifikasi = classification_report(y_test, y_pred, output_dict=True)

# --- Sidebar ---
st.sidebar.title("🎯 Menu")
menu = st.sidebar.radio("Pilih Halaman", ["📊 Analisis Data", "🧠 Prediksi Siswa"])
st.sidebar.caption(f"🧾 Halaman Aktif: {menu}")

# --- Rerun Aman untuk Layout Penuh ---
if "last_menu" not in st.session_state:
    st.session_state.last_menu = menu
if "has_rerun" not in st.session_state:
    st.session_state.has_rerun = False

if menu != st.session_state.last_menu and not st.session_state.has_rerun:
    st.session_state.last_menu = menu
    st.session_state.has_rerun = True
    st.rerun()
else:
    st.session_state.last_menu = menu
    st.session_state.has_rerun = False

# --- 📊 ANALISIS DATA ---
if menu == "📊 Analisis Data":
    st.title("📊 Analisis Data Minat & Bakat Siswa")
    st.markdown("Visualisasi distribusi berdasarkan data input yang tersedia.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Jumlah Siswa per Minat & Bakat")
        fig1 = px.histogram(
            df_raw,
            x="Minat_Bakat",
            color="Minat_Bakat",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Distribusi Gaya Belajar")
        fig2 = px.histogram(
            df_raw,
            x="Gaya_Belajar",
            color="Gaya_Belajar",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Evaluasi Model (Naive Bayes)")
    st.write(f"**Akurasi Model:** {akurasi:.2%}")

    with st.expander("Lihat Detail Laporan Klasifikasi"):
        st.dataframe(pd.DataFrame(laporan_klasifikasi).transpose())

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
