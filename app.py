import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import joblib
import json
import datetime
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Sistem Prediksi Capstone",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths to model files
ABSENSI_MODEL_PATH = Path('absensi/model/absensi_model.h5')
ABSENSI_SCALER_PATH = Path('absensi/model/scaler.joblib')
ABSENSI_METADATA_PATH = Path('absensi/model/model_metadata.json')

STOK_MODEL_PATH = Path('stok/model/stok_model.h5')
STOK_SCALER_PATH = Path('stok/model/scaler.joblib')
STOK_METADATA_PATH = Path('stok/model/model_metadata.json')

# Function to load absensi model and related files
@st.cache_resource
def load_absensi_model():
    model = models.load_model(ABSENSI_MODEL_PATH)
    scaler = joblib.load(ABSENSI_SCALER_PATH)
    
    with open(ABSENSI_METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        
    return model, scaler, metadata

# Function to load stok model and related files
@st.cache_resource
def load_stok_model():
    model = models.load_model(STOK_MODEL_PATH)
    scaler = joblib.load(STOK_SCALER_PATH)
    
    with open(STOK_METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        
    return model, scaler, metadata

# Helper function for absensi prediction
def prediksi_kehadiran(model, scaler, metadata, hari_string, jam_jadwal, kondisi_cuaca, jam_masuk=None):
    """
    Memprediksi kehadiran menggunakan data cuaca dari database

    Args:
        hari_string (str): Nama hari dalam bahasa Inggris (e.g., 'Monday', 'Tuesday', etc.)
        jam_jadwal (str): Waktu jadwal dalam format "HH:MM"
        kondisi_cuaca (str): Kondisi cuaca ('Clear', 'Clouds', 'Rain', 'Thunderstorm')
        jam_masuk (str, optional): Waktu kedatangan dalam format "HH:MM". 
                                 Jika None, akan menggunakan jam_jadwal untuk simulasi
    """
    # Konversi nama hari ke angka (0-6)
    day_map = metadata['day_map']
    hari = day_map.get(hari_string, 0)  # Default ke Senin jika tidak dikenal

    # Konversi waktu jadwal dari string "HH:MM" ke menit
    if ":" in jam_jadwal:
        parts = jam_jadwal.split(":")
        if len(parts) >= 2:
            jam, menit = int(parts[0]), int(parts[1])
            waktu_jadwal = jam * 60 + menit
        else:
            waktu_jadwal = 0
    else:
        waktu_jadwal = 0

    # Konversi waktu kedatangan
    if jam_masuk and ":" in jam_masuk:
        parts = jam_masuk.split(":")
        if len(parts) >= 2:
            jam, menit = int(parts[0]), int(parts[1])
            waktu_kedatangan = jam * 60 + menit
        else:
            waktu_kedatangan = waktu_jadwal
    else:
        # Jika tidak ada jam_masuk yang valid, gunakan jam_jadwal untuk simulasi
        waktu_kedatangan = waktu_jadwal

    # Menghitung selisih waktu (dalam menit)
    selisih_waktu = waktu_kedatangan - waktu_jadwal

    # Memetakan kondisi cuaca ke kategori yang digunakan saat training
    weather_map = metadata['weather_map']
    cuaca_vector = weather_map.get(kondisi_cuaca, [1, 0, 0])  # Default to Clear if unknown

    # Menyiapkan fitur-fitur dalam urutan yang sama dengan training
    fitur = [
        waktu_jadwal,                    # scheduled_time
        waktu_kedatangan,                # arrival_time 
        hari,                            # day_of_week
        1 if hari == 0 else 0,           # is_monday
        1 if hari == 4 else 0,           # is_friday
        cuaca_vector[0],                 # weather_0
        cuaca_vector[1],                 # weather_1
        cuaca_vector[2]                  # weather_2
    ]

    # Mengubah ke numpy array dan melakukan scaling
    fitur_array = np.array([fitur])
    fitur_scaled = scaler.transform(fitur_array)

    # Mendapatkan probabilitas prediksi menggunakan model
    prediksi = float(model.predict(fitur_scaled, verbose=0)[0][0])

    # Menentukan toleransi berdasarkan cuaca
    toleransi = metadata['tolerances'].get(kondisi_cuaca, 1)

    # Menentukan keterlambatan berdasarkan selisih waktu dan toleransi
    is_terlambat = selisih_waktu > toleransi  # Terlambat jika selisih waktu lebih dari toleransi

    return {
        'probabilitas_prediksi': prediksi,
        'kondisi_cuaca': kondisi_cuaca,
        'toleransi_menit': toleransi,
        'kemungkinan_terlambat': is_terlambat,
        'waktu_jadwal': jam_jadwal,
        'waktu_kedatangan': jam_masuk if jam_masuk else f"{(waktu_kedatangan // 60):02d}:{(waktu_kedatangan % 60):02d}",
        'selisih_menit': selisih_waktu
    }

# Helper function for stok prediction
def prediksi_stok(model, scaler, metadata, nama_barang, stok_awal, masuk, keluar, satuan, bulan):
    """
    Memprediksi risiko kehabisan stok untuk suatu barang
    
    Args:
        nama_barang (str): Nama barang yang akan diprediksi
        stok_awal (int): Jumlah stok awal
        masuk (int): Jumlah barang masuk
        keluar (int): Jumlah barang keluar
        satuan (str): Satuan barang (Ekor/Sachet/Kg/dll)
        bulan (int): Bulan (1-12)
    """
    # Clean up satuan input
    satuan = satuan.strip().capitalize()
    
    # Prepare features
    features = [
        stok_awal,
        masuk,
        keluar,
        masuk - keluar,  # stock_movement
        keluar,  # keluar_ma3 (simplifikasi, seharusnya rata-rata 3 bulan terakhir)
        masuk,   # masuk_ma3 (simplifikasi, seharusnya rata-rata 3 bulan terakhir)
        keluar / stok_awal if stok_awal > 0 else 1.0,  # depletion_rate (set to 1.0 if no stock)
        bulan
    ]
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = float(model.predict(features_scaled, verbose=0)[0][0])
    
    # Calculate estimated days until depletion
    if keluar > 0:
        days_until_depletion = max(1, int(np.ceil(stok_awal / keluar)))
    else:
        days_until_depletion = float('inf')
    
    # Adjusted thresholds based on domain knowledge
    if stok_awal == 0 or prediction > 0.7:
        status = "Berisiko"
    elif prediction > 0.4 or (days_until_depletion != float('inf') and days_until_depletion < 14):
        status = "Stabil"
    else:
        status = "Aman"
    
    # Format output
    return {
        'nama_barang': nama_barang,
        'stok_tersedia': int(stok_awal),
        'satuan': satuan,
        'estimasi_habis': f"{days_until_depletion} Hari" if days_until_depletion != float('inf') else "Stabil",
        'probabilitas': prediction,
        'status': status
    }

# Fungsi untuk halaman About
def show_about():
    st.title("📊 Sistem Prediksi Capstone CC25-CF299")
    st.markdown("""
    ### Selamat datang di aplikasi Sistem Prediksi Capstone!
    
    Aplikasi ini menyediakan dua model prediksi untuk membantu bisnis Anda:
    
    1. **Prediksi Keterlambatan Karyawan** - Memprediksi apakah karyawan akan terlambat berdasarkan jadwal, waktu kedatangan, hari, dan kondisi cuaca. Model ini juga menyediakan toleransi waktu berdasarkan kondisi cuaca.
    
    2. **Prediksi Stok Bahan** - Memprediksi status stok bahan (Aman, Stabil, atau Berisiko) berdasarkan data stok awal, barang masuk, barang keluar, dan periode bulan.
    
    ### Cara Penggunaan
    
    1. Pilih model prediksi yang ingin digunakan dari menu navigasi di samping kiri
    2. Masukkan data yang diperlukan pada form yang tersedia
    3. Klik tombol "Prediksi" untuk mendapatkan hasil
    """)
    
    # Cara kerja model section
    st.subheader("Cara Kerja Model")
    
    # Absensi model explanation
    st.markdown("""
    #### 🕒 Model Absensi
    
    Model prediksi keterlambatan karyawan bekerja dengan mempertimbangkan beberapa faktor penting:
    
    1. **Toleransi Berdasarkan Cuaca:**
       - **Cuaca Cerah/Berawan (Clear/Clouds)**: Toleransi 1 menit
       - **Cuaca Hujan/Badai (Rain/Thunderstorm)**: Toleransi 5 menit
    
    2. **Proses Prediksi:**
       - Model menganalisis jadwal masuk dan waktu kedatangan
       - Mempertimbangkan hari (Senin-Minggu) dengan perhatian khusus pada Senin dan Jumat
       - Memproses kondisi cuaca saat itu
       - Menghitung selisih waktu dan membandingkannya dengan toleransi yang berlaku
    
    3. **Output Prediksi:**
       - **TEPAT WAKTU**: Jika selisih waktu ≤ toleransi yang berlaku
       - **TERLAMBAT**: Jika selisih waktu > toleransi yang berlaku
    """)
    
    # Stok model explanation
    st.markdown("""
    #### 📦 Model Stok
    
    Model prediksi stok mengevaluasi risiko kehabisan stok berdasarkan:
    
    1. **Parameter Kunci:**
       - Stok awal barang
       - Jumlah barang masuk
       - Jumlah barang keluar
       - Bulan (periode) yang relevan
       - Pergerakan stok (masuk - keluar)
       - Tingkat penipisan (depletion rate)
    
    2. **Kriteria Status:**
       - **Berisiko** 🚨: 
         * Stok tersedia = 0, ATAU
         * Probabilitas prediksi model > 0.7, ATAU
         * Estimasi habis kurang dari 7 hari
       
       - **Stabil** ⚠️: 
         * Probabilitas prediksi model > 0.4, ATAU
         * Estimasi habis antara 7-14 hari
       
       - **Aman** ✅: 
         * Probabilitas prediksi model < 0.4, DAN
         * Estimasi habis > 14 hari atau stabil    """)
    st.markdown("""
    ### Tentang Proyek
    
    Proyek ini adalah bagian dari Capstone Project untuk kursus Data Science.
    Dikembangkan menggunakan:
    - TensorFlow untuk pemodelan
    - Streamlit untuk antarmuka web
    - Pemrosesan data dengan Pandas & NumPy
    
    ### Tim Machine Learning Flowlyhub
    
    - **Dewi Safira Permata Sari** - MC009D5X0787
    - **Erisa Putri Nabila** - MC229D5X0818
    - **Firda Humaira** - MC009D5X0441
                
    © 2025 DICODING - DBS CAPSTONE
    """)

# Fungsi untuk halaman Absensi
def show_absensi():
    st.title("🕒 Prediksi Keterlambatan Karyawan")
    
    try:
        # Load model and dependencies
        with st.spinner('Memuat model dan data pendukung...'):
            model, scaler, metadata = load_absensi_model()
        
        # Input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Karyawan")
            nama_karyawan = st.text_input("Nama Karyawan", "Karyawan A")
            
            hari_options = list(metadata['day_map'].keys())
            hari = st.selectbox("Hari", hari_options, index=0)
            
            jam_jadwal = st.time_input("Jadwal Masuk", datetime.time(9, 0))
            jam_jadwal_str = jam_jadwal.strftime("%H:%M")
            
            # Menggunakan text_input untuk waktu kedatangan agar lebih presisi
            jam_masuk_str = st.text_input("Waktu Kedatangan (HH:MM)", "09:00")
            
            # Validasi format waktu
            try:
                datetime.datetime.strptime(jam_masuk_str, "%H:%M")
                waktu_valid = True
            except ValueError:
                st.error("Format waktu harus HH:MM (contoh: 09:05)")
                waktu_valid = False
        
        with col2:
            st.subheader("Kondisi Cuaca")
            cuaca_options = list(metadata['weather_map'].keys())
            cuaca = st.selectbox("Kondisi Cuaca", cuaca_options, index=0)
            
            # Display weather tolerance
            toleransi = metadata['tolerances'].get(cuaca, 1)
            st.info(f"Toleransi keterlambatan untuk cuaca {cuaca}: {toleransi} menit")
            
            # Calculate time difference
            try:
                jam_jadwal_dt = datetime.datetime.strptime(jam_jadwal_str, "%H:%M")
                jam_masuk_dt = datetime.datetime.strptime(jam_masuk_str, "%H:%M")
                selisih = (jam_masuk_dt - jam_jadwal_dt).total_seconds() / 60
                
                st.metric("Selisih Waktu", f"{int(selisih)} menit", 
                         delta=f"{int(selisih)}" if selisih != 0 else "0",
                         delta_color="inverse")
            except ValueError:
                # Jika format waktu tidak valid, tampilkan pesan error dan set selisih ke 0
                st.metric("Selisih Waktu", "Error format waktu", delta="0", delta_color="inverse")
        
        # Add prediction button
        if st.button("Prediksi Keterlambatan"):
            # Periksa validitas format waktu sebelum memprediksi
            try:
                datetime.datetime.strptime(jam_masuk_str, "%H:%M")
                with st.spinner('Memproses...'):
                    hasil_prediksi = prediksi_kehadiran(
                        model, scaler, metadata,
                        hari_string=hari,
                        jam_jadwal=jam_jadwal_str,
                        kondisi_cuaca=cuaca,
                        jam_masuk=jam_masuk_str
                    )
                
                # Display results
                st.subheader("Hasil Prediksi")
                
                # Create three columns for the result
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # Display status with color
                with col1:
                    status = "TERLAMBAT" if hasil_prediksi['kemungkinan_terlambat'] else "TEPAT WAKTU"
                    status_color = "red" if hasil_prediksi['kemungkinan_terlambat'] else "green"
                    st.markdown(f"<h1 style='text-align: center; color: {status_color};'>{status}</h1>", unsafe_allow_html=True)
                    
                    # Display confidence percentage
                    confidence = hasil_prediksi['probabilitas_prediksi'] * 100
                    st.progress(min(confidence, 100) / 100)
                    st.text(f"Confidence: {confidence:.1f}%")
                
                # Weather info
                with col2:
                    st.markdown("### Detail Cuaca")
                    st.markdown(f"**Kondisi:** {hasil_prediksi['kondisi_cuaca']}")
                    st.markdown(f"**Toleransi:** {hasil_prediksi['toleransi_menit']} menit")
                    
                # Time details
                with col3:
                    st.markdown("### Detail Waktu")
                    st.markdown(f"**Jadwal Masuk:** {hasil_prediksi['waktu_jadwal']}")
                    st.markdown(f"**Waktu Kedatangan:** {hasil_prediksi['waktu_kedatangan']}")
                    st.markdown(f"**Selisih:** {hasil_prediksi['selisih_menit']} menit")
                    
            except ValueError:
                st.error("Format waktu kedatangan tidak valid. Gunakan format HH:MM (contoh: 09:05)")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.error("Pastikan lokasi file model benar dan model tersedia.")

# Fungsi untuk halaman Stok
def show_stok():
    st.title("📦 Prediksi Stok Bahan")
    
    try:
        # Load model and dependencies
        with st.spinner('Memuat model dan data pendukung...'):
            model, scaler, metadata = load_stok_model()
        
        # Input fields
        st.subheader("Data Stok")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nama_barang = st.text_input("Nama Barang", "Ikan Nila")
            stok_awal = st.number_input("Stok Awal", min_value=0, value=100)
            masuk = st.number_input("Barang Masuk", min_value=0, value=50)
            
        with col2:
            keluar = st.number_input("Barang Keluar", min_value=0, value=30)
            satuan = st.text_input("Satuan", "Ekor")
            bulan = st.slider("Bulan", min_value=1, max_value=12, value=datetime.datetime.now().month)
        
        # Add prediction button
        if st.button("Prediksi Status Stok"):
            with st.spinner('Memproses...'):
                hasil_prediksi = prediksi_stok(
                    model, scaler, metadata,
                    nama_barang=nama_barang,
                    stok_awal=stok_awal,
                    masuk=masuk,
                    keluar=keluar,
                    satuan=satuan,
                    bulan=bulan
                )
            
            # Display results
            st.subheader("Hasil Prediksi")
            
            # Status indicator
            status = hasil_prediksi['status']
            if status == "Aman":
                status_color = "green"
                emoji = "✅"
            elif status == "Stabil":
                status_color = "orange"
                emoji = "⚠️"
            else:  # Berisiko
                status_color = "red"
                emoji = "🚨"
                
            # Create three columns for the result
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Status
            with col1:
                st.markdown(f"<h1 style='text-align: center; color: {status_color};'>{emoji} {status}</h1>", 
                           unsafe_allow_html=True)
                
                # Display confidence percentage
                confidence = hasil_prediksi['probabilitas'] * 100
                st.progress(min(confidence, 100) / 100)
                st.text(f"Confidence: {confidence:.1f}%")
            
            # Stock details
            with col2:
                st.markdown("### Detail Stok")
                st.markdown(f"**Nama Barang:** {hasil_prediksi['nama_barang']}")
                st.markdown(f"**Stok Tersedia:** {hasil_prediksi['stok_tersedia']} {hasil_prediksi['satuan']}")
                
            # Estimation
            with col3:
                st.markdown("### Estimasi")
                st.markdown(f"**Estimasi Habis:** {hasil_prediksi['estimasi_habis']}")
                st.markdown(f"**Pergerakan:** +{masuk} / -{keluar} {hasil_prediksi['satuan']}")
                st.markdown(f"**Sisa Akhir:** {stok_awal + masuk - keluar} {hasil_prediksi['satuan']}")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.error("Pastikan lokasi file model benar dan model tersedia.")

# Main function
def main():
    # Add sidebar styling
    st.markdown(
        """
        <style>
        /* Styling the sidebar navigation */
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100%;
            border: none;
            border-radius: 5px;
            margin-bottom: 10px;
            padding: 10px 15px;
            text-align: left;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        /* Hover effect for buttons */
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #e1e5f2;
            color: #1e3c72;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
      # Add a sidebar title
    st.sidebar.title("Navigasi")
    
    # Use session_state to manage the current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "About"
    
    # Create sidebar navigation with icons
    if st.sidebar.button("📚 About", 
                       type="primary" if st.session_state.current_page == "About" else "secondary",
                       use_container_width=True):
        st.session_state.current_page = "About"
        st.rerun()
    
    if st.sidebar.button("🕒 Absensi", 
                       type="primary" if st.session_state.current_page == "Absensi" else "secondary",
                       use_container_width=True):
        st.session_state.current_page = "Absensi"
        st.rerun()
    
    if st.sidebar.button("📦 Stok", 
                       type="primary" if st.session_state.current_page == "Stok" else "secondary",
                       use_container_width=True):
        st.session_state.current_page = "Stok"
        st.rerun()
    
    # Get current page from session state
    current_page = st.session_state.current_page
    
    # Display the selected page
    if current_page == "About":
        show_about()
    elif current_page == "Absensi":
        show_absensi()
    elif current_page == "Stok":
        show_stok()
          # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tim Machine Learning Flowlyhub")
    st.sidebar.markdown("- **Dewi Safira Permata Sari**  \nMC009D5X0787")
    st.sidebar.markdown("- **Erisa Putri Nabila**  \nMC229D5X0818")
    st.sidebar.markdown("- **Firda Humaira**  \nMC009D5X0441")
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 DICODING - DBS CAPSTONE")

if __name__ == "__main__":
    main()
