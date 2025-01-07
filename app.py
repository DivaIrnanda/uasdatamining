import cv2
import streamlit as st
from ultralytics import YOLO

# Load model
model = YOLO('E:/Data Mining/a6/best3.pt')

# Streamlit UI
st.title("Aplikasi Bahasa Isyarat ASL dengan YOLOv8")
st.text("Aplikasi ini mengenali gestur tangan dengan kata ('Aku Cinta Kamu', 'Halo', 'Iya').")
st.text("Tekan 'Start' untuk memulai deteksi. Tekan 'Stop' untuk menghentikan deteksi.")

# Streamlit state management untuk kontrol start/stop
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Tombol untuk memulai deteksi
start_button = st.button("Start")
stop_button = st.button("Stop")

# Placeholder untuk tampilan video
frame_window = st.empty()

# Tombol Start: Mulai deteksi
if start_button:
    st.session_state.is_running = True

# Tombol Stop: Hentikan deteksi
if stop_button:
    st.session_state.is_running = False

# Proses deteksi
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)  # Akses kamera lokal
    while cap.isOpened() and st.session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak dapat membaca frame dari kamera.")
            break

        # Prediksi dengan YOLOv8
        results = model(frame)
        annotated_frame = results[0].plot()  # Tambahkan bounding box pada frame

        # Konversi BGR ke RGB untuk Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Tampilkan frame di Streamlit
        frame_window.image(rgb_frame, channels="RGB")

        # Hentikan jika tombol 'Stop' ditekan
        if not st.session_state.is_running:
            break

    cap.release()
    cv2.destroyAllWindows()

st.text("Deteksi selesai.")

# About section
st.sidebar.title("About")
st.sidebar.info("""
**Aplikasi Bahasa Isyarat ASL dengan YOLOv8**  
Aplikasi ini dirancang untuk mendeteksi dan mengenali gestur tangan bahasa isyarat Amerika (ASL) dengan menggunakan model YOLOv8 yang dilatih khusus.

### Fitur
- Deteksi gestur tangan:  
  - **Aku Cinta Kamu**  
  - **Halo**  
  - **Iya**
- Pemrosesan video langsung dari kamera untuk deteksi secara real-time.


### Penggunaan
1. Tekan tombol **Start** untuk memulai deteksi.  
2. Tekan tombol **Stop** untuk menghentikan deteksi.  
3. Hasil deteksi akan ditampilkan secara real-time.

### Catatan
Model YOLO dilatih pada dataset khusus untuk bahasa isyarat ASL. Untuk hasil optimal, pastikan kamera memiliki pencahayaan yang baik dan tangan terlihat jelas dalam frame.

""")
