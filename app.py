import cv2
import streamlit as st
from ultralytics import YOLO

# Load model
model = YOLO('E:/Data Mining/a6/best3.pt')

# Streamlit UI
st.set_page_config(page_title="SIBI Recognition App", page_icon="🔍")
st.title("🖐 Aplikasi Bahasa Isyarat SIBI dengan YOLOv8")
st.markdown(
    """Aplikasi ini dapat mengenali gestur tangan dengan kata:
- **Aku Cinta Kamu**
- **Halo**
- **Iya**
"""
)
st.info("🚀 Tekan **Start** untuk memulai deteksi dan **Stop** untuk menghentikan deteksi.")

# State management untuk kontrol start/stop
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Sidebar: Tentang aplikasi
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.markdown(
        """**Aplikasi Bahasa Isyarat SIBI dengan YOLOv8** dirancang untuk mendeteksi dan mengenali gestur tangan dalam bahasa isyarat Indonesia.

### Fitur Utama
- Deteksi gestur tangan secara real-time.

### Cara Penggunaan
1. Tekan tombol **Start** untuk memulai deteksi.
2. Tekan tombol **Stop** untuk menghentikan deteksi.
3. Pastikan kamera memiliki pencahayaan yang baik dan tangan terlihat jelas.
""")

# Layout utama untuk kontrol
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    start_button = st.button("▶️ Start")

with col3:
    stop_button = st.button("⏹️ Stop")

# Placeholder untuk tampilan video
frame_window = st.empty()

# Tombol Start: Mulai deteksi
if start_button:
    st.session_state.is_running = True

# Tombol Stop: Hentikan deteksi
if stop_button:
    st.session_state.is_running = False
    
# Placeholder untuk tampilan video (diletakkan di tengah)
center_frame = st.container()

# Tombol Start: Mulai deteksi
if start_button:
    st.session_state.is_running = True

# Tombol Stop: Hentikan deteksi
if stop_button:
    st.session_state.is_running = False

# Proses deteksi
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)  # Akses kamera lokal
    with center_frame:
        st.write("### Tampilan Kamera")
        frame_window = st.empty()  # Placeholder video

    while cap.isOpened() and st.session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("\u274c Tidak dapat membaca frame dari kamera.")
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

