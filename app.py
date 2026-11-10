import os
import io
import cv2
import gdown
import numpy as np
from PIL import Image
import qrcode
import streamlit as st
from ultralytics import YOLO

# Import cơ sở dữ liệu bệnh và phác đồ điều trị từ file disease_data.py
from disease_data import DISEASE_DATABASE

# --- 1. CẤU HÌNH TRANG WEB STREAMLIT ---
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Thanh Long",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Ứng dụng AI Chẩn Đoán Bệnh Thanh Long & Phác Đồ Điều Trị")
st.write(
    "Vui lòng tải lên hình ảnh cành thanh long. Hệ thống sẽ tự động phân tích, "
    "khoanh vùng khung hình chữ nhật vết bệnh và cung cấp phác đồ điều trị chi tiết."
)

# --- 2. ÁNH XẠ CHÍNH XÁC TÊN LỚP MÔ HÌNH YOLOV11 ---
RAW_TO_VIETNAMESE = {
    'chay_nang': 'Cháy nắng',
    'dom_than': 'Đốm thân',
    'than_thu': 'Thán thư',
    'thoi_nhun': 'Thối nhũn'
}

NAME_TO_KEY = {
    'Cháy nắng': 'chay_nang',
    'Đốm thân': 'dom_than',
    'Thán thư': 'than_thu',
    'Thối nhũn': 'thoi_nhun'
}

# Bảng màu BGR chuẩn cho từng lớp bệnh
CLASS_COLORS = {
    'chay_nang': (0, 165, 255),  # Cam
    'dom_than': (255, 195, 0),   # Xanh da trời
    'than_thu': (50, 50, 255),   # Đỏ
    'thoi_nhun': (0, 215, 255)   # Vàng
}

# --- 3. TẢI MÔ HÌNH YOLOV11 (TỰ ĐỘNG TẢI TỪ GOOGLE DRIVE NẾU THIẾU) ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        file_id = "1Pa6V_-ZJd-wIw0_iNyez77SK-WwMyQ8t"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return YOLO(model_path)

model = load_model()

# --- 4. THANH BÊN (SIDEBAR) ĐỂ TẠO MÃ QR TRUY CẬP DI ĐỘNG ---
st.sidebar.header("📱 Truy cập qua Điện thoại")
st.sidebar.write("Nhập URL ứng dụng của bạn (Streamlit Cloud hoặc IP mạng nội bộ LAN):")
app_url = st.sidebar.text_input(
    "Nhập hoặc sửa lại URL bên dưới:", 
    "http://172.20.10.3:8505"
)

if app_url:
    qr = qrcode.QRCode(version=1, box_size=6, border=2)
    qr.add_data(app_url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.sidebar.image(byte_im, caption="Quét mã QR để mở trên điện thoại", use_container_width=True)

# --- 5. KHU VỰC CHÍNH: TẢI ẢNH VÀ PHÂN TÍCH ---
uploaded_file = st.file_uploader("Chọn ảnh thanh long (định dạng jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Ảnh gốc tải lên")
        st.image(image, use_container_width=True)
    
    if st.button("🔍 Phân tích hình ảnh", type="primary"):
        with st.spinner("AI đang phân tích vết bệnh và khoanh vùng..."):
            img_array_rgb = np.array(image)
            img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
            
            CONF_THRESHOLD = 0.35
            results = model.predict(img_array_bgr, conf=CONF_THRESHOLD, iou=0.45)
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                with col2:
                    st.subheader("🎯 Kết quả nhận diện")
                    st.image(image, use_container_width=True)
                st.success("✅ Không phát hiện vết bệnh rõ ràng nào trên cành/quả thanh long (Cây khỏe mạnh).")
            else:
                annotated_img = img_array_bgr.copy()
                detected_labels = []

                FONT_SCALE = 1.0       # Kích thước chữ to nét cho di động
                FONT_THICKNESS = 2   # Độ đậm nét chữ
                BOX_THICKNESS = 3    # Độ dày khung hình chữ nhật

                for box in result.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < CONF_THRESHOLD:
                        continue

                    cls_id = int(box.cls[0].cpu().numpy())
                    raw_label = model.names[cls_id]
                    vn_label = RAW_TO_VIETNAMESE.get(raw_label, raw_label)
                    color = CLASS_COLORS.get(raw_label, (0, 255, 0))

                    detected_labels.append(vn_label)

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # 1. Chỉ vẽ duy nhất khung hình chữ nhật Bounding Box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

                    # 2. Nội dung & Tính toán khối nền chữ
                    label_text = f"{vn_label} {conf:.2f}"
                    (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
                    
                    text_y = y1 - 10 if (y1 - h - 15) > 0 else y1 + h + 10
                    rect_y1 = y1 - h - 15 if (y1 - h - 15) > 0 else y1
                    rect_y2 = y1 if (y1 - h - 15) > 0 else y1 + h + 15

                    cv2.rectangle(annotated_img, (x1, rect_y1), (x1 + w + 10, rect_y2), color, -1)
                    cv2.putText(
                        annotated_img, 
                        label_text, 
                        (x1 + 5, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        FONT_SCALE, 
                        (255, 255, 255), 
                        FONT_THICKNESS, 
                        cv2.LINE_AA
                    )

                unique_labels = list(dict.fromkeys(detected_labels))
                res_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("🎯 Kết quả khoanh vùng vết bệnh")
                    st.image(res_rgb, use_container_width=True)
                
                st.error(f"🚨 Phát hiện (các) bệnh: **{', '.join(unique_labels)}**")
                
                # --- 6. THÔNG TIN CHI TIẾT & PHÁC ĐỒ ĐIỀU TRỊ ---
                st.markdown("---")
                st.header("📋 THÔNG TIN CHI TIẾT & PHÁC ĐỒ ĐIỀU TRỊ")
                
                displayed_keys = set()
                for label in unique_labels:
                    db_key = NAME_TO_KEY.get(label)
                    if db_key and db_key in DISEASE_DATABASE and db_key not in displayed_keys:
                        displayed_keys.add(db_key)
                        info = DISEASE_DATABASE[db_key]
                        d_name = info.get('display_name', label)
                        severity = info.get('severity', 'Cảnh báo')
                        
                        st.error(f"🔴 **Phát hiện bệnh: {d_name}** (Mức độ nguy hại: {severity})")
                        st.markdown(f"**📖 Mô tả triệu chứng:** {info.get('description', '')}")
                        st.markdown(f"**🔬 Nguyên nhân chính:** {info.get('causes', '')}")
                        
                        st.markdown("**🛠️ Hướng xử lý và phác đồ điều trị đề xuất:**")
                        for idx, step in enumerate(info.get('treatment', []), 1):
                            st.markdown(f"{idx}. {step}")
                        
                        st.markdown("---")
