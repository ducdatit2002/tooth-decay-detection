import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# Tải mô hình YOLOv8
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Đảm bảo bạn đã có file best.pt trong thư mục làm việc

model = load_model()

# Tiêu đề của ứng dụng
st.title("Ứng Dụng Phát Hiện Sâu Răng với YOLOv8")

# Mô tả ngắn
st.write("""
Tải lên hình ảnh hoặc chụp hình bằng camera để hệ thống tự động phát hiện sâu răng và hiển thị các bounding boxes.
""")

# Tùy chọn tải lên hoặc chụp hình
option = st.radio("Chọn phương thức nhập hình ảnh:", ("Tải lên từ máy tính", "Chụp hình bằng camera"))

uploaded_file = None

if option == "Tải lên từ máy tính":
    uploaded_file = st.file_uploader("Chọn hình ảnh", type=["jpg", "jpeg", "png"])
elif option == "Chụp hình bằng camera":
    camera_file = st.camera_input("Chụp hình bằng camera")
    if camera_file:
        uploaded_file = camera_file

if uploaded_file is not None:
    # Hiển thị hình ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Hình ảnh đã tải lên', use_container_width=True)

    # Phát hiện đối tượng
    with st.spinner("Đang phát hiện sâu răng..."):
        results = model.predict(image)

    # Lấy kết quả đầu tiên
    result = results[0]

    # Tạo đối tượng vẽ trên hình
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Danh sách bounding boxes
    boxes = []

    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        class_id = int(box.cls[0].item())
        confidence = round(box.conf[0].item(), 2)

        # Nếu chỉ phát hiện lớp 0 (sâu răng)
        if class_id != 0:
            continue

        label = result.names[class_id]
        boxes.append([x1, y1, x2, y2, label, confidence])

        # Vẽ bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Vẽ nhãn và độ tin cậy
        text = f"{label} {confidence}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="red")
        draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)

    # Hiển thị hình ảnh với bounding boxes
    st.image(image, caption='Hình ảnh sau khi phát hiện', use_container_width=True)

    # Hiển thị bảng kết quả
    if boxes:
        st.subheader("Kết Quả Phát Hiện")
        # Tạo DataFrame để hiển thị bảng đẹp hơn
        import pandas as pd
        df = pd.DataFrame(boxes, columns=["X1", "Y1", "X2", "Y2", "Loại Đối Tượng", "Độ Tin Cậy"])
        st.table(df)
    else:
        st.write("Không phát hiện sâu răng nào trong hình ảnh này.")
