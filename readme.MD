# Dự Án Nhận Diện Sâu Răng

## Giới Thiệu

**Dự án Nhận Diện Sâu Răng** sử dụng mô hình YOLOv8 để phát hiện sâu răng từ hình ảnh. Người dùng có thể tải lên hình ảnh từ máy tính hoặc chụp hình trực tiếp bằng camera. Hệ thống sẽ tự động phát hiện sâu răng và hiển thị các bounding boxes kèm thông tin chi tiết.

## Tính Năng

- **Phát hiện sâu răng**: Sử dụng YOLOv8 để xác định vị trí sâu răng trên hình ảnh.
- **Giao diện thân thiện**: Tải lên hình ảnh hoặc chụp hình bằng camera trực tiếp.
- **Hiển thị kết quả**: Hiển thị hình ảnh đã được đánh dấu sâu răng và bảng kết quả chi tiết.

## Công Nghệ Sử Dụng

- **Python 3.11**
- **Streamlit**: Để xây dựng giao diện người dùng.
- **YOLOv8**: Để phát hiện đối tượng sâu răng.
- **Pillow**: Để xử lý hình ảnh.
- **Ultralytics**: Thư viện hỗ trợ YOLOv8.

## Cài Đặt

### Bước 1: Cài Đặt Conda và Tạo Môi Trường

Nếu bạn chưa cài đặt Anaconda hoặc Miniconda, hãy tải và cài đặt từ [trang chính thức của Conda](https://docs.conda.io/en/latest/miniconda.html).

Tạo môi trường ảo với Python 3.11:

```bash
conda create -n tooth_decay_env python=3.11
conda activate tooth_decay_env
```

### Bước 2: Cài Đặt Các Yêu Cầu

Chuyển đến thư mục dự án và cài đặt các thư viện cần thiết bằng `pip`:

```bash
pip install -r requirements.txt
```

**Lưu ý:** Đảm bảo bạn đã có file `best.pt` trong thư mục làm việc để mô hình YOLOv8 hoạt động đúng.

## Chạy Ứng Dụng

Khởi chạy ứng dụng Streamlit:

```bash
streamlit run main.py
```

## Cấu Trúc Thư Mục

```
tooth-decay-detection/
├── main.py
├── best.pt
├── requirements.txt
├── README.md
└── .gitignore
```

## Sử Dụng

1. **Chọn phương thức nhập hình ảnh**: Tải lên từ máy tính hoặc chụp hình bằng camera.
2. **Tải lên hoặc chụp hình**: Sử dụng tùy chọn đã chọn để cung cấp hình ảnh.
3. **Phát hiện sâu răng**: Hệ thống sẽ tự động phát hiện và đánh dấu sâu răng trên hình ảnh.
4. **Xem kết quả**: Xem hình ảnh đã được đánh dấu và bảng kết quả chi tiết.



## Liên Hệ

Nếu bạn có bất kỳ câu hỏi hoặc đề xuất nào, vui lòng liên hệ với tôi qua email: **ducdatit2002@gmail.com** hoặc [GitHub](https://github.com/ducdatit2002).

---
