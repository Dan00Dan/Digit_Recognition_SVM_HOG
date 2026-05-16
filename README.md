# 🧠 Nhận dạng chữ số viết tay - SVM + HOG

Ứng dụng nhận dạng chữ số viết tay được xây dựng bằng Python, sử dụng **SVM (RBF kernel)** kết hợp với **HOG (Histogram of Oriented Gradients)** để trích xuất đặc trưng từ ảnh.

## 🚀 Tính năng
- ✍️ Vẽ chữ số trực tiếp trên canvas Tkinter  
- 🧮 Tiền xử lý ảnh và trích xuất đặc trưng HOG  
- 🤖 Dự đoán bằng mô hình SVM huấn luyện trên tập dữ liệu **MNIST**  
- 📊 Hiển thị kết quả Top-3 và ảnh 28×28 sau xử lý  
- 💾 Lưu và nạp model HOG (svm_hog.pkl) tự động  
- 🎨 Giao diện thân thiện, thuần **Tkinter + Pillow**

## 🧩 Cấu trúc thư mục
```
Digit_Recognition_SVM_HOG/
├── models/
│ └── svm_hog.pkl
├── app.py
├── train_svm_hog.py
├── requirements.txt
└── README.md
```

## 🧰 Cách chạy
```bash
pip install -r requirements.txt
python train_svm_hog.py    # Huấn luyện model
python app.py              # Chạy giao diện
📸 Kết quả minh họa
```
👩‍💻 Tác giả

Trần Thị Như Quỳnh – 2374802010428
Khoa Công Nghệ Thông Tin, Trường Đại học Văn Lang
