import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import joblib
import os

print("📥 Đang tải dữ liệu MNIST (70.000 ảnh 28x28)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Giảm scale về [0,1]
X = X / 255.0

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Hàm trích xuất HOG ===
def extract_hog_features(data):
    print("🔧 Đang trích xuất HOG features...")
    features = []
    for i, img in enumerate(data):
        if i % 5000 == 0 and i > 0:
            print(f"  → Xử lý {i}/{len(data)} ảnh...")
        img_2d = img.reshape((28, 28))
        hog_feat = hog(
            img_2d,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True
        )
        features.append(hog_feat)
    return np.array(features)

# === Trích xuất HOG cho toàn bộ dataset ===
start_time = time.time()
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)
print(f"⏱️ Hoàn tất trích xuất HOG trong {time.time() - start_time:.2f} giây.\n")

# === Huấn luyện Linear SVM ===
print("⚙️ Đang huấn luyện mô hình Linear SVM...")
start_time = time.time()

clf = LinearSVC(C=1.0, max_iter=3000, verbose=0)
clf.fit(X_train_hog, y_train)

train_time = time.time() - start_time
print(f"✅ Hoàn tất huấn luyện trong {train_time:.2f} giây.\n")

# === Đánh giá mô hình ===
y_pred = clf.predict(X_test_hog)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy (Linear SVM + HOG - MNIST): {acc:.4f}\n")
print(classification_report(y_test, y_pred))

# === Hiển thị Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - Linear SVM + HOG (Acc: {acc:.4f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# === Lưu model ===
os.makedirs("models", exist_ok=True)
pack = {"clf": clf}
joblib.dump(pack, "models/svm_hog.pkl")
print("💾 Đã lưu model Linear SVM (HOG) vào 'models/svm_hog.pkl'")
