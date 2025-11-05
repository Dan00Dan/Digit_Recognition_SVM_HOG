import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFilter
from skimage.feature import hog
import numpy as np
import joblib, os, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "models/svm_hog.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Chưa thấy model. Hãy chạy train_svm_hog.py trước.")

pack = joblib.load(MODEL_PATH)
clf = pack["clf"]

CANVAS_SIZE = 280
BRUSH_SIZE = 14
BG_COLOR = "#f4f4f8"
PRIMARY = "#1565c0"
SUCCESS = "#2e7d32"


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Nhận dạng chữ số viết tay (SVM + HOG)")

        w, h = 520, 870
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws // 2) - (w // 2)
        y = (hs // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(520, 850)

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
        style.map("TButton", background=[("active", "#e3f2fd")])

        # ===== TIÊU ĐỀ =====
        tk.Label(root, text="Nhận dạng chữ số viết tay",
                 font=("Segoe UI", 22, "bold"), bg=BG_COLOR, fg=PRIMARY).pack(pady=(20, 0))
        tk.Label(root, text="Dựa trên HOG + SVM (RBF kernel)",
                 font=("Segoe UI", 11), bg=BG_COLOR, fg="#424242").pack(pady=(0, 15))

        # ===== CANVAS =====
        canvas_frame = tk.Frame(root, bg=BG_COLOR)
        canvas_frame.pack(pady=10)
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", highlightbackground="#b0bec5", highlightthickness=2)
        self.canvas.pack()
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.img)
        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # ===== PREVIEW =====
        tk.Label(root, text="Ảnh 28×28 sau xử lý:", font=("Segoe UI", 10, "italic"),
                 bg=BG_COLOR, fg="#616161").pack(pady=(15, 3))
        self.preview_box = tk.Label(root, bg="white", width=140, height=140, relief="ridge", bd=2)
        self.preview_box.pack()

        # ===== NÚT =====
        btn_frame = tk.Frame(root, bg=BG_COLOR)
        btn_frame.pack(pady=18)
        ttk.Button(btn_frame, text="🧮 Dự đoán", command=self.predict).grid(row=0, column=0, padx=10)
        ttk.Button(btn_frame, text="🧹 Xóa", command=self.clear).grid(row=0, column=1, padx=10)
        ttk.Button(btn_frame, text="❌ Thoát", command=root.destroy).grid(row=0, column=2, padx=10)

        # ===== KẾT QUẢ =====
        self.result_box = tk.Label(root, text="Kết quả: ?", font=("Consolas", 18, "bold"),
                                   bg=PRIMARY, fg="white", width=25, height=2, relief="flat")
        self.result_box.pack(pady=(20, 10))

        self.proba_box = tk.Label(root, text="Top-3: -", font=("Segoe UI", 11),
                                  bg=BG_COLOR, fg="#333")
        self.proba_box.pack(pady=(0, 15))

        # ===== FOOTER =====
        footer = tk.Label(root,
            text="Trần Thị Như Quỳnh – 2374802010428 | Khoa CNTT – ĐH Văn Lang © 2025",
            font=("Segoe UI", 9),
            bg=BG_COLOR, fg="#9e9e9e")
        footer.pack(side="bottom", fill="x", pady=10)

    # ===== VẼ =====
    def paint(self, e):
        x, y = e.x, e.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=BRUSH_SIZE, fill="black", capstyle=tk.ROUND)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=BRUSH_SIZE)
        self.last_x, self.last_y = x, y
        self.update_preview()

    def reset(self, e):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.img)
        self.result_box.config(text="Kết quả: ?", bg=PRIMARY, fg="white")
        self.proba_box.config(text="Top-3: -")
        self.preview_box.config(image="")

    # ===== PREVIEW =====
    def update_preview(self):
        img = self.img.copy()
        bbox = ImageOps.invert(img).getbbox()
        if bbox:
            img = img.crop(bbox)
        img = ImageOps.expand(img, border=20, fill=255)
        img = ImageOps.pad(img, (28, 28), color=255)
        small = img.resize((140, 140), Image.NEAREST)
        preview_img = ImageTk.PhotoImage(small)
        self.preview_box.config(image=preview_img)
        self.preview_box.image = preview_img

    # ===== TIỀN XỬ LÝ (HOG) =====
    def preprocess(self):
        img = self.img.copy()
        bbox = ImageOps.invert(img).getbbox()
        if bbox:
            img = img.crop(bbox)
        else:
            raise ValueError("Ảnh trống - chưa vẽ!")

        img = ImageOps.expand(img, border=20, fill=255)
        img = ImageOps.pad(img, (28, 28), color=255)
        img = ImageOps.autocontrast(img).filter(ImageFilter.GaussianBlur(0.6))
        img = ImageOps.invert(img)
        arr = np.array(img).astype("float32") / 255.0

        feat = hog(arr, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), block_norm="L2-Hys",
                   transform_sqrt=True)
        return feat.reshape(1, -1), arr

    # ===== DỰ ĐOÁN ===== 
    def predict(self):
        try:
            X, img28 = self.preprocess()
            y_pred = clf.predict(X)[0]
            scores = clf.decision_function(X)[0]
            top3 = np.argsort(scores)[::-1][:3]
            top3_str = ", ".join([f"{i}: {scores[i]:.2f}" for i in top3])

            self.result_box.config(text=f"Kết quả: {y_pred}", bg=SUCCESS, fg="white")
            self.proba_box.config(text=f"Top-3: {top3_str}")

            plt.imshow(img28, cmap='gray')
            plt.title(f"Ảnh model nhìn thấy (Dự đoán: {y_pred})")
            plt.axis("off")
            plt.show()

        except Exception as e:
            messagebox.showerror("Lỗi khi dự đoán", str(e))
            print("❌ Lỗi:", e)


# ===== CHẠY APP =====
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
