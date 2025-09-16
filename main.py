import cv2
import torch

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import torchvision.transforms as transforms

# ===== Load Model =====

import torch.nn as nn
from torchvision import transforms, models


# --- Config ---
num_classes = 7
import os
save_path = os.path.join(os.path.dirname(__file__), "src", "models", "Final_ModelV2.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Model ---
model = models.resnet18(weights=None)  # no pretrained, matches your training
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)
model.eval()

# --- Emotion labels ---
emotions = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}

def predict_emotion(pil_img):
    input_img = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_img)
        _, predicted = torch.max(outputs, 1)
    return emotions[predicted.item()]


# ===== GUI App =====

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection - Modern GUI")
        self.root.geometry("500x550")
        self.root.configure(bg="#232946")

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 12, 'bold'), padding=8, background="#eebbc3", foreground="#232946")
        style.configure('TLabel', background="#232946", foreground="#eebbc3", font=('Segoe UI', 12))
        style.configure('Header.TLabel', font=('Segoe UI', 20, 'bold'), background="#232946", foreground="#eebbc3")
        style.map('TButton', background=[('active', '#b8c1ec')])

        # Header
        self.header = ttk.Label(self.root, text="Emotion Detection", style='Header.TLabel', anchor='center')
        self.header.pack(pady=(20, 10))

        # Video/Image display area (centered)
        self.display_size = 350
        self.img_frame = tk.Frame(self.root, width=self.display_size, height=self.display_size, bg="#121629", highlightbackground="#eebbc3", highlightthickness=2)
        self.img_frame.pack(pady=10)
        self.img_frame.pack_propagate(False)
        self.label_img = tk.Label(self.img_frame, bg="#121629")
        self.label_img.pack(expand=True)


        # Result text
        self.result_label = ttk.Label(self.root, text="Result: None", font=("Segoe UI", 16, "bold"), foreground="#eebbc3", background="#232946")
        self.result_label.pack(pady=(10, 20))

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#232946")
        btn_frame.pack(pady=10)

        self.btn_webcam = ttk.Button(btn_frame, text="Use Webcam", command=self.start_webcam)
        self.btn_webcam.grid(row=0, column=0, padx=10)

        self.btn_browse = ttk.Button(btn_frame, text="Browse Image", command=self.browse_image)
        self.btn_browse.grid(row=0, column=1, padx=10)

        self.btn_stop = ttk.Button(btn_frame, text="Stop Webcam", command=self.stop_webcam)
        self.btn_stop.grid(row=0, column=2, padx=10)

        # Webcam variables
        self.cap = None
        self.running = False

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.label_img.config(image="")
        self.result_label.config(text="Result: None")

    def update_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame for display
                frame_resized = cv2.resize(frame, (self.display_size, self.display_size))
                pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                emotion = predict_emotion(pil_img)

                cv2.putText(frame_resized, emotion, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=im_pil)

                self.label_img.imgtk = imgtk
                self.label_img.config(image=imgtk)

                self.result_label.config(text=f"Result: {emotion}")

            self.root.after(20, self.update_frame)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if file_path:
            img = cv2.imread(file_path)
            # Resize to display size for consistency
            img_resized = cv2.resize(img, (self.display_size, self.display_size))
            pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            emotion = predict_emotion(pil_img)

            # Overlay emotion label
            cv2.putText(img_resized, emotion, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=im_pil)

            # Center the image in the label (set fixed size)
            self.label_img.config(width=self.display_size, height=self.display_size, image=imgtk)
            self.label_img.imgtk = imgtk

            self.result_label.config(text=f"Result: {emotion}")


# ===== Run App =====
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()