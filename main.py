import cv2
from PIL import Image
import numpy as np

# Yüz tanıma modeli yükleniyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Resmi yükle
img = cv2.imread('erkekler.jpg')

# Gri tona dönüştür
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Her yüz için
for i, (x, y, w, h) in enumerate(faces):
    # Yüzü kırp
    cropped_img = img[y:y + h, x:x + w]

    # BGR'den RGB'ye dönüştür
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    # Kırpmış resmi Pillow kütüphanesiyle aç
    pil_image = Image.fromarray(cropped_img_rgb)

    # Her yüz için farklı bir dosya adı belirleyin
    dosya_adı = f"yuz_{i + 1}_portre.jpg"

    # Masaüstüne kaydet
    pil_image.save(dosya_adı)

print(f"{len(faces)} yüz portre başarıyla kaydedildi.")
