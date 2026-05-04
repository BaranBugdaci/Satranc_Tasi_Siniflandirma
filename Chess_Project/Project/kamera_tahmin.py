import cv2
import numpy as np
import tensorflow as tf

# ── Model ve sınıfları yükle ──
model   = tf.keras.models.load_model('best_chess_model.keras')
CLASSES = ['Rook', 'Queen', 'Pawn', 'Knight', 'King', 'Bishop']
SINIF_TR = {
    'Rook':   'Kale',
    'Queen':  'Vezir',
    'Pawn':   'Piyon',
    'Knight': 'At',
    'King':   'Şah',
    'Bishop': 'Fil'
}
RENKLER = {
    'Rook':   (0, 165, 255),
    'Queen':  (147, 20, 255),
    'Pawn':   (0, 200, 0),
    'Knight': (255, 127, 0),
    'King':   (0, 215, 255),
    'Bishop': (255, 50, 50)
}

# ── Kamerayı aç ──
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

print("Kamera açıldı.")
print("Satranç taşını ekranın ortasındaki kareye tut.")
print("Çıkmak için 'Q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── Ortada ROI karesi çiz ──
    roi_size = min(h, w) // 2
    x1 = (w - roi_size) // 2
    y1 = (h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # ROI bölgesini al ve modele gönder
    roi        = frame[y1:y2, x1:x2]
    roi_rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized   = cv2.resize(roi_rgb, (224, 224))
    roi_normalized = roi_resized / 255.0
    roi_input  = np.expand_dims(roi_normalized, axis=0)

    preds      = model.predict(roi_input, verbose=0)[0]
    pred_idx   = np.argmax(preds)
    pred_class = CLASSES[pred_idx]
    pred_tr    = SINIF_TR[pred_class]
    confidence = preds[pred_idx] * 100
    renk       = RENKLER[pred_class]

    # ── ROI çerçevesi ──
    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 3)

    # ── Ana tahmin yazısı ──
    etiket = f"{pred_tr} ({pred_class})  %{confidence:.1f}"
    cv2.rectangle(frame, (x1, y1 - 45), (x2, y1), renk, -1)
    cv2.putText(frame, etiket, (x1 + 8, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Sol üstte tüm sınıf olasılıkları ──
    panel_x = 10
    cv2.rectangle(frame, (panel_x, 10), (220, 30 + len(CLASSES) * 28), (30, 30, 30), -1)
    cv2.putText(frame, "Olasiliklar:", (panel_x + 5, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for i, (cls, prob) in enumerate(zip(CLASSES, preds)):
        y_pos  = 50 + i * 28
        bar_w  = int(prob * 180)
        r      = RENKLER[cls]
        # Bar
        cv2.rectangle(frame, (panel_x + 5, y_pos - 14),
                      (panel_x + 5 + bar_w, y_pos + 4), r, -1)
        # Yazı
        yazi = f"{SINIF_TR[cls]}: %{prob*100:.1f}"
        cv2.putText(frame, yazi, (panel_x + 8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255) if prob == max(preds) else (180, 180, 180), 1)

    # ── Alt köşe: kullanım talimatı ──
    cv2.putText(frame, "Q: Cikis", (w - 120, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow('Satranc Tasi Siniflandirici', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Kamera kapatildi.")