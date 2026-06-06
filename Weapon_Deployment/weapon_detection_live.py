import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = load_model("final_weapon_detection_model.keras")

# ============================================================
# CLASS LABELS
# ============================================================

class_names = [
    "knife",
    "no_weapon",
    "pistol"
]

# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Webcam started...")
print("Press Q to quit")

# ============================================================
# REAL-TIME DETECTION LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip webcam for natural view
    frame = cv2.flip(frame, 1)

    # Resize image for model
    img = cv2.resize(frame, (224, 224))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    label = class_names[predicted_class]

    # ========================================================
    # BOX COLOR
    # ========================================================

    if label == "no_weapon":
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red

    # Draw rectangle
    h, w, _ = frame.shape

    cv2.rectangle(
        frame,
        (50, 50),
        (w - 50, h - 50),
        color,
        3
    )

    # Display label
    text = f"{label} ({confidence:.2f})"

    cv2.putText(
        frame,
        text,
        (50, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Window
    cv2.imshow(
        "Weapon Detection - Live",
        frame
    )

    # Quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()