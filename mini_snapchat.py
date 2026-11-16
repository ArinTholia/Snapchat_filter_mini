import cv2
import os
import math
import numpy as np
import mediapipe as mp
import time


# Paths

base_path = os.path.dirname(os.path.abspath(__file__))

# Face filters
face_cascade_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
sunglasses_path = os.path.join(base_path, "filters", "sunglasses.png")
mustache_path = os.path.join(base_path, "filters", "mustache.png")
dog_ears_path = os.path.join(base_path, "filters", "dog_ears.png")
heart_path = os.path.join(base_path, "filters", "heart.png")

# Button icons
icon_sunglasses_path = os.path.join(base_path, "filters", "icons", "sunglasses_icon.png")
icon_mustache_path = os.path.join(base_path, "filters", "icons", "mustache_icon.png")
icon_dog_ears_path = os.path.join(base_path, "filters", "icons", "dog_ears_icon.png")
icon_heart_path = os.path.join(base_path, "filters", "icons", "heart_icon.png")


# Load Face Cascade

face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Face cascade not found!")
    exit()


# Load Filter Images

sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
mustache = cv2.imread(mustache_path, cv2.IMREAD_UNCHANGED)
dog_ears = cv2.imread(dog_ears_path, cv2.IMREAD_UNCHANGED)
heart_img = cv2.imread(heart_path, cv2.IMREAD_UNCHANGED)

# Load Button Icons
icon_sunglasses = cv2.imread(icon_sunglasses_path, cv2.IMREAD_UNCHANGED)
icon_mustache = cv2.imread(icon_mustache_path, cv2.IMREAD_UNCHANGED)
icon_dog_ears = cv2.imread(icon_dog_ears_path, cv2.IMREAD_UNCHANGED)
icon_heart = cv2.imread(icon_heart_path, cv2.IMREAD_UNCHANGED)

if any(x is None for x in [sunglasses, mustache, dog_ears, heart_img,
                           icon_sunglasses, icon_mustache, icon_dog_ears, icon_heart]):
    print("One or more images not found!")
    exit()

# Resize large images once
sunglasses = cv2.resize(sunglasses, (800, 800))
mustache = cv2.resize(mustache, (800, 800))
dog_ears = cv2.resize(dog_ears, (800, 800))

# Resize icons for buttons
icon_size = 30
icon_sunglasses = cv2.resize(icon_sunglasses, (icon_size, icon_size))
icon_mustache = cv2.resize(icon_mustache, (icon_size, icon_size))
icon_dog_ears = cv2.resize(icon_dog_ears, (icon_size, icon_size))
icon_heart = cv2.resize(icon_heart, (icon_size, icon_size))


# UI Colors & Theme
SIDEBAR_TOP    = (40, 20, 60)       # dark purple top
SIDEBAR_BOT    = (20, 10, 40)       # darker purple bottom
BTN_NORMAL     = (60, 40, 90)       # muted purple button
BTN_HOVER      = (200, 100, 20)     # orange glow on hover
BTN_ACTIVE     = (30, 160, 60)      # green for active filter
BTN_BORDER     = (180, 100, 255)    # purple border
TEXT_COLOR     = (255, 255, 255)
TITLE_COLOR    = (255, 200, 80)     # golden title
SIDEBAR_W      = 170
CORNER_R       = 10


# Overlay Function

def overlay_filter(frame, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return frame
    roi = frame[y1:y2, x1:x2]
    overlay_bgr = overlay_resized[:, :, :3]
    alpha = overlay_resized[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=-1)
    blended = (overlay_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    return frame


def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1, border_color=None):
    """Draw a filled rounded rectangle, with optional border."""
    # filled
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness)
    # border
    if border_color:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), border_color, 1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), border_color, 1)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, border_color, 1)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, border_color, 1)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, border_color, 1)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, border_color, 1)


def draw_gradient_sidebar(frame, width, height):
    """Draw a vertical gradient sidebar on the left."""
    for i in range(height):
        alpha = i / height
        b = int(SIDEBAR_TOP[0] * (1 - alpha) + SIDEBAR_BOT[0] * alpha)
        g = int(SIDEBAR_TOP[1] * (1 - alpha) + SIDEBAR_BOT[1] * alpha)
        r = int(SIDEBAR_TOP[2] * (1 - alpha) + SIDEBAR_BOT[2] * alpha)
        frame[i, :width] = (b, g, r)
    return frame


def draw_progress_bar(frame, x, y, w, h, progress, color):
    """Draw a thin progress bar under a hovered button."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), color, -1)


# MediaPipe HandLandmarker Setup

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode
from mediapipe.tasks.python.vision.core import image as mp_image

model_path = os.path.join(base_path, "hand_landmarker.task")

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
timestamp_ms = 0

# Buttons Setup

buttons = {
    "Glasses":  {"pos": (10, 70,  150, 50), "icon": icon_sunglasses, "filter": 1},
    "Mustache": {"pos": (10, 135, 150, 50), "icon": icon_mustache,   "filter": 2},
    "B&W":      {"pos": (10, 200, 150, 50), "icon": None,            "filter": 3},
    "Dog Ears": {"pos": (10, 265, 150, 50), "icon": icon_dog_ears,   "filter": 4},
    "Heart":    {"pos": (10, 330, 150, 50), "icon": icon_heart,      "filter": 5},
    "Normal":   {"pos": (10, 395, 150, 50), "icon": None,            "filter": 0},
}

hover_start    = {}
hover_duration = 0.5
filter_mode    = 0
filter_names   = {0: "Normal", 1: "Glasses", 2: "Mustache", 3: "B&W", 4: "Dog Ears", 5: "Heart"}


# Camera Setup

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Main Loop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ── Face Filters ──────────────────────────────────────────────
    for (x, y, w, h) in faces:
        if filter_mode == 1:
            frame = overlay_filter(frame, sunglasses, x, y + h//4, w, h//3)
        elif filter_mode == 2:
            frame = overlay_filter(frame, mustache, x, y + h//2, w, h//3)
        elif filter_mode == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif filter_mode == 4:
            overlay_resized = cv2.resize(dog_ears, (w, h))
            ears_only  = overlay_resized[0:int(h * 0.5), :, :]
            ear_width  = int(w * 1.3)
            ear_height = int(h * 0.5)
            ear_x = x - int(w * 0.15)
            ear_y = y - int(h * 0.35)
            frame = overlay_filter(frame, ears_only, ear_x, ear_y, ear_width, ear_height)
            nose_width  = int(w * 0.4)
            nose_height = int(h * 0.3)
            nose_x = x + w//2 - nose_width//2
            nose_y = y + h//2 - nose_height//2
            nose_only = overlay_resized[int(h*0.55):int(h*0.85), int(w*0.3):int(w*0.7)]
            frame = overlay_filter(frame, nose_only, nose_x, nose_y, nose_width, nose_height)

    # ── Hand Detection ────────────────────────────────────────────
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    hand_result = hand_landmarker.detect_for_video(mp_img, timestamp_ms)
    timestamp_ms += 1

    finger_x = finger_y = None
    if hand_result.hand_landmarks:
        hand      = hand_result.hand_landmarks[0]
        index_tip = hand[8]
        thumb_tip = hand[4]

        finger_x = int(index_tip.x * w_frame)
        finger_y = int(index_tip.y * h_frame)

        if filter_mode == 5:
            x1 = int(thumb_tip.x * w_frame)
            y1 = int(thumb_tip.y * h_frame)
            x2 = int(index_tip.x * w_frame)
            y2 = int(index_tip.y * h_frame)
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < 0.15 * w_frame:
                heart_size = 120
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)
                heart_x = mid_x - heart_size // 2
                heart_y = mid_y - heart_size - 40
                frame = overlay_filter(frame, heart_img, heart_x, heart_y, heart_size, heart_size)

    # ── Gradient Sidebar ──────────────────────────────────────────
    draw_gradient_sidebar(frame, SIDEBAR_W, h_frame)

    # Thin accent line separating sidebar from camera feed
    cv2.line(frame, (SIDEBAR_W, 0), (SIDEBAR_W, h_frame), BTN_BORDER, 2)

    # ── Title ─────────────────────────────────────────────────────
    cv2.putText(frame, "MiniSnap", (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, TITLE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, "by Arin & Rishav", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 170, 255), 1, cv2.LINE_AA)

    # Thin divider under title
    cv2.line(frame, (10, 58), (SIDEBAR_W - 10, 58), BTN_BORDER, 1)

    # ── Buttons ───────────────────────────────────────────────────
    current_time = time.time()
    for name, info in buttons.items():
        bx, by, bw, bh = info["pos"]
        icon           = info["icon"]
        is_active      = (filter_mode == info["filter"])

        btn_color = BTN_ACTIVE if is_active else BTN_NORMAL
        bar_color = BTN_HOVER

        hovered = (finger_x is not None and
                   bx <= finger_x <= bx + bw and
                   by <= finger_y <= by + bh)

        if hovered:
            btn_color = BTN_HOVER
            if name not in hover_start:
                hover_start[name] = current_time
            elif current_time - hover_start[name] >= hover_duration:
                filter_mode = info["filter"]
        else:
            hover_start.pop(name, None)

        # Draw rounded button
        draw_rounded_rect(frame, bx, by, bw, bh, CORNER_R,
                          btn_color, border_color=BTN_BORDER)

        # Active glow outline
        if is_active:
            draw_rounded_rect(frame, bx - 2, by - 2, bw + 4, bh + 4,
                              CORNER_R + 2, (0, 0, 0), thickness=1,
                              border_color=(50, 220, 80))

        # Hover progress bar at bottom of button
        if hovered and name in hover_start:
            progress = min((current_time - hover_start[name]) / hover_duration, 1.0)
            draw_progress_bar(frame, bx, by + bh - 4, bw, 4, progress, (255, 180, 0))

        # Label
        cv2.putText(frame, name, (bx + 10, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

        # Icon
        if icon is not None:
            icon_h, icon_w = icon.shape[:2]
            icon_x = bx + bw - icon_w - 8
            icon_y = by + (bh - icon_h) // 2
            frame = overlay_filter(frame, icon, icon_x, icon_y, icon_w, icon_h)

    # ── Finger cursor dot ─────────────────────────────────────────
    if finger_x is not None:
        cv2.circle(frame, (finger_x, finger_y), 10, (0, 220, 255), -1)
        cv2.circle(frame, (finger_x, finger_y), 12, (255, 255, 255), 1)

    # ── Active filter label (bottom-right HUD) ────────────────────
    label = f"Filter: {filter_names.get(filter_mode, 'Normal')}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    hud_x = w_frame - tw - 20
    hud_y = h_frame - 20
    cv2.rectangle(frame, (hud_x - 8, hud_y - th - 8),
                  (hud_x + tw + 8, hud_y + 8), (30, 20, 50), -1)
    cv2.rectangle(frame, (hud_x - 8, hud_y - th - 8),
                  (hud_x + tw + 8, hud_y + 8), BTN_BORDER, 1)
    cv2.putText(frame, label, (hud_x, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, TITLE_COLOR, 2, cv2.LINE_AA)

    # ── Hint text ─────────────────────────────────────────────────
    cv2.putText(frame, "Point finger to select filter | ESC to quit",
                (SIDEBAR_W + 10, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imshow("MiniSnapchat  |  Arin Tholia & Rishav Kumar Singh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
