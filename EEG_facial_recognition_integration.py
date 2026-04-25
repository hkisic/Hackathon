"""
Emotion Classification System
Combines EEG (Unicorn Hybrid Black + gpype) and facial recognition (hsemotion)
to classify 4 emotional states: Happy, Sad, Angry, Contempt

Requirements:
    pip install hsemotion-onnx opencv-python matplotlib numpy gpype
"""

import threading
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import gpype as gp

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
EEG_SAMPLE_RATE = 250       # Hz (Unicorn default)
WINDOW_SECONDS  = 2         # seconds of EEG to average over
FUSION_EEG_W    = 0.5       # weight for EEG in final fusion (0–1)
FUSION_FACE_W   = 0.5       # weight for face in final fusion (0–1)


# ─────────────────────────────────────────────
#  4-EMOTION VALENCE / AROUSAL MAP
# ─────────────────────────────────────────────
#  Based on the circumplex model of affect:
#    Happy   → positive valence, moderate-high arousal
#    Sad     → negative valence, low arousal
#    Angry   → negative valence, high arousal
#    Contempt→ negative valence, low-moderate arousal

EMOTION_TO_VA = {
    "happy":    ( 0.80,  0.40),
    "sad":      (-0.60, -0.60),
    "angry":    (-0.80,  0.70),
    "contempt": (-0.50, -0.20),
}

# hsemotion may return slightly different label names
LABEL_MAP = {
    "happiness": "happy",
    "happy":     "happy",
    "sadness":   "sad",
    "sad":       "sad",
    "anger":     "angry",
    "angry":     "angry",
    "contempt":  "contempt",
}

def closest_emotion(v, a):
    """Return whichever of the 4 emotions is closest to (v, a)."""
    best, best_dist = "happy", float("inf")
    for name, (ev, ea) in EMOTION_TO_VA.items():
        d = (v - ev) ** 2 + (a - ea) ** 2
        if d < best_dist:
            best_dist = d
            best = name
    return best


# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
lock  = threading.Lock()
state = {
    "eeg_valence":   0.0,
    "eeg_arousal":   0.0,
    "alpha_power":   0.0,
    "beta_power":    0.0,
    "alpha_asym":    0.0,
    "face_valence":  0.0,
    "face_arousal":  0.0,
    "face_emotion":  "none",
    "face_detected": False,
    "final_valence": 0.0,
    "final_arousal": 0.0,
    "final_emotion": "happy",
    # Calibration ranges (overwritten during calibration)
    "cal_alpha_min":  0.1,
    "cal_alpha_max":  1.0,
    "cal_beta_min":   0.1,
    "cal_beta_max":   1.0,
    "cal_asym_min":  -0.5,
    "cal_asym_max":   0.5,
}

valence_history = deque(maxlen=10)
arousal_history = deque(maxlen=10)


# ─────────────────────────────────────────────
#  EEG UTILITIES
# ─────────────────────────────────────────────
def bandpower(signal, fs, low, high):
    """Estimate power in a frequency band using FFT."""
    n = len(signal)
    if n == 0:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    freqs    = np.fft.rfftfreq(n, 1.0 / fs)
    mask     = (freqs >= low) & (freqs <= high)
    return float(np.mean(fft_vals[mask])) if mask.any() else 0.0

def normalize(value, min_val, max_val):
    """Normalize a value to the range [-1, 1]."""
    if max_val == min_val:
        return 0.0
    norm = (value - min_val) / (max_val - min_val)
    return float(np.clip(norm * 2 - 1, -1.0, 1.0))


# ─────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────
def run_calibration():
    """
    Two-phase calibration:
      Phase 1 — Eyes closed, relaxed  → max alpha, min beta
      Phase 2 — Eyes open, focused    → min alpha, max beta
    """
    print("\n" + "=" * 50)
    print("  CALIBRATION")
    print("=" * 50)

    def collect_samples(duration=10):
        alphas, betas, asyms = [], [], []
        for _ in range(duration * 10):
            with lock:
                alphas.append(state["alpha_power"])
                betas.append(state["beta_power"])
                asyms.append(state["alpha_asym"])
            time.sleep(0.1)
        return np.mean(alphas), np.mean(betas), np.mean(asyms)

    # Phase 1: Relaxed
    input("\nPhase 1: RELAXED\n"
          "Close your eyes, breathe slowly, and relax.\n"
          "Press Enter when ready...")
    print("Recording for 10 seconds...")
    for i in range(10, 0, -1):
        print(f"  {i}s remaining...", end="\r")
        time.sleep(1)
    alpha_relax, beta_relax, asym_relax = collect_samples()
    print("\n✓ Relaxed state recorded")

    # Phase 2: Focused
    input("\nPhase 2: FOCUSED\n"
          "Open your eyes, stare at a fixed point, and concentrate hard.\n"
          "Press Enter when ready...")
    print("Recording for 10 seconds...")
    for i in range(10, 0, -1):
        print(f"  {i}s remaining...", end="\r")
        time.sleep(1)
    alpha_focus, beta_focus, asym_focus = collect_samples()
    print("\n✓ Focused state recorded")

    with lock:
        state["cal_alpha_min"] = min(alpha_relax, alpha_focus)
        state["cal_alpha_max"] = max(alpha_relax, alpha_focus)
        state["cal_beta_min"]  = min(beta_relax,  beta_focus)
        state["cal_beta_max"]  = max(beta_relax,  beta_focus)
        state["cal_asym_min"]  = min(asym_relax,  asym_focus)
        state["cal_asym_max"]  = max(asym_relax,  asym_focus)

    print("\n✓ Calibration complete!")
    print(f"  Alpha : {state['cal_alpha_min']:.3f} → {state['cal_alpha_max']:.3f}")
    print(f"  Beta  : {state['cal_beta_min']:.3f}  → {state['cal_beta_max']:.3f}")
    print(f"  Asym  : {state['cal_asym_min']:.3f}  → {state['cal_asym_max']:.3f}\n")


# ─────────────────────────────────────────────
#  EEG THREAD  (gpype + Unicorn Hybrid Black)
# ─────────────────────────────────────────────
def eeg_thread():
    print("[EEG] Connecting to Unicorn Hybrid Black...")

    buffer_left  = deque(maxlen=WINDOW_SECONDS * EEG_SAMPLE_RATE)
    buffer_right = deque(maxlen=WINDOW_SECONDS * EEG_SAMPLE_RATE)

    # Unicorn channel layout: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
    # C3 (index 1) = left frontal, C4 (index 3) = right frontal
    source   = gp.UnicornHybridBlack(device_serial=None)  # auto-detect
    splitter = gp.Splitter(source)
    left_ch  = gp.ChannelSelector(splitter, channels=[1])  # C3
    right_ch = gp.ChannelSelector(splitter, channels=[3])  # C4

    def on_data_left(data):
        if data is not None and len(data) > 0:
            buffer_left.extend(data[:, 0].tolist())

    def on_data_right(data):
        if data is not None and len(data) > 0:
            buffer_right.extend(data[:, 0].tolist())
            _update_eeg_state()

    def _update_eeg_state():
        if len(buffer_left) < EEG_SAMPLE_RATE:
            return

        sig_l = np.array(buffer_left)
        sig_r = np.array(buffer_right)

        alpha_l = bandpower(sig_l, EEG_SAMPLE_RATE, 8,  13)
        alpha_r = bandpower(sig_r, EEG_SAMPLE_RATE, 8,  13)
        beta_l  = bandpower(sig_l, EEG_SAMPLE_RATE, 13, 30)
        beta_r  = bandpower(sig_r, EEG_SAMPLE_RATE, 13, 30)

        alpha = (alpha_l + alpha_r) / 2
        beta  = (beta_l  + beta_r)  / 2
        asym  = alpha_r - alpha_l   # frontal alpha asymmetry → valence

        # Arousal = beta/alpha ratio, normalized against calibration
        ratio     = beta / (alpha + 1e-6)
        cal_r_min = state["cal_beta_min"]  / (state["cal_alpha_max"] + 1e-6)
        cal_r_max = state["cal_beta_max"]  / (state["cal_alpha_min"] + 1e-6)
        arousal   = normalize(ratio, cal_r_min, cal_r_max)

        # Valence = frontal alpha asymmetry, normalized against calibration
        valence = normalize(asym, state["cal_asym_min"], state["cal_asym_max"])

        with lock:
            state["alpha_power"] = alpha
            state["beta_power"]  = beta
            state["alpha_asym"]  = asym
            state["eeg_arousal"] = float(np.clip(arousal, -1, 1))
            state["eeg_valence"] = float(np.clip(valence, -1, 1))

    left_ch.on_data  = on_data_left
    right_ch.on_data = on_data_right
    source.start()
    print("[EEG] Unicorn connected and streaming")


# ─────────────────────────────────────────────
#  FACE THREAD  (hsemotion + webcam)
# ─────────────────────────────────────────────
def face_thread():
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    print("[FACE] Loading hsemotion model...")
    recognizer   = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[FACE] Could not open webcam — face modality disabled")
        return

    print("[FACE] Webcam active")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img   = frame[y:y+h, x:x+w]
            face_rgb   = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                raw_emotion, _ = recognizer.predict_emotions(face_rgb, logits=False)
                emotion = LABEL_MAP.get(raw_emotion.lower(), None)

                # Only update state if it's one of our 4 target emotions
                if emotion in EMOTION_TO_VA:
                    valence, arousal = EMOTION_TO_VA[emotion]
                    with lock:
                        state["face_emotion"]  = emotion
                        state["face_valence"]  = valence
                        state["face_arousal"]  = arousal
                        state["face_detected"] = True
                else:
                    with lock:
                        state["face_detected"] = False

            except Exception as e:
                print(f"[FACE] Error: {e}")
                with lock:
                    state["face_detected"] = False
        else:
            with lock:
                state["face_detected"] = False

        time.sleep(0.1)

    cap.release()


# ─────────────────────────────────────────────
#  FUSION
# ─────────────────────────────────────────────
def fuse():
    """Combine EEG and face signals into a single valence/arousal score."""
    with lock:
        eeg_v         = state["eeg_valence"]
        eeg_a         = state["eeg_arousal"]
        face_v        = state["face_valence"]
        face_a        = state["face_arousal"]
        face_detected = state["face_detected"]

    # If face not detected, rely fully on EEG
    ew = 1.0 if not face_detected else FUSION_EEG_W
    fw = 0.0 if not face_detected else FUSION_FACE_W

    final_v = ew * eeg_v + fw * face_v
    final_a = ew * eeg_a + fw * face_a

    # Smooth over recent history to reduce jitter
    valence_history.append(final_v)
    arousal_history.append(final_a)
    smooth_v = float(np.mean(valence_history))
    smooth_a = float(np.mean(arousal_history))

    emotion = closest_emotion(smooth_v, smooth_a)

    with lock:
        state["final_valence"] = smooth_v
        state["final_arousal"] = smooth_a
        state["final_emotion"] = emotion


# ─────────────────────────────────────────────
#  REAL-TIME PLOT
# ─────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":    "#FFD700",
    "sad":      "#4fc3f7",
    "angry":    "#ef5350",
    "contempt": "#ab47bc",
}

def launch_plot():
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Draw the 4 reference emotion markers
    for name, (ev, ea) in EMOTION_TO_VA.items():
        color = EMOTION_COLORS[name]
        ax.plot(ev, ea, 's', color=color, markersize=14, alpha=0.5, zorder=2)
        ax.text(ev + 0.06, ea + 0.08, name.upper(), fontsize=10,
                color=color, fontfamily="monospace", fontweight="bold", alpha=0.8)

    # Moving dot + trail
    dot,        = ax.plot([], [], 'o', markersize=20, zorder=5,
                          markeredgecolor="white", markeredgewidth=2)
    trail_x, trail_y = deque(maxlen=40), deque(maxlen=40)
    trail_line, = ax.plot([], [], '-', alpha=0.25, linewidth=2, color="white")

    emotion_text = ax.text(0, 1.3, "", ha="center", fontsize=18,
                           color="white", fontfamily="monospace", fontweight="bold")

    info_text = ax.text(-1.45, -1.42, "", fontsize=8.5,
                        color="#888888", fontfamily="monospace")

    # Axes styling
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Valence  ←  Negative  |  Positive  →",
                  color="#666666", fontsize=10)
    ax.set_ylabel("Arousal  ↕", color="#666666", fontsize=10)
    ax.tick_params(colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
    ax.set_title("Real-Time Emotion Classification",
                 color="white", fontsize=13, pad=15, fontfamily="monospace")

    def update(frame):
        fuse()

        with lock:
            v       = state["final_valence"]
            a       = state["final_arousal"]
            emotion = state["final_emotion"]
            alpha_p = state["alpha_power"]
            beta_p  = state["beta_power"]
            face_e  = state["face_emotion"]
            face_ok = state["face_detected"]
            eeg_v   = state["eeg_valence"]
            eeg_a   = state["eeg_arousal"]

        color = EMOTION_COLORS.get(emotion, "white")
        trail_x.append(v)
        trail_y.append(a)

        dot.set_data([v], [a])
        dot.set_color(color)
        trail_line.set_data(list(trail_x), list(trail_y))
        emotion_text.set_text(f"● {emotion.upper()}")
        emotion_text.set_color(color)

        face_str = face_e if face_ok else "no face detected"
        info_text.set_text(
            f"EEG   α={alpha_p:.3f}  β={beta_p:.3f}  "
            f"V={eeg_v:+.2f}  A={eeg_a:+.2f}\n"
            f"FACE  {face_str}  "
            f"V={state['face_valence']:+.2f}  A={state['face_arousal']:+.2f}"
        )

        return dot, trail_line, emotion_text, info_text

    ani = animation.FuncAnimation(fig, update, interval=200, blit=True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  EMOTION CLASSIFICATION SYSTEM")
    print("  Happy | Sad | Angry | Contempt")
    print("=" * 50)

    # Start EEG thread
    t_eeg = threading.Thread(target=eeg_thread, daemon=True)
    t_eeg.start()
    time.sleep(2)  # give EEG time to start streaming

    # Calibration
    run_calibration()

    # Start face thread
    t_face = threading.Thread(target=face_thread, daemon=True)
    t_face.start()
    time.sleep(1)

    print("\n[INFO] Launching real-time plot — close window to exit\n")
    launch_plot()