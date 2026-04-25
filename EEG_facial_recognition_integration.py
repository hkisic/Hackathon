"""
Emotion Classification System
Combines EEG (Unicorn + gpype) and facial recognition (hsemotion) 
to classify emotions on the valence-arousal plane in real time.

MODES:
  - Set MOCK_EEG = True  to simulate EEG data (for Mac / no Unicorn)
  - Set MOCK_EEG = False to use real gpype EEG pipeline

Requirements:
  pip install hsemotion-onnx opencv-python matplotlib numpy
  (pip install gpype  — only needed when MOCK_EEG = False)
"""

import threading
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from collections import deque

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MOCK_EEG        = True   # ← set False when running on Windows with gpype
CALIBRATE       = True   # ← set False to skip calibration and use defaults
WINDOW_SECONDS  = 2      # seconds of EEG to average over
EEG_SAMPLE_RATE = 250    # Hz (Unicorn default)
FUSION_EEG_W    = 0.5    # weight for EEG in final fusion (0–1)
FUSION_FACE_W   = 0.5    # weight for face in final fusion (0–1)


# ─────────────────────────────────────────────
#  EMOTION → VALENCE / AROUSAL MAP
#  Based on the circumplex model of affect
# ─────────────────────────────────────────────
EMOTION_TO_VA = {
    "happy":     ( 0.80,  0.30),
    "surprise":  ( 0.20,  0.80),
    "fear":      (-0.80,  0.70),
    "anger":     (-0.70,  0.60),
    "disgust":   (-0.60, -0.30),
    "sad":       (-0.40, -0.50),
    "sadness":   (-0.40, -0.50),
    "neutral":   ( 0.00, -0.60),
    "contempt":  (-0.30, -0.20),
    "relaxed":   ( 0.40, -0.40),
}

def emotion_label_to_va(label: str):
    label = label.lower().strip()
    return EMOTION_TO_VA.get(label, (0.0, 0.0))


# ─────────────────────────────────────────────
#  SHARED STATE  (thread-safe via lock)
# ─────────────────────────────────────────────
lock = threading.Lock()
state = {
    # EEG outputs
    "eeg_valence":   0.0,
    "eeg_arousal":   0.0,
    "alpha_power":   0.0,
    "beta_power":    0.0,
    "alpha_asym":    0.0,   # right - left alpha (frontal)

    # Face outputs
    "face_valence":  0.0,
    "face_arousal":  0.0,
    "face_emotion":  "neutral",
    "face_detected": False,

    # Fused outputs
    "final_valence": 0.0,
    "final_arousal": 0.0,
    "final_emotion": "neutral",

    # Calibration
    "cal_alpha_min":  0.0,
    "cal_alpha_max":  1.0,
    "cal_beta_min":   0.0,
    "cal_beta_max":   1.0,
    "cal_asym_min":  -1.0,
    "cal_asym_max":   1.0,
    "calibrated":     False,
}

# History for smoothing (last N values)
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
    """Normalize value to [-1, 1] range."""
    if max_val == min_val:
        return 0.0
    norm = (value - min_val) / (max_val - min_val)   # 0 to 1
    return float(np.clip(norm * 2 - 1, -1.0, 1.0))   # -1 to 1


# ─────────────────────────────────────────────
#  MOCK EEG THREAD  (simulates gpype output)
# ─────────────────────────────────────────────
def mock_eeg_thread():
    """Generates fake EEG-like data to test the pipeline on Mac."""
    print("[EEG] Mock mode active — simulating EEG signals")
    t = 0
    while True:
        t += 0.1
        # Simulate slowly drifting alpha/beta power + asymmetry
        alpha = 0.5 + 0.4 * np.sin(t * 0.3) + random.gauss(0, 0.05)
        beta  = 0.5 + 0.4 * np.cos(t * 0.2) + random.gauss(0, 0.05)
        asym  = 0.3 * np.sin(t * 0.15)       + random.gauss(0, 0.03)

        # Arousal = beta/alpha ratio, normalized
        ratio = beta / (alpha + 1e-6)
        arousal = normalize(ratio,
                            state["cal_beta_min"] / (state["cal_alpha_max"] + 1e-6),
                            state["cal_beta_max"] / (state["cal_alpha_min"] + 1e-6))

        # Valence = frontal alpha asymmetry, normalized
        valence = normalize(asym, state["cal_asym_min"], state["cal_asym_max"])

        with lock:
            state["alpha_power"] = alpha
            state["beta_power"]  = beta
            state["alpha_asym"]  = asym
            state["eeg_arousal"] = float(np.clip(arousal, -1, 1))
            state["eeg_valence"] = float(np.clip(valence, -1, 1))

        time.sleep(0.1)


# ─────────────────────────────────────────────
#  REAL EEG THREAD  (uses gpype — Windows only)
# ─────────────────────────────────────────────
def real_eeg_thread():
    """Real gpype pipeline. Only runs when MOCK_EEG = False."""
    try:
        import gpype as gp
    except ImportError:
        print("[EEG] gpype not found — switching to mock mode")
        mock_eeg_thread()
        return

    print("[EEG] Starting real gpype pipeline...")
    buffer_left  = deque(maxlen=WINDOW_SECONDS * EEG_SAMPLE_RATE)
    buffer_right = deque(maxlen=WINDOW_SECONDS * EEG_SAMPLE_RATE)

    # ── Build gpype pipeline ──
    # Unicorn channels: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
    # We use C3 (idx=1) as left frontal, C4 (idx=3) as right frontal
    source   = gp.UnicornHybridBlack(device_serial=None)   # auto-detect
    splitter = gp.Splitter(source)

    left_ch  = gp.ChannelSelector(splitter, channels=[1])   # C3
    right_ch = gp.ChannelSelector(splitter, channels=[3])   # C4

    def on_data(data):
        if data is not None and len(data) > 0:
            buffer_left.extend(data[:, 0].tolist())
            buffer_right.extend(data[:, 0].tolist())

            if len(buffer_left) >= EEG_SAMPLE_RATE:
                sig_l = np.array(buffer_left)
                sig_r = np.array(buffer_right)

                alpha_l = bandpower(sig_l, EEG_SAMPLE_RATE, 8,  13)
                alpha_r = bandpower(sig_r, EEG_SAMPLE_RATE, 8,  13)
                beta_l  = bandpower(sig_l, EEG_SAMPLE_RATE, 13, 30)
                beta_r  = bandpower(sig_r, EEG_SAMPLE_RATE, 13, 30)

                alpha   = (alpha_l + alpha_r) / 2
                beta    = (beta_l  + beta_r)  / 2
                asym    = alpha_r - alpha_l

                ratio   = beta / (alpha + 1e-6)
                arousal = normalize(ratio,
                                    state["cal_beta_min"] / (state["cal_alpha_max"] + 1e-6),
                                    state["cal_beta_max"] / (state["cal_alpha_min"] + 1e-6))
                valence = normalize(asym, state["cal_asym_min"], state["cal_asym_max"])

                with lock:
                    state["alpha_power"] = alpha
                    state["beta_power"]  = beta
                    state["alpha_asym"]  = asym
                    state["eeg_arousal"] = float(np.clip(arousal, -1, 1))
                    state["eeg_valence"] = float(np.clip(valence, -1, 1))

    left_ch.on_data  = on_data
    right_ch.on_data = on_data
    source.start()


# ─────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────
def run_calibration():
    """
    Two-phase calibration:
      1. Eyes closed (relaxed) → max alpha, min beta
      2. Focus task            → min alpha, max beta
    """
    print("\n" + "="*50)
    print("  CALIBRATION")
    print("="*50)
    samples = {"alpha": [], "beta": [], "asym": []}

    def collect(duration, label):
        print(f"\n→ {label}")
        print(f"  Hold for {duration} seconds...")
        for i in range(duration, 0, -1):
            print(f"  {i}...", end="\r")
            time.sleep(1)
        print("  Collecting data...   ")
        for _ in range(20):
            with lock:
                samples["alpha"].append(state["alpha_power"])
                samples["beta"].append(state["beta_power"])
                samples["asym"].append(state["alpha_asym"])
            time.sleep(0.1)

    # Phase 1: Relaxed (eyes closed)
    input("\nPhase 1: RELAXED STATE\nClose your eyes and relax.\nPress Enter when ready...")
    collect(10, "Eyes closed — relax completely")
    alpha_relax = np.mean(samples["alpha"][-20:])
    beta_relax  = np.mean(samples["beta"][-20:])
    asym_relax  = np.mean(samples["asym"][-20:])

    # Phase 2: Focus
    input("\nPhase 2: FOCUS STATE\nStare at a fixed point and concentrate hard.\nPress Enter when ready...")
    collect(10, "Eyes open — focus and concentrate")
    alpha_focus = np.mean(samples["alpha"][-20:])
    beta_focus  = np.mean(samples["beta"][-20:])
    asym_focus  = np.mean(samples["asym"][-20:])

    with lock:
        state["cal_alpha_min"] = min(alpha_relax, alpha_focus)
        state["cal_alpha_max"] = max(alpha_relax, alpha_focus)
        state["cal_beta_min"]  = min(beta_relax,  beta_focus)
        state["cal_beta_max"]  = max(beta_relax,  beta_focus)
        state["cal_asym_min"]  = min(asym_relax,  asym_focus)
        state["cal_asym_max"]  = max(asym_relax,  asym_focus)
        state["calibrated"]    = True

    print("\n✓ Calibration complete!")
    print(f"  Alpha range: {state['cal_alpha_min']:.3f} → {state['cal_alpha_max']:.3f}")
    print(f"  Beta  range: {state['cal_beta_min']:.3f}  → {state['cal_beta_max']:.3f}")
    print(f"  Asym  range: {state['cal_asym_min']:.3f}  → {state['cal_asym_max']:.3f}\n")


# ─────────────────────────────────────────────
#  FACE RECOGNITION THREAD
# ─────────────────────────────────────────────
def face_thread():
    """Runs hsemotion on webcam frames continuously."""
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    except ImportError:
        print("[FACE] hsemotion-onnx not found — face modality disabled")
        return

    print("[FACE] Loading hsemotion model...")
    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[FACE] Could not open webcam")
        return

    print("[FACE] Webcam active")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                emotion, scores = recognizer.predict_emotions(face_rgb, logits=False)
                valence, arousal = emotion_label_to_va(emotion)

                with lock:
                    state["face_emotion"]  = emotion
                    state["face_valence"]  = valence
                    state["face_arousal"]  = arousal
                    state["face_detected"] = True
            except Exception as e:
                print(f"[FACE] Prediction error: {e}")
        else:
            with lock:
                state["face_detected"] = False

        time.sleep(0.1)

    cap.release()


# ─────────────────────────────────────────────
#  FUSION
# ─────────────────────────────────────────────
def fuse():
    """Combines EEG and face signals into a final valence/arousal score."""
    with lock:
        eeg_v = state["eeg_valence"]
        eeg_a = state["eeg_arousal"]
        face_v = state["face_valence"]
        face_a = state["face_arousal"]
        face_detected = state["face_detected"]

    # If face not detected, rely fully on EEG
    if not face_detected:
        ew, fw = 1.0, 0.0
    else:
        ew, fw = FUSION_EEG_W, FUSION_FACE_W

    final_v = ew * eeg_v + fw * face_v
    final_a = ew * eeg_a + fw * face_a

    # Smooth over history
    valence_history.append(final_v)
    arousal_history.append(final_a)
    smooth_v = float(np.mean(valence_history))
    smooth_a = float(np.mean(arousal_history))

    # Map fused coordinates back to nearest emotion label
    label = closest_emotion(smooth_v, smooth_a)

    with lock:
        state["final_valence"] = smooth_v
        state["final_arousal"] = smooth_a
        state["final_emotion"] = label

def closest_emotion(v, a):
    """Returns the emotion label closest to (v, a) in the circumplex."""
    best, best_dist = "neutral", float("inf")
    for name, (ev, ea) in EMOTION_TO_VA.items():
        d = (v - ev)**2 + (a - ea)**2
        if d < best_dist:
            best_dist = d
            best = name
    return best


# ─────────────────────────────────────────────
#  REAL-TIME PLOT
# ─────────────────────────────────────────────
def launch_plot():
    """Animated valence-arousal plot updated in real time."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Reference emotion markers
    ref_emotions = {
        "Fear":      (-0.80,  0.70),
        "Anger":     (-0.70,  0.60),
        "Surprise":  ( 0.20,  0.80),
        "Happiness": ( 0.80,  0.30),
        "Disgust":   (-0.60, -0.30),
        "Sadness":   (-0.40, -0.50),
        "Neutral":   ( 0.00, -0.60),
        "Relaxed":   ( 0.40, -0.40),
        "Satisfied": ( 0.60, -0.50),
    }

    # Dot that moves
    dot, = ax.plot([], [], 'o', color="#00ffcc", markersize=18, zorder=5,
                   markeredgecolor="white", markeredgewidth=2)
    # Trail
    trail_x, trail_y = deque(maxlen=30), deque(maxlen=30)
    trail_line, = ax.plot([], [], '-', color="#00ffcc", alpha=0.3, linewidth=2)

    emotion_text = ax.text(0, 1.15, "", ha="center", va="center",
                           fontsize=16, color="white",
                           fontfamily="monospace", fontweight="bold")

    eeg_text = ax.text(-1.45, -1.35, "", fontsize=9, color="#aaaaaa",
                       fontfamily="monospace")

    face_text = ax.text(0.3, -1.35, "", fontsize=9, color="#aaaaaa",
                        fontfamily="monospace")

    def setup_axes():
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_xlabel("Valence  (Negative ← → Positive)",
                      color="#888888", fontsize=11)
        ax.set_ylabel("Arousal  (Low ← → High)",
                      color="#888888", fontsize=11)
        ax.tick_params(colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        # Quadrant labels
        for txt, x, y in [("HIGH AROUSAL", 0, 1.35),
                           ("LOW AROUSAL",  0, -1.35),
                           ("NEGATIVE", -1.35, 0),
                           ("POSITIVE",  1.35, 0)]:
            ax.text(x, y, txt, ha="center", va="center",
                    fontsize=8, color="#555555", fontfamily="monospace")

        # Reference emotion dots
        for name, (ev, ea) in ref_emotions.items():
            ax.plot(ev, ea, 's', color="#ff6b6b", markersize=7, alpha=0.6)
            ax.text(ev + 0.05, ea + 0.07, name, fontsize=8,
                    color="#ff6b6b", alpha=0.8)

        ax.set_title("Real-Time Emotion Classification",
                     color="white", fontsize=14, pad=20, fontfamily="monospace")

    setup_axes()

    def update(frame):
        fuse()

        with lock:
            v = state["final_valence"]
            a = state["final_arousal"]
            emotion = state["final_emotion"]
            alpha_p = state["alpha_power"]
            beta_p  = state["beta_power"]
            face_e  = state["face_emotion"]
            face_ok = state["face_detected"]
            eeg_v   = state["eeg_valence"]
            eeg_a   = state["eeg_arousal"]

        trail_x.append(v)
        trail_y.append(a)
        dot.set_data([v], [a])
        trail_line.set_data(list(trail_x), list(trail_y))
        emotion_text.set_text(f"● {emotion.upper()}")

        eeg_text.set_text(
            f"EEG  α={alpha_p:.2f}  β={beta_p:.2f}\n"
            f"     V={eeg_v:+.2f}  A={eeg_a:+.2f}"
        )
        face_label = face_e if face_ok else "no face"
        face_text.set_text(f"FACE  {face_label}\n"
                           f"      V={state['face_valence']:+.2f}  "
                           f"A={state['face_arousal']:+.2f}")

        return dot, trail_line, emotion_text, eeg_text, face_text

    ani = FuncAnimation(fig, update, interval=200, blit=True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  EMOTION CLASSIFICATION SYSTEM")
    print("  EEG + Facial Expression Fusion")
    print("="*50)
    print(f"  EEG mode : {'MOCK (simulated)' if MOCK_EEG else 'REAL (gpype)'}")
    print(f"  Fusion   : EEG {FUSION_EEG_W*100:.0f}% / Face {FUSION_FACE_W*100:.0f}%")
    print("="*50 + "\n")

    # Start EEG thread
    eeg_fn = mock_eeg_thread if MOCK_EEG else real_eeg_thread
    t_eeg = threading.Thread(target=eeg_fn, daemon=True)
    t_eeg.start()

    # Wait briefly for EEG to produce initial values
    time.sleep(1.5)

    # Calibration (optional)
    if CALIBRATE:
        run_calibration()
    else:
        print("[INFO] Skipping calibration — using default ranges")
        with lock:
            state["cal_alpha_min"] = 0.1
            state["cal_alpha_max"] = 1.0
            state["cal_beta_min"]  = 0.1
            state["cal_beta_max"]  = 1.0
            state["cal_asym_min"]  = -0.5
            state["cal_asym_max"]  =  0.5
            state["calibrated"]    = True

    # Start face thread
    t_face = threading.Thread(target=face_thread, daemon=True)
    t_face.start()

    print("[INFO] Launching real-time plot...")
    print("[INFO] Close the plot window to exit.\n")

    # Launch animated plot (blocks until window closed)
    launch_plot()