"""
Emotion + Mismatch System
EEG: Unicorn Hybrid Black + g.Pype
CV: Webcam + hsemotion-onnx

Outputs:
- Live valence/arousal plot
- Face emotion
- EEG biomarkers
- I6 mismatch event when EEG and face disagree
"""

import threading
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import gpype as gp


# ============================================================
# CONFIG
# ============================================================

WINDOW_SECONDS = 2
EEG_SAMPLE_RATE = 250

FUSION_EEG_W = 0.5
FUSION_FACE_W = 0.5

MISMATCH_COOLDOWN_SEC = 5.0
MISMATCH_SCORE_TH = 1.2
HIGH_SEVERITY_TH = 1.7


# ============================================================
# EMOTION MAP
# ============================================================

EMOTION_TO_VA = {
    "happy":    (0.80, 0.40),
    "sad":      (-0.60, -0.60),
    "angry":    (-0.80, 0.70),
    "contempt": (-0.50, -0.20),
}

LABEL_MAP = {
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "contempt": "contempt",
}

EMOTION_COLORS = {
    "happy": "#FFD700",
    "sad": "#4fc3f7",
    "angry": "#ef5350",
    "contempt": "#ab47bc",
}


# ============================================================
# SHARED STATE
# ============================================================

lock = threading.Lock()

state = {
    "eeg_valence": 0.0,
    "eeg_arousal": 0.0,
    "eeg_cognitive_load": 0.0,

    "alpha_power": 0.0,
    "beta_power": 0.0,
    "theta_power": 0.0,

    "face_valence": 0.0,
    "face_arousal": 0.0,
    "face_emotion": "none",
    "face_detected": False,

    "final_valence": 0.0,
    "final_arousal": 0.0,
    "final_emotion": "happy",
}

valence_history = deque(maxlen=10)
arousal_history = deque(maxlen=10)

last_mismatch_time = 0.0


# ============================================================
# EEG HELPERS
# ============================================================

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def record_gpype_phase(node_dict, duration):
    samples = {key: [] for key in node_dict.keys()}
    start_time = time.time()

    while time.time() - start_time < duration:
        for key, node in node_dict.items():
            val = node.get_value()
            if val is not None and val > 0:
                samples[key].append(val)
        time.sleep(0.1)

    return {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in samples.items()
    }


def calibrate_gpype_abt(abt_node_dict, duration=15):
    import winsound

    print("\n--- EEG CALIBRATION PHASE 1: EYES CLOSED ---")
    input("Press Enter, then CLOSE YOUR EYES immediately.")

    eyes_closed_avgs = record_gpype_phase(abt_node_dict, duration)

    winsound.Beep(1000, 500)
    print("DONE. You can open your eyes.")

    print("\n--- EEG CALIBRATION PHASE 2: MENTAL MATH ---")
    print("Task: Count backwards from 1000 by 7s in your head.")
    input("Press Enter to start.")

    math_avgs = record_gpype_phase(abt_node_dict, duration)

    winsound.Beep(1000, 500)
    print("EEG CALIBRATION COMPLETE.\n")

    return {
        "alpha": {
            "min": math_avgs["alpha"],
            "max": eyes_closed_avgs["alpha"],
        },
        "beta": {
            "min": eyes_closed_avgs["beta"],
            "max": math_avgs["beta"],
        },
        "theta": {
            "min": eyes_closed_avgs["theta"],
            "max": math_avgs["theta"],
        },
    }


def compute_eeg_biomarkers(scores):
    eeg_arousal = scores["beta"]
    eeg_cognitive_load = scores["theta"]

    # Demo-friendly valence proxy:
    # alpha = calmer/more positive
    # theta = effort/strain
    eeg_valence = scores["alpha"] - scores["theta"]
    eeg_valence = clamp(eeg_valence)

    return {
        "arousal": eeg_arousal,
        "valence": eeg_valence,
        "cognitive_load": eeg_cognitive_load,
    }


def make_i1_message(biomarkers):
    return {
        "type": "biomarker_tick",
        "source": "I1",
        "ts_py": time.time(),
        "payload": {
            "arousal": biomarkers["arousal"],
            "valence": biomarkers["valence"],
            "cognitive_load": biomarkers["cognitive_load"],
            "signal_quality": "ok",
            "baseline_ready": True,
        }
    }


# ============================================================
# EEG THREAD: REAL G.PYPE PIPELINE
# ============================================================

def eeg_thread():
    print("[EEG] Starting g.Pype pipeline...")

    app = gp.MainApp()
    pipeline = gp.Pipeline()

    source = gp.UnicornSource(device_id=0)
    pipeline.add_node(source)

    bandpass = gp.BandpassFilter(f_low=1.0, f_high=50.0, order=4)
    notch = gp.NotchFilter(f_center=50.0, bandwidth=2.0)

    pipeline.add_node(bandpass, input_node=source)
    pipeline.add_node(notch, input_node=bandpass)

    # Unicorn layout from your docs/code:
    # 0=Fz, 1=C3, 2=Cz, 3=C4, 4=Pz, 5=PO7, 6=Oz, 7=PO8
    frontal_node = gp.Router(input_channels=[0])      # Fz
    occipital_node = gp.Router(input_channels=[6])    # Oz

    pipeline.add_node(frontal_node, input_node=notch)
    pipeline.add_node(occipital_node, input_node=notch)

    theta_power = gp.ThetaPower(smoothing=0.5)
    alpha_power = gp.AlphaPower(smoothing=0.5)
    beta_power = gp.BetaPower(smoothing=0.5)

    pipeline.add_node(theta_power, input_node=frontal_node)
    pipeline.add_node(alpha_power, input_node=occipital_node)
    pipeline.add_node(beta_power, input_node=source)

    theta_smooth = gp.MovingAverage(window_size=250)
    alpha_smooth = gp.MovingAverage(window_size=250)
    beta_smooth = gp.MovingAverage(window_size=250)

    pipeline.add_node(theta_smooth, input_node=theta_power)
    pipeline.add_node(alpha_smooth, input_node=alpha_power)
    pipeline.add_node(beta_smooth, input_node=beta_power)

    pipeline.start()
    print("[EEG] Pipeline started.")

    abt_nodes = {
        "alpha": alpha_smooth,
        "beta": beta_smooth,
        "theta": theta_smooth,
    }

    time.sleep(2)
    limits = calibrate_gpype_abt(abt_nodes, duration=15)

    try:
        while True:
            cur = {
                key: node.get_value() or 0.0
                for key, node in abt_nodes.items()
            }

            scores = {}

            for band in ["alpha", "beta", "theta"]:
                mn = limits[band]["min"]
                mx = limits[band]["max"]
                raw_score = (cur[band] - mn) / (mx - mn) if (mx - mn) != 0 else 0.0
                scores[band] = max(0.0, min(1.0, raw_score))

            biomarkers = compute_eeg_biomarkers(scores)
            i1_msg = make_i1_message(biomarkers)

            with lock:
                state["alpha_power"] = scores["alpha"]
                state["beta_power"] = scores["beta"]
                state["theta_power"] = scores["theta"]

                state["eeg_valence"] = biomarkers["valence"]
                state["eeg_arousal"] = biomarkers["arousal"]
                state["eeg_cognitive_load"] = biomarkers["cognitive_load"]

            # Optional: print compact EEG status
            print(
                f"[EEG] V={biomarkers['valence']:+.2f} "
                f"A={biomarkers['arousal']:+.2f} "
                f"C={biomarkers['cognitive_load']:+.2f}",
                end="\r"
            )

            time.sleep(0.5)

    except KeyboardInterrupt:
        pipeline.stop()
    except Exception as e:
        print(f"\n[EEG] Error: {e}")
        pipeline.stop()


# ============================================================
# FACE THREAD: WEBCAM + HSEMOTION
# ============================================================

def face_thread():
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    except ImportError:
        print("[FACE] hsemotion-onnx not found.")
        return

    print("[FACE] Loading hsemotion model...")
    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[FACE] Could not open webcam.")
        return

    print("[FACE] Webcam active.")

    while True:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            with lock:
                state["face_detected"] = False
            time.sleep(0.1)
            continue

        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        try:
            raw_emotion, scores = recognizer.predict_emotions(face_rgb, logits=False)
            raw_emotion = raw_emotion.lower().strip()

            emotion = LABEL_MAP.get(raw_emotion)

            if emotion in EMOTION_TO_VA:
                valence, arousal = EMOTION_TO_VA[emotion]

                with lock:
                    state["face_emotion"] = emotion
                    state["face_valence"] = valence
                    state["face_arousal"] = arousal
                    state["face_detected"] = True
            else:
                with lock:
                    state["face_emotion"] = raw_emotion
                    state["face_detected"] = False

        except Exception as e:
            print(f"\n[FACE] Error: {e}")
            with lock:
                state["face_detected"] = False

        time.sleep(0.1)

    cap.release()


# ============================================================
# FUSION FOR VISUALIZATION ONLY
# ============================================================

def closest_emotion(v, a):
    best = "happy"
    best_dist = float("inf")

    for name, (ev, ea) in EMOTION_TO_VA.items():
        d = (v - ev) ** 2 + (a - ea) ** 2
        if d < best_dist:
            best_dist = d
            best = name

    return best


def fuse_for_plot():
    with lock:
        eeg_v = state["eeg_valence"]
        eeg_a = state["eeg_arousal"]
        face_v = state["face_valence"]
        face_a = state["face_arousal"]
        face_detected = state["face_detected"]

    if face_detected:
        ew = FUSION_EEG_W
        fw = FUSION_FACE_W
    else:
        ew = 1.0
        fw = 0.0

    final_v = ew * eeg_v + fw * face_v
    final_a = ew * eeg_a + fw * face_a

    valence_history.append(final_v)
    arousal_history.append(final_a)

    smooth_v = float(np.mean(valence_history))
    smooth_a = float(np.mean(arousal_history))

    emotion = closest_emotion(smooth_v, smooth_a)

    with lock:
        state["final_valence"] = smooth_v
        state["final_arousal"] = smooth_a
        state["final_emotion"] = emotion


# ============================================================
# MISMATCH DETECTION: PRODUCT LOGIC
# ============================================================

def detect_mismatch_from_state():
    global last_mismatch_time

    now = time.time()

    if now - last_mismatch_time < MISMATCH_COOLDOWN_SEC:
        return None

    with lock:
        eeg_v = state["eeg_valence"]
        eeg_a = state["eeg_arousal"]
        face_v = state["face_valence"]
        face_a = state["face_arousal"]
        face_detected = state["face_detected"]
        face_emotion = state["face_emotion"]

    if not face_detected:
        return None

    valence_gap = abs(face_v - eeg_v)
    arousal_gap = abs(face_a - eeg_a)

    score = 0.65 * valence_gap + 0.35 * arousal_gap

    if score < MISMATCH_SCORE_TH:
        return None

    severity = "high" if score >= HIGH_SEVERITY_TH else "medium"
    action = "gentle_lay_down" if severity == "high" else "tilt_head"

    last_mismatch_time = now

    return {
        "type": "mismatch_triggered",
        "source": "I6",
        "ts_py": now,
        "payload": {
            "severity": severity,
            "suggested_dog_action": action,
            "cooldown_sec": MISMATCH_COOLDOWN_SEC,
            "debug": {
                "face_emotion": face_emotion,
                "face_valence": round(face_v, 2),
                "face_arousal": round(face_a, 2),
                "eeg_valence": round(eeg_v, 2),
                "eeg_arousal": round(eeg_a, 2),
                "score": round(score, 2),
            }
        }
    }


# ============================================================
# PLOT
# ============================================================

def launch_plot():
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    for name, (ev, ea) in EMOTION_TO_VA.items():
        color = EMOTION_COLORS[name]
        ax.plot(ev, ea, "s", color=color, markersize=14, alpha=0.5)
        ax.text(
            ev + 0.06,
            ea + 0.08,
            name.upper(),
            fontsize=10,
            color=color,
            fontfamily="monospace",
            fontweight="bold",
        )

    dot, = ax.plot(
        [],
        [],
        "o",
        markersize=20,
        markeredgecolor="white",
        markeredgewidth=2,
    )

    trail_x = deque(maxlen=40)
    trail_y = deque(maxlen=40)
    trail_line, = ax.plot([], [], "-", alpha=0.25, linewidth=2, color="white")

    emotion_text = ax.text(
        0,
        1.3,
        "",
        ha="center",
        fontsize=18,
        color="white",
        fontfamily="monospace",
        fontweight="bold",
    )

    info_text = ax.text(
        -1.45,
        -1.42,
        "",
        fontsize=8.5,
        color="#888888",
        fontfamily="monospace",
    )

    mismatch_text = ax.text(
        0,
        -1.3,
        "",
        ha="center",
        fontsize=11,
        color="#ffcc00",
        fontfamily="monospace",
        fontweight="bold",
    )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(0, color="#333333", linewidth=1)

    ax.set_xlabel(
        "Valence  ← Negative | Positive →",
        color="#666666",
        fontsize=10,
    )
    ax.set_ylabel("Arousal ↑", color="#666666", fontsize=10)
    ax.tick_params(colors="#444444")

    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    ax.set_title(
        "Real-Time Emotion + Mismatch Detection",
        color="white",
        fontsize=13,
        pad=15,
        fontfamily="monospace",
    )

    last_event_holder = {"text": "", "time": 0.0}

    def update(frame):
        fuse_for_plot()

        event = detect_mismatch_from_state()

        if event:
            print("\nMISMATCH:", event)
            payload = event["payload"]
            last_event_holder["text"] = (
                f"MISMATCH: {payload['severity'].upper()} | "
                f"{payload['suggested_dog_action']}"
            )
            last_event_holder["time"] = time.time()

        with lock:
            v = state["final_valence"]
            a = state["final_arousal"]
            emotion = state["final_emotion"]

            alpha_p = state["alpha_power"]
            beta_p = state["beta_power"]
            theta_p = state["theta_power"]

            eeg_v = state["eeg_valence"]
            eeg_a = state["eeg_arousal"]
            eeg_c = state["eeg_cognitive_load"]

            face_e = state["face_emotion"]
            face_ok = state["face_detected"]
            face_v = state["face_valence"]
            face_a = state["face_arousal"]

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
            f"EEG   α={alpha_p:.2f} β={beta_p:.2f} θ={theta_p:.2f} "
            f"| V={eeg_v:+.2f} A={eeg_a:+.2f} C={eeg_c:+.2f}\n"
            f"FACE  {face_str} | V={face_v:+.2f} A={face_a:+.2f}"
        )

        if time.time() - last_event_holder["time"] < 3.0:
            mismatch_text.set_text(last_event_holder["text"])
        else:
            mismatch_text.set_text("")

        return dot, trail_line, emotion_text, info_text, mismatch_text

    ani = animation.FuncAnimation(fig, update, interval=200, blit=True)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  EEG + WEBCAM EMOTION / MISMATCH SYSTEM")
    print("  Emotions: Happy | Sad | Angry | Contempt")
    print("=" * 55)

    t_eeg = threading.Thread(target=eeg_thread, daemon=True)
    t_eeg.start()

    # EEG calibration happens inside eeg_thread.
    # Wait a little before starting face.
    time.sleep(2)

    t_face = threading.Thread(target=face_thread, daemon=True)
    t_face.start()

    time.sleep(1)

    print("\n[INFO] Launching real-time plot.")
    print("[INFO] Close the plot window to exit.\n")

    launch_plot()