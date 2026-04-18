from ultralytics import YOLO
import cv2
import serial
import time
import argparse

# ─────────────────────────────────────────────
# CONFIG  — edit these if needed
# ─────────────────────────────────────────────
MODEL_1_PATH   = "model_1.pt"
MODEL_2_PATH   = "model_2.pt"
SERIAL_PORT    = "/dev/ttyUSB0"   # Linux (Raspberry Pi). Use /dev/ttyACM0 if this fails.
BAUD_RATE      = 9600
CONF_THRESHOLD = 0.4              # Minimum confidence to count a detection
SEND_INTERVAL  = 5                # Seconds between serial sends
OUTPUT_FILE    = "output.mp4"

# FSL class labels (both models share the same 5 classes)
FSL_CLASSES = {"Mahal Kita", "Paumanhin", "Pinsan", "Salamat", "Walang Anuman"}

# Optional: remap model output names if they differ slightly
CLASS_NAME_MAPPING = {
    "mahal kita":    "Mahal Kita",
    "paumanhin":     "Paumanhin",
    "pinsan":        "Pinsan",
    "salamat":       "Salamat",
    "walang anuman": "Walang Anuman",
}

# Box colors per model so you can see which model made each detection
MODEL_COLORS = {
    "model_1": (0, 255, 0),    # Green  — model_1.pt
    "model_2": (255, 165, 0),  # Orange — model_2.pt
}


def normalize_class(name: str) -> str:
    """Lowercase-strip lookup, then return original if not in map."""
    return CLASS_NAME_MAPPING.get(name.lower().strip(), name)


def run_model(model, frame, model_tag: str, annotated: "cv2 frame"):
    """
    Run a single YOLO model on the frame.
    Draws bounding boxes on `annotated` and returns a dict of
    {class_name: best_confidence} for all detections above threshold.
    """
    detections = {}
    results = model.predict(source=frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    color = MODEL_COLORS[model_tag]

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = normalize_class(result.names[class_id])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"[{model_tag}] {class_name} {confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Keep the highest-confidence detection per class across this model
            if class_name not in detections or confidence > detections[class_name]:
                detections[class_name] = confidence

    return detections


def ensemble_detections(det1: dict, det2: dict) -> str | None:
    """
    Merge detections from both models.
    If both models agree on a class, average their confidences (boosts score).
    Return the class with the highest final confidence, or None.
    """
    all_classes = set(det1) | set(det2)
    if not all_classes:
        return None

    scores = {}
    for cls in all_classes:
        c1 = det1.get(cls, 0.0)
        c2 = det2.get(cls, 0.0)
        if c1 > 0 and c2 > 0:
            # Both models agree — reward with an average boost
            scores[cls] = (c1 + c2) / 2 * 1.1
        else:
            scores[cls] = max(c1, c2)

    best_class = max(scores, key=scores.get)
    return best_class


def main():
    parser = argparse.ArgumentParser(description="FSL Dual-Model YOLO + Arduino")
    parser.add_argument("--model1",    default=MODEL_1_PATH,   help="Path to first  .pt file")
    parser.add_argument("--model2",    default=MODEL_2_PATH,   help="Path to second .pt file")
    parser.add_argument("--port",      default=SERIAL_PORT,    help="Serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--conf",      default=CONF_THRESHOLD, type=float, help="Confidence threshold")
    parser.add_argument("--interval",  default=SEND_INTERVAL,  type=int,   help="Serial send interval (s)")
    parser.add_argument("--no-serial", action="store_true",                help="Disable serial (test mode)")
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────
    print("[INFO] Loading model_1 ...")
    model1 = YOLO(args.model1)
    print("[INFO] Loading model_2 ...")
    model2 = YOLO(args.model2)

    # ── Camera setup ─────────────────────────────────────────────
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Could not open camera.")

    fps          = int(video.get(cv2.CAP_PROP_FPS)) or 30   # fallback if cam returns 0
    frame_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    writer       = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

    # ── Serial setup ─────────────────────────────────────────────
    arduino = None
    if not args.no_serial:
        try:
            arduino = serial.Serial(args.port, args.baud_rate if hasattr(args, 'baud_rate') else BAUD_RATE)
            print(f"[INFO] Serial connected on {args.port}")
        except serial.SerialException as e:
            print(f"[WARN] Serial not available: {e}. Running without Arduino.")

    last_sent_time = time.time()

    # ── Legend overlay helper ─────────────────────────────────────
    def draw_legend(frame):
        cv2.rectangle(frame, (8, 8), (230, 50), (0, 0, 0), -1)
        cv2.putText(frame, "Green  = model_1.pt", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, "Orange = model_2.pt", (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    # ── Main loop ────────────────────────────────────────────────
    print("[INFO] Starting detection. Press 'q' to quit.")
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("[WARN] Failed to read frame. Exiting.")
                break

            annotated = frame.copy()

            # Run both models
            det1 = run_model(model1, frame, "model_1", annotated)
            det2 = run_model(model2, frame, "model_2", annotated)

            # Combine results
            best_sign = ensemble_detections(det1, det2)

            # Show agreed label at top of frame
            if best_sign:
                cv2.putText(
                    annotated,
                    f"Sign: {best_sign}",
                    (10, frame_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                )

            draw_legend(annotated)
            writer.write(annotated)
            cv2.imshow("FSL Dual-Model Detection", annotated)

            # ── Serial send ──────────────────────────────────────
            now = time.time()
            if now - last_sent_time >= args.interval:
                if best_sign and arduino and arduino.is_open:
                    arduino.write(best_sign.encode())
                    print(f"[SERIAL] Sent: {best_sign}")
                elif best_sign:
                    print(f"[TEST]   Detected: {best_sign}")
                last_sent_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        writer.release()
        cv2.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()
        print(f"[INFO] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()