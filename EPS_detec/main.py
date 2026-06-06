from ultralytics import YOLO
import cv2
import numpy as np

# ==========================
# CONFIGURACIÓN
# ==========================

MODEL_PATH = r"YOLO-Weights\ppe.pt"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CONF_THRESHOLD = 0.5

FONT_SCALE = 1.2
FONT_THICKNESS = 3

# ==========================
# CARGAR MODELO
# ==========================

model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    "Protective Helmet",
    "Shield",
    "Jacket",
    "Dust Mask",
    "Eye Wear",
    "Glove",
    "Protective Boots"
]


# ==========================
# LETTERBOX
# ==========================

def create_letterbox(frame, target_width, target_height):

    h, w = frame.shape[:2]

    scale = min(
        target_width / w,
        target_height / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# ==========================
# DETECCIÓN DE VIDEO
# ==========================

def video_detection(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(
            f"No se pudo abrir el video: {video_path}"
        )

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    out = cv2.VideoWriter(
        "output.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (frame_width, frame_height)
    )

    WINDOW_NAME = "YOLO PPE Detection"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    while True:

        success, frame = cap.read()

        if not success:
            print("Fin del video.")
            break

        results = model(frame)

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                cls = int(box.cls[0])

                if cls >= len(CLASS_NAMES):
                    continue

                class_name = CLASS_NAMES[cls]

                label = (
                    f"{class_name} "
                    f"{conf:.2f}"
                )

                # ======================
                # COLOR POR CLASE
                # ======================

                if class_name == "Dust Mask":
                    color = (0, 204, 255)

                elif class_name == "Glove":
                    color = (222, 82, 175)

                elif class_name == "Protective Helmet":
                    color = (0, 149, 255)

                else:
                    color = (85, 45, 255)

                # ======================
                # BOUNDING BOX
                # ======================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # ======================
                # LABEL GRANDE
                # ======================

                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    FONT_THICKNESS
                )

                label_y = max(
                    y1 - text_h - 15,
                    0
                )

                cv2.rectangle(
                    frame,
                    (x1, label_y),
                    (
                        x1 + text_w + 12,
                        label_y + text_h + baseline + 12
                    ),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1 + 6,
                        label_y + text_h + 2
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    (255, 255, 255),
                    FONT_THICKNESS,
                    cv2.LINE_AA
                )

        # Guardar frame original procesado
        out.write(frame)

        # Crear frame para visualización
        frame_show = create_letterbox(
            frame,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )

        cv2.imshow(
            WINDOW_NAME,
            frame_show
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            print(
                "Proceso detenido por el usuario."
            )
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    video_detection(
        r"static\files\demo4.mp4"
    )
