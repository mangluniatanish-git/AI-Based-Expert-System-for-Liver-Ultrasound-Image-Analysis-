import cv2
import numpy as np

TARGET_MAX_DIM = 512


def resize_to_target(bgr):
    """Resize an image so its longest side equals TARGET_MAX_DIM, keeping
    aspect ratio. Returns (resized_bgr, scale_factor_applied)."""
    h, w = bgr.shape[:2]
    scale = TARGET_MAX_DIM / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return bgr, 1.0
    resized = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return resized, scale


def preprocess_image_array(original_bgr):
    resized_bgr, scale = resize_to_target(original_bgr)


    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)


    despeckled = cv2.medianBlur(gray, 5)
    despeckled = cv2.fastNlMeansDenoising(despeckled, h=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)

    normalized = cv2.normalize(despeckled, None, 0, 255, cv2.NORM_MINMAX)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    return resized_bgr, enhanced, scale


def preprocess_image(image_path):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        original_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        original_bgr = None

    if original_bgr is None:
        original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return preprocess_image_array(original_bgr)


def select_liver_roi(enhanced_gray):

    h_img, w_img = enhanced_gray.shape
    _, thresh = cv2.threshold(enhanced_gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, w_img, h_img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w * h < 0.10 * (w_img * h_img):
        return 0, 0, w_img, h_img

    return x, y, w, h
