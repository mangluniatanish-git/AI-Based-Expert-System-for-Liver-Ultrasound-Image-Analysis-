import json

import cv2
import numpy as np

def load_polygon_json(path):

    try:
        with open(path) as f:
            pts = json.load(f)
    except FileNotFoundError:
        return None
    return np.array(pts, dtype=np.float64)


def scale_polygon(polygon, scale):
    if polygon is None:
        return None
    return np.round(polygon * scale).astype(np.int32)


def polygon_to_contour(polygon):

    return polygon.reshape((-1, 1, 2))


def bbox_of_polygon(polygon, img_shape):

    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    h_img, w_img = img_shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    return x, y, w, h

def detect_candidate_regions(gray_roi):
    parenchyma_mask = (gray_roi > 25).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    parenchyma_mask = cv2.morphologyEx(cv2.morphologyEx(parenchyma_mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

    p_pixels = gray_roi[parenchyma_mask == 255]
    if len(p_pixels) < 1000:
        return []

    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blurred[parenchyma_mask == 255], 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, hypoechoic = cv2.threshold(blurred, otsu_val, 255, cv2.THRESH_BINARY_INV)
    hypoechoic = cv2.bitwise_and(hypoechoic, hypoechoic, mask=parenchyma_mask)

    m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cv2.morphologyEx(hypoechoic, cv2.MORPH_OPEN, m_kernel), cv2.MORPH_CLOSE, m_kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def best_candidate_region(gray_roi, min_area=250, max_area_fraction=0.35, min_contrast=0.10, min_score=0.45):
    h_roi, w_roi = gray_roi.shape[:2]
    roi_area = h_roi * w_roi
    max_area = roi_area * max_area_fraction

    parenchyma_mask = (gray_roi > 25).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    parenchyma_mask = cv2.morphologyEx(cv2.morphologyEx(parenchyma_mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

    p_pixels = gray_roi[parenchyma_mask == 255]
    if len(p_pixels) < 1000:
        return None
    p_mean = float(np.mean(p_pixels))

    candidates = detect_candidate_regions(gray_roi)

    best_contour = None
    best_score = -1

    for contour in candidates:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularity = min(circularity, 1.0)

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue

        aspect_ratio = float(w) / float(h)
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            continue

        extent = area / (w * h)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if circularity < 0.15 or solidity < 0.45:
            continue

        mask = np.zeros(gray_roi.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        pixels = gray_roi[mask == 255]
        if len(pixels) == 0:
            continue

        candidate_mean = float(np.mean(pixels))

        # Reject black non-acoustic background / transducer boundary
        if candidate_mean < 25:
            continue

        contrast = abs(candidate_mean - p_mean) / (p_mean + 1e-6)
        if contrast < min_contrast:
            continue

        area_score = min(area / (roi_area * 0.06), 1.0)

        score = (
            0.35 * min(contrast * 2.5, 1.0) +
            0.25 * area_score +
            0.15 * circularity +
            0.15 * solidity +
            0.10 * extent
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_score >= min_score:
        return best_contour
    return None
