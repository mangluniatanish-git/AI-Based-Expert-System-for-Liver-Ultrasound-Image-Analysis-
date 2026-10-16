import math
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES = [

    "liver_mean_intensity",
    "liver_std_intensity",
    "liver_glcm_contrast",
    "liver_glcm_energy",
    "liver_glcm_homogeneity",
    "liver_edge_density",

    "mass_detected",

    "mass_relative_area",
    "mass_mean_intensity",
    "mass_std_intensity",
    "mass_intensity_ratio",
    "mass_liver_contrast",
    "mass_glcm_contrast",
    "mass_glcm_energy",
    "mass_glcm_homogeneity",
    "mass_circularity",
    "mass_extent",
    "mass_solidity",
    "mass_aspect_ratio",
    "mass_edge_density",
]

def _glcm_features(patch, levels=32):

    if patch is None:
        return 0.0, 0.0, 0.0

    if patch.size == 0:
        return 0.0, 0.0, 0.0

    if patch.ndim != 2:
        return 0.0, 0.0, 0.0

    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0, 0.0, 0.0

    quantized = (
        patch.astype(np.float64) / 256.0 * levels
    ).astype(np.uint8)

    quantized = np.clip(
        quantized,
        0,
        levels - 1
    )

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=levels,
        symmetric=True,
        normed=True
    )

    contrast = float(
        graycoprops(glcm, "contrast").mean()
    )

    energy = float(
        graycoprops(glcm, "energy").mean()
    )

    homogeneity = float(
        graycoprops(glcm, "homogeneity").mean()
    )

    return contrast, energy, homogeneity

def _edge_density(patch):
    if patch is None:
        return 0.0

    if patch.size == 0:
        return 0.0

    if patch.ndim != 2:
        return 0.0

    edges = cv2.Canny(
        patch,
        40,
        120
    )

    return float(
        np.count_nonzero(edges)
    ) / float(edges.size)


def extract_liver_features(enhanced_gray, liver_bbox):

    x, y, w, h = liver_bbox

    image_h, image_w = enhanced_gray.shape[:2]

    x = max(0, min(x, image_w))
    y = max(0, min(y, image_h))

    w = max(0, min(w, image_w - x))
    h = max(0, min(h, image_h - y))

    liver_roi = enhanced_gray[
        y:y + h,
        x:x + w
    ]

    if liver_roi.size == 0:
        return {
            "liver_mean_intensity": 0.0,
            "liver_std_intensity": 0.0,
            "liver_glcm_contrast": 0.0,
            "liver_glcm_energy": 0.0,
            "liver_glcm_homogeneity": 0.0,
            "liver_edge_density": 0.0,
        }, 1

    mean_intensity = float(
        np.mean(liver_roi)
    )

    std_intensity = float(
        np.std(liver_roi)
    )

    contrast, energy, homogeneity = _glcm_features(
        liver_roi
    )

    edge_density = _edge_density(
        liver_roi
    )

    liver_area = (
        liver_roi.shape[0] *
        liver_roi.shape[1]
    )

    return {
        "liver_mean_intensity": round(
            mean_intensity, 3
        ),

        "liver_std_intensity": round(
            std_intensity, 3
        ),

        "liver_glcm_contrast": round(
            contrast, 3
        ),

        "liver_glcm_energy": round(
            energy, 4
        ),

        "liver_glcm_homogeneity": round(
            homogeneity, 4
        ),

        "liver_edge_density": round(
            edge_density, 4
        ),
    }, liver_area

def extract_mass_features(
    enhanced_gray,
    mass_contour,
    liver_area,
    liver_mean_intensity
):

    if mass_contour is None:
        return empty_mass_features()

    if len(mass_contour) < 3:
        return empty_mass_features()

    area = float(
        cv2.contourArea(mass_contour)
    )

    perimeter = float(
        cv2.arcLength(
            mass_contour,
            True
        )
    )

    x, y, w, h = cv2.boundingRect(
        mass_contour
    )

    if area <= 0 or w <= 0 or h <= 0:
        return empty_mass_features()

    if perimeter > 0:
        circularity = (
            4.0 * math.pi * area /
            (perimeter ** 2)
        )

        circularity = min(
            float(circularity),
            1.0
        )
    else:
        circularity = 0.0

    bbox_area = float(w * h)

    extent = (
        area / bbox_area
        if bbox_area > 0
        else 0.0
    )

    hull = cv2.convexHull(
        mass_contour
    )

    hull_area = float(
        cv2.contourArea(hull)
    )

    solidity = (
        area / hull_area
        if hull_area > 0
        else 0.0
    )

    aspect_ratio = (
        float(w) / float(h)
        if h > 0
        else 0.0
    )

    mask = np.zeros(
        enhanced_gray.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [mass_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    mass_pixels = enhanced_gray[
        mask == 255
    ]

    if mass_pixels.size == 0:
        return empty_mass_features()

    mean_intensity = float(
        np.mean(mass_pixels)
    )

    std_intensity = float(
        np.std(mass_pixels)
    )

    if liver_mean_intensity > 1e-6:
        intensity_ratio = (
            mean_intensity /
            liver_mean_intensity
        )
    else:
        intensity_ratio = 0.0

    if liver_mean_intensity > 1e-6:
        liver_contrast = (
            abs(
                mean_intensity -
                liver_mean_intensity
            )
            / liver_mean_intensity
        )
    else:
        liver_contrast = 0.0

    patch = enhanced_gray[
        y:y + h,
        x:x + w
    ]

    if patch.size == 0:
        contrast = 0.0
        energy = 0.0
        homogeneity = 0.0
        edge_density = 0.0

    else:

        local_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        local_contour = (
            mass_contour.copy()
        )

        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y

        cv2.drawContours(
            local_mask,
            [local_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        lesion_pixels_local = patch[
            local_mask == 255
        ]

        if lesion_pixels_local.size > 1:

            masked_patch = patch.copy()

            lesion_mean_local = float(
                np.mean(lesion_pixels_local)
            )

            masked_patch[
                local_mask == 0
            ] = int(
                np.clip(
                    lesion_mean_local,
                    0,
                    255
                )
            )

            contrast, energy, homogeneity = (
                _glcm_features(masked_patch)
            )

            edge_density = _edge_density(
                masked_patch
            )

        else:

            contrast = 0.0
            energy = 0.0
            homogeneity = 0.0
            edge_density = 0.0

    return {
        "mass_detected": 1,

        "mass_relative_area": round(
            area / liver_area,
            5
        ) if liver_area else 0.0,

        "mass_mean_intensity": round(
            mean_intensity,
            3
        ),

        "mass_std_intensity": round(
            std_intensity,
            3
        ),

        "mass_intensity_ratio": round(
            intensity_ratio,
            4
        ),

        "mass_liver_contrast": round(
            liver_contrast,
            4
        ),

        "mass_glcm_contrast": round(
            contrast,
            3
        ),

        "mass_glcm_energy": round(
            energy,
            4
        ),

        "mass_glcm_homogeneity": round(
            homogeneity,
            4
        ),

        "mass_circularity": round(
            circularity,
            4
        ),

        "mass_extent": round(
            extent,
            4
        ),

        "mass_solidity": round(
            solidity,
            4
        ),

        "mass_aspect_ratio": round(
            aspect_ratio,
            4
        ),

        "mass_edge_density": round(
            edge_density,
            4
        ),

        "bbox": (
            x,
            y,
            w,
            h
        ),
    }


def empty_mass_features():

    return {
        "mass_detected": 0,

        "mass_relative_area": 0.0,

        "mass_mean_intensity": 0.0,

        "mass_std_intensity": 0.0,

        "mass_intensity_ratio": 0.0,

        "mass_liver_contrast": 0.0,

        "mass_glcm_contrast": 0.0,

        "mass_glcm_energy": 0.0,

        "mass_glcm_homogeneity": 0.0,

        "mass_circularity": 0.0,

        "mass_extent": 0.0,

        "mass_solidity": 0.0,

        "mass_aspect_ratio": 0.0,

        "mass_edge_density": 0.0,

        "bbox": None,
    }


def features_dict_to_vector(features):

    return np.array(
        [
            features[name]
            for name in FEATURE_NAMES
        ],
        dtype=np.float64
    )