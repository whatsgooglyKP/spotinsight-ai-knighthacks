"""
RoadGuard Agent: Lightweight Computer Vision Feature Extraction.
Extracts edge density, surface distress, thermal intensity, and void ratios
from SPOT robot and drone inspection imagery.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
from PIL import Image, ImageFilter, ImageOps


class VisionFeatureExtractor:
    """
    Extracts quantitative visual distress features from inspection images
    without requiring heavy deep learning weights, making it ideal for edge SPOT payloads.
    """

    def __init__(self):
        pass

    def extract_features_from_image(
        self, image_input: Union[str, Path, Image.Image]
    ) -> Dict[str, float]:
        """
        Extracts structural distress features:
        - cv_edge_density: High-frequency crack/fracture texture percentage
        - cv_surface_void_ratio: Pothole / dark void pixel ratio
        - cv_thermal_hotspot_ratio: Saturated thermal/hotspot pixel ratio
        - cv_luminance_contrast: Structural surface roughness/heterogeneity
        - cv_overall_distress_score: Composite 0-100 distress index
        """
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return self._default_features()
            img = Image.open(p).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            return self._default_features()

        # Resize to standard analysis resolution (256x256)
        img_resized = img.resize((256, 256))
        np_rgb = np.array(img_resized, dtype=np.float32)

        # 1. Grayscale & Luminance Contrast
        gray = ImageOps.grayscale(img_resized)
        np_gray = np.array(gray, dtype=np.float32)
        mean_lum = float(np.mean(np_gray))
        contrast_lum = float(np.std(np_gray))

        # 2. Edge / Crack Texture Detection via High-Pass Filtering
        edge_img = gray.filter(ImageFilter.FIND_EDGES)
        np_edges = np.array(edge_img, dtype=np.float32)
        edge_threshold = 40.0
        edge_density = float(np.mean(np_edges > edge_threshold))

        # 3. Surface Void / Pothole Dark Pixel Ratio
        # Potholes and cavity voids appear as deeply shadowed or low-reflectance regions
        void_threshold = max(20.0, mean_lum - 1.8 * contrast_lum)
        void_ratio = float(np.mean(np_gray < void_threshold))

        # 4. Thermal Hotspot / Oxidation Color Ratio
        # Red-channel dominance over green/blue represents thermal heat signatures or surface rust/wear
        r = np_rgb[:, :, 0]
        g = np_rgb[:, :, 1]
        b = np_rgb[:, :, 2]
        thermal_hotspot_mask = (r > 160.0) & (r > g * 1.25) & (r > b * 1.25)
        thermal_hotspot_ratio = float(np.mean(thermal_hotspot_mask))

        # 5. Composite Visual Distress Score (0.0 to 100.0)
        # Higher score = more severe visual degradation
        composite_score = (
            (edge_density * 45.0)
            + (void_ratio * 35.0)
            + (thermal_hotspot_ratio * 20.0)
        ) * 100.0
        composite_score = float(np.clip(composite_score, 0.0, 100.0))

        return {
            "cv_edge_density": round(edge_density, 4),
            "cv_surface_void_ratio": round(void_ratio, 4),
            "cv_thermal_hotspot_ratio": round(thermal_hotspot_ratio, 4),
            "cv_luminance_contrast": round(contrast_lum, 2),
            "cv_overall_distress_score": round(composite_score, 2),
        }

    def _default_features(self) -> Dict[str, float]:
        return {
            "cv_edge_density": 0.05,
            "cv_surface_void_ratio": 0.02,
            "cv_thermal_hotspot_ratio": 0.0,
            "cv_luminance_contrast": 25.0,
            "cv_overall_distress_score": 10.0,
        }
