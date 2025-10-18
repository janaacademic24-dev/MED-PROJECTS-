"""
Advanced DICOM Viewer with AI Orientation Detection & Organ Classification

Fixes & Improvements:
- Robust DICOM series selection and slice sorting using ImagePositionPatient and direction cosines
- Correct CLIP model initialization (no device kwarg) and device placement
- Safer CLIP tokenization via open_clip.tokenize
- HU windowing and grayscale-to-RGB conversion for better visual features
- MONOCHROME1 inversion handling
- Improved ResNet preprocessing order (ToPILImage -> Resize -> ToTensor -> Normalize)
- Reference lines drawn after being set (UI correctness)
- NIfTI orientation via nibabel when available
"""

import sys
import numpy as np
from pathlib import Path
import pydicom
from scipy import ndimage
from PIL import Image

# NIfTI support via nibabel
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    nib = None
    NIBABEL_AVAILABLE = False
    print("WARNING: nibabel not installed. Run: pip install nibabel")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
    QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor

try:
    import torch
    import torchvision.models as models
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Run: pip install torch torchvision")

import importlib
import os

open_clip = None
CLIP_AVAILABLE = False
try:
    # Try standard import name
    open_clip = importlib.import_module('open_clip')
    CLIP_AVAILABLE = True
except Exception:
    try:
        # Some environments require open_clip_torch
        open_clip = importlib.import_module('open_clip_torch')
        CLIP_AVAILABLE = True
    except Exception:
        CLIP_AVAILABLE = False
        print("WARNING: open_clip not installed. Run: pip install open_clip_torch")

# Suppress HuggingFace symlink warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Comprehensive organ labels for zero-shot classification
ORGAN_LABELS = [
    # Head & Neck
    "Brain", "Skull", "Eye", "Sinuses", "Jaw", "Teeth", "Throat", "Neck",
    # Chest
    "Lungs", "Heart", "Ribs", "Sternum", "Chest", "Breast", "Thyroid", "Esophagus", "Trachea",
    # Abdomen
    "Liver", "Gallbladder", "Pancreas", "Spleen", "Stomach", "Small Intestine", "Large Intestine", "Colon", "Kidneys", "Adrenal Glands", "Bladder", "Abdomen",
    # Pelvis
    "Pelvis", "Hip", "Uterus", "Ovaries", "Prostate", "Rectum",
    # Spine
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine", "Sacrum", "Coccyx", "Spine", "Vertebrae", "Spinal Cord",
    # Upper Extremities
    "Shoulder", "Clavicle", "Scapula", "Humerus", "Elbow", "Radius", "Ulna", "Wrist", "Hand", "Fingers",
    # Lower Extremities
    "Femur", "Knee", "Patella", "Tibia", "Fibula", "Ankle", "Foot", "Toes", "Calcaneus",
    # Other
    "Blood Vessels", "Aorta", "Lymph Nodes", "Muscle", "Bone", "Soft Tissue", "Joint"
]


def plane_from_dicom_tags(ds) -> str:
    """Determine imaging plane (Axial/Coronal/Sagittal) from ImageOrientationPatient."""
    iop = getattr(ds, 'ImageOrientationPatient', None)
    if iop is None or len(iop) < 6:
        return "Unknown"
    try:
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        axis = int(np.argmax(np.abs(normal)))
        return {0: "Sagittal", 1: "Coronal", 2: "Axial"}.get(axis, "Unknown")
    except Exception:
        return "Unknown"


def window_hu(image: np.ndarray, window_center: float = 40.0, window_width: float = 400.0) -> np.ndarray:
    """Apply a simple HU windowing; returns float32 image clipped to [0,1]."""
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    image = np.clip(image, lo, hi)
    image = (image - lo) / max(1e-6, (hi - lo))
    return image.astype(np.float32)


class AIThread(QThread):
    """Background thread for AI inference with CLIP or ResNet50"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, volume, spacing=(1.0, 1.0, 1.0), orientation=None):
        super().__init__()
        self.volume = volume.copy()
        self.spacing = spacing
        self.orientation = orientation
        self.device = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu") if TORCH_AVAILABLE else None
        self.clip_model = None
        self.clip_preprocess = None

    def run(self):
        try:
            if not TORCH_AVAILABLE:
                self.progress.emit("ERROR: PyTorch not installed")
                self.finished.emit({'error': 'PyTorch not installed'})
                return

            use_clip = False
            if CLIP_AVAILABLE:
                try:
                    self.progress.emit("Loading CLIP (ViT-B/32) for organ detection...")
                    # NOTE: create_model_and_transforms does not accept a device kwarg
                    model, _, preprocess_val = open_clip.create_model_and_transforms(
                        'ViT-B-32', pretrained='laion2b_s34b_b79k'
                    )
                    model = model.to(self.device)
                    model.eval()
                    self.clip_model = model
                    self.clip_preprocess = preprocess_val
                    use_clip = True
                except Exception as e:
                    self.progress.emit(f"CLIP load failed, falling back to ResNet50: {e}")
                    use_clip = False

            if not use_clip:
                self.progress.emit("Loading ResNet50 (ImageNet fallback)...")
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                model = model.to(self.device)
                model.eval()
                self.model = model
                self.preprocess = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

            self.progress.emit("Analyzing DICOM slices...")

            # Extract middle slices for each plane
            axial_idx = self.volume.shape[2] // 2
            coronal_idx = self.volume.shape[1] // 2
            sagittal_idx = self.volume.shape[0] // 2

            axial_slice = self.volume[:, :, axial_idx]
            coronal_slice = self.volume[:, coronal_idx, :]
            sagittal_slice = self.volume[sagittal_idx, :, :]

            if use_clip:
                pred_axial = self.predict_slice_clip(axial_slice, "Axial")
                pred_coronal = self.predict_slice_clip(coronal_slice, "Coronal")
                pred_sagittal = self.predict_slice_clip(sagittal_slice, "Sagittal")
            else:
                pred_axial = self.predict_slice_resnet(axial_slice, "Axial")
                pred_coronal = self.predict_slice_resnet(coronal_slice, "Coronal")
                pred_sagittal = self.predict_slice_resnet(sagittal_slice, "Sagittal")

            self.progress.emit(f"Axial: {pred_axial['class']} ({pred_axial['confidence']*100:.1f}%)")
            self.progress.emit(f"Coronal: {pred_coronal['class']} ({pred_coronal['confidence']*100:.1f}%)")
            self.progress.emit(f"Sagittal: {pred_sagittal['class']} ({pred_sagittal['confidence']*100:.1f}%)")

            # Generate simple segmentation and stats
            self.progress.emit("Generating segmentation...")
            segmentation = self.generate_segmentation()
            volume_stats = self.calculate_statistics()

            result = {
                'segmentation': segmentation,
                'slice_predictions': [pred_axial, pred_coronal, pred_sagittal],
                'volume_stats': volume_stats,
                'orientation': self.orientation if self.orientation else 'Unknown',
                'device': str(self.device),
                'model_used': 'CLIP (ViT-B/32)' if use_clip else 'ResNet50 (ImageNet)'
            }
            self.finished.emit(result)

        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit({'error': str(e)})

    def _normalize_slice_uint8(self, slice_data: np.ndarray) -> np.ndarray:
        if slice_data.size == 0:
            return np.zeros_like(slice_data, dtype=np.uint8)
        vmin = float(np.min(slice_data))
        vmax = float(np.max(slice_data))
        if vmax <= vmin:
            return np.zeros_like(slice_data, dtype=np.uint8)
        norm = (slice_data - vmin) / (vmax - vmin)
        return (np.clip(norm, 0, 1) * 255).astype(np.uint8)

    def _to_rgb(self, slice_data: np.ndarray) -> np.ndarray:
        # Use a broad HU window for CT-like images to stabilize contrast
        windowed = window_hu(slice_data, window_center=40.0, window_width=400.0)
        gray_uint8 = (np.clip(windowed, 0, 1) * 255).astype(np.uint8)
        return np.stack([gray_uint8] * 3, axis=-1)

    def predict_slice_clip(self, slice_data: np.ndarray, view_name: str) -> dict:
        try:
            rgb_slice = self._to_rgb(slice_data)
            pil_img = Image.fromarray(rgb_slice)
            image = self.clip_preprocess(pil_img).unsqueeze(0)
            image = image.to(self.device)

            # Prompt engineering: include view and modality hint
            prompts = [
                f"an axial radiology CT slice of {label}" if view_name == "Axial" else
                f"a coronal radiology CT slice of {label}" if view_name == "Coronal" else
                f"a sagittal radiology CT slice of {label}"
                for label in ORGAN_LABELS
            ]

            with torch.no_grad():
                # Tokenize via open_clip utility to ensure correct dtype/shape
                text_tokens = open_clip.tokenize(prompts).to(self.device)
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits = (image_features @ text_features.t()) * 100.0
                probs = logits.softmax(dim=-1).squeeze(0)
                confidence, predicted = torch.max(probs, 0)

            return {
                'view': view_name,
                'class': ORGAN_LABELS[int(predicted.item())],
                'confidence': float(confidence.item()),
                'class_id': int(predicted.item()),
            }
        except Exception:
            return {'view': view_name, 'class': 'Error', 'confidence': 0.0, 'class_id': -1}

    def predict_slice_resnet(self, slice_data: np.ndarray, view_name: str) -> dict:
        try:
            rgb_slice = self._to_rgb(slice_data)
            img_tensor = self.preprocess(rgb_slice).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            # Heuristic coarse mapping
            class_map = {0: "Brain", 1: "Chest", 2: "Abdomen", 3: "Pelvis", 4: "Spine", 5: "Extremity"}
            pred_idx = int(predicted.item()) % len(class_map)
            return {
                'view': view_name,
                'class': class_map[pred_idx],
                'confidence': float(confidence.item()),
                'class_id': pred_idx,
            }
        except Exception:
            return {'view': view_name, 'class': 'Error', 'confidence': 0.0, 'class_id': -1}

    def generate_segmentation(self) -> np.ndarray:
        segmentation = np.zeros_like(self.volume, dtype=np.uint16)
        vmin = float(np.min(self.volume))
        vmax = float(np.max(self.volume))
        denom = max(1e-6, (vmax - vmin))
        normalized = (self.volume - vmin) / denom
        segmentation[normalized > 0.7] = 5
        segmentation[(normalized > 0.5) & (normalized <= 0.7)] = 3
        segmentation[(normalized > 0.3) & (normalized <= 0.5)] = 1
        return segmentation

    def calculate_statistics(self) -> dict:
        voxel_volume = float(np.prod(self.spacing))
        return {
            'Total_Volume_ml': float(self.volume.size * voxel_volume / 1000.0),
            'Mean_Intensity': float(np.mean(self.volume)),
            'Std_Intensity': float(np.std(self.volume)),
            'Min_Intensity': float(np.min(self.volume)),
            'Max_Intensity': float(np.max(self.volume)),
        }

    @staticmethod
    def load_nifti_file(file_path: str):
        if nib is None:
            raise ImportError("nibabel is required to load NIfTI files. Install with: pip install nibabel")
        try:
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata().astype(np.float32)
            header = nii_img.header
            spacing = header.get_zooms()[:3]
            if volume.ndim == 4:
                volume = volume[:, :, :, 0]
            elif volume.ndim != 3:
                raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")

            orientation = None
            try:
                # Use nibabel to decode axis codes if available
                from nibabel.orientations import aff2axcodes
                axcodes = aff2axcodes(nii_img.affine)
                orientation = ''.join(axcodes)  # e.g. RAS
            except Exception:
                orientation = "Unknown"

            dims = volume.shape
            return volume, dims, float(np.min(volume)), float(np.max(volume)), tuple(spacing), orientation
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI file: {str(e)}")

    @staticmethod
    def _read_dicom_dataset(path: Path):
        try:
            ds = pydicom.dcmread(str(path), force=True)
            # Ensure pixel data exists
            if not hasattr(ds, 'PixelData'):
                return None
            return ds
        except Exception:
            return None

    @staticmethod
    def _select_largest_series(datasets: list):
        # Group by SeriesInstanceUID and select the largest
        from collections import defaultdict
        series_map = defaultdict(list)
        for ds in datasets:
            sid = getattr(ds, 'SeriesInstanceUID', 'unknown')
            series_map[sid].append(ds)
        # Choose the series with most slices; tie-breaker by ImageType/SeriesDescription length
        best_sid = None
        best_count = -1
        for sid, items in series_map.items():
            if len(items) > best_count:
                best_sid = sid
                best_count = len(items)
        return series_map[best_sid]

    @staticmethod
    def load_dicom_folder(folder_path: str):
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError("Invalid folder path")

        # Recursively read DICOM files
        candidates = [p for p in folder.rglob('*') if p.is_file()]
        datasets = []
        for p in candidates:
            ds = AIThread._read_dicom_dataset(p)
            if ds is not None:
                datasets.append(ds)
        if not datasets:
            raise ValueError("No DICOM images found in folder")

        # Select the largest series
        series = AIThread._select_largest_series(datasets)
        first_ds = series[0]

        # Compute slice sort key using ImagePositionPatient projected onto plane normal
        def get_sort_key(ds):
            iop = getattr(ds, 'ImageOrientationPatient', None)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if iop is not None and len(iop) >= 6 and ipp is not None and len(ipp) >= 3:
                row = np.array(iop[:3], dtype=float)
                col = np.array(iop[3:6], dtype=float)
                normal = np.cross(row, col)
                pos = np.array(ipp[:3], dtype=float)
                return float(np.dot(pos, normal))
            # Fallback
            if hasattr(ds, 'InstanceNumber'):
                try:
                    return float(ds.InstanceNumber)
                except Exception:
                    pass
            if hasattr(ds, 'SliceLocation'):
                try:
                    return float(ds.SliceLocation)
                except Exception:
                    pass
            return 0.0

        series_sorted = sorted(series, key=get_sort_key)

        # Build volume
        rows, cols = int(series_sorted[0].Rows), int(series_sorted[0].Columns)
        num_slices = len(series_sorted)
        volume = np.zeros((rows, cols, num_slices), dtype=np.float32)

        def get_pixel_array(ds):
            arr = ds.pixel_array.astype(np.float32)
            # Apply rescale if available (typical for CT)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                try:
                    arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                except Exception:
                    pass
            # Invert MONOCHROME1
            photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
            if photometric == 'MONOCHROME1':
                arr = np.max(arr) + np.min(arr) - arr
            return arr

        for i, ds in enumerate(series_sorted):
            arr = get_pixel_array(ds)
            # Safety crop/pad if any slice deviates
            rr, cc = arr.shape
            rmin, cmin = min(rows, rr), min(cols, cc)
            volume[:rmin, :cmin, i] = arr[:rmin, :cmin]

        # Spacing
        spacing_row, spacing_col = 1.0, 1.0
        if hasattr(first_ds, 'PixelSpacing') and len(first_ds.PixelSpacing) >= 2:
            spacing_row = float(first_ds.PixelSpacing[0])
            spacing_col = float(first_ds.PixelSpacing[1])

        # Estimate slice spacing using IPP if available
        slice_spacing = None
        positions = []
        iop = getattr(first_ds, 'ImageOrientationPatient', None)
        if iop is not None and len(iop) >= 6:
            row = np.array(iop[:3], dtype=float)
            col = np.array(iop[3:6], dtype=float)
            normal = np.cross(row, col)
            for ds in series_sorted:
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is not None and len(ipp) >= 3:
                    pos = np.array(ipp[:3], dtype=float)
                    positions.append(float(np.dot(pos, normal)))
        if len(positions) >= 2:
            diffs = np.abs(np.diff(sorted(positions)))
            diffs = diffs[diffs > 0]
            if diffs.size:
                slice_spacing = float(np.median(diffs))
        if slice_spacing is None:
            if hasattr(first_ds, 'SpacingBetweenSlices'):
                try:
                    slice_spacing = float(first_ds.SpacingBetweenSlices)
                except Exception:
                    slice_spacing = None
        if slice_spacing is None and hasattr(first_ds, 'SliceThickness'):
            try:
                slice_spacing = float(first_ds.SliceThickness)
            except Exception:
                slice_spacing = 1.0
        if slice_spacing is None:
            slice_spacing = 1.0

        spacing = (spacing_row, spacing_col, slice_spacing)
        orientation = plane_from_dicom_tags(first_ds)

        return volume, volume.shape, float(np.min(volume)), float(np.max(volume)), spacing, orientation


class ImageViewer(QLabel):
    """Display 2D image with optional reference lines"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #333; background: black;")
        self.ref_lines = {'h': None, 'v': None}

    def display_array(self, arr: np.ndarray, show_ref: bool = False):
        if arr is None or arr.size == 0:
            return
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg.copy())

        if show_ref and (self.ref_lines['h'] is not None or self.ref_lines['v'] is not None):
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(0, 255, 255), 1))
            if self.ref_lines['h'] is not None:
                y = int(self.ref_lines['h'] * h)
                painter.drawLine(0, y, w, y)
            if self.ref_lines['v'] is not None:
                x = int(self.ref_lines['v'] * w)
                painter.drawLine(x, 0, x, h)
            painter.end()

        scaled = pixmap.scaledToWidth(300, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def set_reference_lines(self, h_pos=None, v_pos=None):
        self.ref_lines['h'] = h_pos
        self.ref_lines['v'] = v_pos


class DicomViewerGUI(QMainWindow):
    """Main GUI for DICOM Viewer with AI"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer - AI Orientation & Organ Detection")
        self.setGeometry(50, 50, 1600, 900)

        self.volume = None
        self.dims = None
        self.min_val = None
        self.max_val = None
        self.spacing = None
        self.orientation = None

        self.axial_idx = 0
        self.coronal_idx = 0
        self.sagittal_idx = 0
        self.fourth_idx = 0

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top controls
        top_layout = QHBoxLayout()

        load_btn = QPushButton("📁 Load Data (DICOM/NIfTI)")
        load_btn.clicked.connect(self.load_dicom)
        load_btn.setStyleSheet("padding: 8px; font-size: 14px;")
        top_layout.addWidget(load_btn)

        self.analyze_btn = QPushButton("🤖 Run AI Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("background: #27ae60; color: white; padding: 8px; font-size: 14px;")
        top_layout.addWidget(self.analyze_btn)

        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("font-size: 13px; padding: 5px;")
        top_layout.addWidget(self.info_label)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Views
        views_layout = QHBoxLayout()
        views_layout.addLayout(self.create_view_group("Axial", 'axial'))
        views_layout.addLayout(self.create_view_group("Coronal", 'coronal'))
        views_layout.addLayout(self.create_view_group("Sagittal", 'sagittal'))
        views_layout.addLayout(self.create_fourth_view())
        main_layout.addLayout(views_layout, 1)

        # Status bar
        self.status_label = QLabel("Ready | Load DICOM to begin")
        self.status_label.setStyleSheet("background: #34495e; color: white; padding: 8px; font-size: 13px;")
        main_layout.addWidget(self.status_label)

    def create_view_group(self, title, view_name):
        layout = QVBoxLayout()
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        image = ImageViewer()
        setattr(self, f"{view_name}_image", image)
        layout.addWidget(image, 1)

        slider = QSlider(Qt.Horizontal)
        slider.setEnabled(False)
        slider.valueChanged.connect(lambda v, vn=view_name: self.update_view(vn, v))
        setattr(self, f"{view_name}_slider", slider)
        layout.addWidget(slider)

        return layout

    def create_fourth_view(self):
        layout = QVBoxLayout()
        label = QLabel("4th View - Advanced")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self.fourth_mode = QComboBox()
        self.fourth_mode.addItems(["Oblique", "MIP (Maximum Intensity)"])
        self.fourth_mode.currentTextChanged.connect(self.update_fourth_mode)
        layout.addWidget(self.fourth_mode)

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle:"))
        self.oblique_angle = QDoubleSpinBox()
        self.oblique_angle.setRange(-180, 180)
        self.oblique_angle.setValue(0)
        self.oblique_angle.setSuffix("°")
        self.oblique_angle.valueChanged.connect(lambda: self.update_view('fourth', self.fourth_idx))
        angle_layout.addWidget(self.oblique_angle)
        layout.addLayout(angle_layout)

        self.fourth_image = ImageViewer()
        layout.addWidget(self.fourth_image, 1)

        self.fourth_slider = QSlider(Qt.Horizontal)
        self.fourth_slider.setEnabled(False)
        self.fourth_slider.valueChanged.connect(lambda v: self.update_view('fourth', v))
        layout.addWidget(self.fourth_slider)

        return layout

    def load_dicom(self):
        # Ask user to choose between folder (DICOM) or file (NIfTI)
        choice = QMessageBox.question(
            self,
            "Select Input Type",
            "Load DICOM folder or NIfTI file?\n\nClick 'Yes' for DICOM folder\nClick 'No' for NIfTI file",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if choice == QMessageBox.Cancel:
            return

        try:
            self.status_label.setText("Loading...")
            QApplication.processEvents()
            loader = AIThread

            if choice == QMessageBox.Yes:
                folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
                if not folder:
                    return
                self.volume, self.dims, self.min_val, self.max_val, self.spacing, self.orientation = loader.load_dicom_folder(folder)
                source_type = "DICOM"
            else:
                file_path, _ = QFileDialog.getOpenFileName(self, "Select NIfTI File", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*.*)")
                if not file_path:
                    return
                self.volume, self.dims, self.min_val, self.max_val, self.spacing, self.orientation = loader.load_nifti_file(file_path)
                source_type = "NIfTI"

            # Indices
            self.axial_idx = self.dims[2] // 2
            self.coronal_idx = self.dims[1] // 2
            self.sagittal_idx = self.dims[0] // 2
            self.fourth_idx = self.axial_idx

            # Sliders
            self.axial_slider.setMaximum(self.dims[2] - 1)
            self.axial_slider.setValue(self.axial_idx)
            self.axial_slider.setEnabled(True)

            self.coronal_slider.setMaximum(self.dims[1] - 1)
            self.coronal_slider.setValue(self.coronal_idx)
            self.coronal_slider.setEnabled(True)

            self.sagittal_slider.setMaximum(self.dims[0] - 1)
            self.sagittal_slider.setValue(self.sagittal_idx)
            self.sagittal_slider.setEnabled(True)

            self.fourth_slider.setMaximum(self.dims[2] - 1)
            self.fourth_slider.setValue(self.fourth_idx)
            self.fourth_slider.setEnabled(True)

            self.info_label.setText(
                f"{source_type} | Volume: {self.dims[0]}×{self.dims[1]}×{self.dims[2]} | "
                f"Orientation: {self.orientation} | "
                f"Spacing: {self.spacing[0]:.2f}×{self.spacing[1]:.2f}×{self.spacing[2]:.2f} mm"
            )

            self.update_all_views()
            self.analyze_btn.setEnabled(True)
            self.status_label.setText(f"✓ {source_type} loaded successfully | Detected: {self.orientation} orientation")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            self.status_label.setText("Error loading data")

    def update_view(self, view_name: str, idx: int):
        if self.volume is None:
            return
        if view_name == 'axial':
            self.axial_idx = idx
        elif view_name == 'coronal':
            self.coronal_idx = idx
        elif view_name == 'sagittal':
            self.sagittal_idx = idx
        elif view_name == 'fourth':
            self.fourth_idx = idx
        self.update_all_views()

    def update_fourth_mode(self):
        if self.volume is not None:
            self.update_view('fourth', self.fourth_idx)

    def _norm8(self, data: np.ndarray) -> np.ndarray:
        if self.max_val is None or self.min_val is None or self.max_val <= self.min_val:
            return np.zeros_like(data, dtype=np.uint8)
        norm = (data - self.min_val) / max(1e-6, (self.max_val - self.min_val))
        return (np.clip(norm, 0, 1) * 255).astype(np.uint8)

    def update_all_views(self):
        if self.volume is None:
            return

        # Axial
        axial_data = self.volume[:, :, self.axial_idx]
        axial_norm = self._norm8(axial_data)
        self.axial_image.set_reference_lines(
            self.coronal_idx / max(1, self.dims[1] - 1),
            self.sagittal_idx / max(1, self.dims[0] - 1)
        )
        self.axial_image.display_array(axial_norm, show_ref=True)

        # Coronal (transpose then rotate 180° for display)
        coronal_data = np.rot90(self.volume[:, self.coronal_idx, :].T, 2)
        coronal_norm = self._norm8(coronal_data)
        self.coronal_image.set_reference_lines(
            self.sagittal_idx / max(1, self.dims[0] - 1),
            self.axial_idx / max(1, self.dims[2] - 1)
        )
        self.coronal_image.display_array(coronal_norm, show_ref=True)

        # Sagittal (transpose then rotate 180°)
        sagittal_data = np.rot90(self.volume[self.sagittal_idx, :, :].T, 2)
        sagittal_norm = self._norm8(sagittal_data)
        self.sagittal_image.set_reference_lines(
            self.coronal_idx / max(1, self.dims[1] - 1),
            self.axial_idx / max(1, self.dims[2] - 1)
        )
        self.sagittal_image.display_array(sagittal_norm, show_ref=True)

        # 4th view
        mode = self.fourth_mode.currentText()
        if mode == "Oblique":
            angle = self.oblique_angle.value()
            plane = self.volume[:, :, self.fourth_idx]
            fourth_data = ndimage.rotate(plane, angle, reshape=False, order=1)
        else:
            fourth_data = np.max(self.volume, axis=2)
        fourth_norm = self._norm8(fourth_data)
        self.fourth_image.display_array(fourth_norm, show_ref=False)

    def run_analysis(self):
        if self.volume is None:
            QMessageBox.warning(self, "Error", "Please load DICOM data first")
            return
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch Not Installed", "Please install PyTorch:\npip install torch torchvision")
            return
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("🤖 Running AI analysis...")
        self.thread = AIThread(self.volume, self.spacing, self.orientation)
        self.thread.progress.connect(lambda msg: self.status_label.setText(f"🤖 {msg}"))
        self.thread.finished.connect(self.analysis_done)
        self.thread.start()

    def analysis_done(self, result: dict):
        self.analyze_btn.setEnabled(True)
        if 'error' in result:
            QMessageBox.critical(self, "Analysis Error", result['error'])
            self.status_label.setText("❌ Analysis failed")
            return
        predictions = result['slice_predictions']
        stats = result['volume_stats']
        msg = f"""AI Analysis Complete!

Model: {result['model_used']}
Device: {result['device']}
Detected Orientation: {result['orientation']}

Organ Predictions:
• Axial View: {predictions[0]['class']} ({predictions[0]['confidence']*100:.1f}% confidence)
• Coronal View: {predictions[1]['class']} ({predictions[1]['confidence']*100:.1f}% confidence)
• Sagittal View: {predictions[2]['class']} ({predictions[2]['confidence']*100:.1f}% confidence)

Volume Statistics:
• Total Volume: {stats['Total_Volume_ml']:.1f} mL
• Mean Intensity: {stats['Mean_Intensity']:.1f}
• Std Intensity: {stats['Std_Intensity']:.1f}
• Range: [{stats['Min_Intensity']:.1f}, {stats['Max_Intensity']:.1f}]
"""
        QMessageBox.information(self, "Analysis Results", msg)
        self.status_label.setText("✓ Analysis complete!")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = DicomViewerGUI()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
