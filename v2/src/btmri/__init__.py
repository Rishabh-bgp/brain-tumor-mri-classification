"""Brain tumour MRI classification (glioma / meningioma / pituitary / no tumour)."""
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary"}
__all__ = ["CLASSES", "CLASS_LABELS"]
