from .model import Model, Prediction

try:
    from .dataset import SegDataset, AlignCollate
except Exception:
    SegDataset = None
    AlignCollate = None

