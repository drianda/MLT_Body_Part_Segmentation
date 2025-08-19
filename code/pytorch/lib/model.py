import os
import torch
from PIL import Image
import numpy as np

from .arch import Architecture


class Model(object):
    """
    Minimal model wrapper for inference on modern PyTorch (CPU by default).
    """

    def __init__(self, labels, input_shape, load_model_path='', usegpu=False):
        # labels: list of class names; n_classes = len(labels)
        # input_shape: (N, C, H, W), e.g. (1, 3, 256, 256)
        assert len(input_shape) == 4 and input_shape[1] == 3, "input_shape must be (N,3,H,W)"
        self.labels = labels
        self.n_classes = len(labels)
        self.input_shape = input_shape
        self.load_model_path = load_model_path or ''
        self.usegpu = bool(usegpu) and torch.cuda.is_available()

        # Build network
        self.model = Architecture(self.n_classes, self.input_shape, usegpu=self.usegpu)
        if self.usegpu:
            self.model = self.model.cuda()
        self.model.eval()

        # Load weights (CPU-safe, non-strict)
        self.__load_weights()

    def __load_weights(self):
        if not self.load_model_path:
            return

        assert os.path.isfile(self.load_model_path), \
            'Model : {} does not exists!'.format(self.load_model_path)

        # Load checkpoint onto CPU so it works without CUDA
        state = torch.load(self.load_model_path, map_location="cpu")

        # Non-strict load: ignore keys that aren't present (e.g., p_logit params)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        try:
            print("Loaded with strict=False. Missing:", missing, "Unexpected:", unexpected)
        except Exception:
            pass


class Prediction(object):
    """
    Minimal prediction wrapper compatible with pred_folder.py:
    predict(image_path, n_samples=...) -> (PIL_image, pred_array_uint8, vis_var, variance)
    """
    def __init__(self, img_h, img_w, mean, std, model: Model):
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

        self.net = model.model.eval()
        self.usegpu = model.usegpu

    def _preprocess(self, pil_img: Image.Image):
        # resize to network input, normalize, CHW
        img = pil_img.convert("RGB").resize((self.img_w, self.img_h), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        t = torch.from_numpy(arr).unsqueeze(0)  # 1x3xH xW
        if self.usegpu:
            t = t.cuda(non_blocking=True)
        return t

    @torch.no_grad()
    def predict(self, image_path, n_samples=1):
        # Note: n_samples ignored here (this is deterministic inference)
        pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = pil.size

        inp = self._preprocess(pil)
        logits = self.net(inp)                      # 1 x C x H x W
        pred_small = torch.argmax(logits, dim=1)    # 1 x H x W
        pred_small = pred_small.squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize predictions back to the original image size
        pred_img = Image.fromarray(pred_small, mode="L").resize((orig_w, orig_h), Image.NEAREST)
        pred_np = np.array(pred_img, dtype=np.uint8)

        # For API compatibility with the repo's pred_folder.py
        vis_var = None
        variance = None
        return pil, pred_np, vis_var, variance
