"""Configurable patch encoder (WP011). Hackathon default: bioptimus/H-optimus-0 (Apache-2.0).
UNI (MahmoodLab, CC-BY-NC-ND-4.0, gated) is kept as the historical research configuration only.

Both encoders take a 224 x 224 RGB patch (112 um at 0.5 um/px) and return one L2-normalised vector; the
acquisition code never looks at the dimension, so 1024 (UNI) and 1536 (H-optimus-0) are interchangeable
downstream as long as a classifier is trained on the same encoder's embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENCODERS = {
    "hoptimus0": dict(hub="hf-hub:bioptimus/H-optimus-0", dim=1536, mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517),
                      autocast_fp16=True, dynamic_img_size=False, license="Apache-2.0", gated=True,
                      note="official preprocessing: 224 px, vendor mean/std, fp16 autocast recommended on GPU (fp32 on CPU)"),
    "uni": dict(hub="hf-hub:MahmoodLab/UNI", dim=1024, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                autocast_fp16=False, dynamic_img_size=True, license="CC-BY-NC-ND-4.0 (non-commercial research only)", gated=True,
                note="historical research configuration; not used for hackathon demos"),
}


@dataclass
class EncoderSpec:
    name: str
    hub: str
    dim: int
    mean: tuple
    std: tuple
    autocast_fp16: bool
    license: str


def spec(name: str) -> EncoderSpec:
    e = ENCODERS[name]
    return EncoderSpec(name, e["hub"], e["dim"], e["mean"], e["std"], e["autocast_fp16"], e["license"])


def build_encoder(name: str, device: str = "cuda"):
    """Return (model, transform, spec). transform: PIL/ndarray 224 RGB -> normalised tensor [3,224,224]."""
    import timm, torch
    from torchvision import transforms
    e = ENCODERS[name]
    model = timm.create_model(e["hub"], pretrained=True, init_values=1e-5, dynamic_img_size=e["dynamic_img_size"]).to(device).eval()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=e["mean"], std=e["std"])])   # input already 224 px
    return model, tf, spec(name)


def encode(model, sp: EncoderSpec, x, device: str, l2_normalize: bool = True) -> np.ndarray:
    """x: tensor [B,3,224,224] -> float32 [B, dim] (L2-normalised by default)."""
    import torch
    with torch.inference_mode():
        x = x.to(device)
        if sp.autocast_fp16 and str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
        else:
            out = model(x)
        out = out.float()
        if l2_normalize:
            out = torch.nn.functional.normalize(out, dim=1)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.float32)
