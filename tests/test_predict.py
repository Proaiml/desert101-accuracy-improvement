"""Kayıtlı ağırlıklar predict.py modeline yüklenir ve bir görüntü sınıflandırılır (CPU)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict  # noqa: E402


def test_saved_weights_load_and_predict(tmp_path):
    model = predict.load_model()
    img = tmp_path / "x.png"
    Image.fromarray((np.random.rand(90, 70, 3) * 255).astype("uint8")).save(img)
    name, conf = predict.predict(model, img, ["a", "b", "c", "d"])
    assert name in "abcd" and 0.0 <= conf <= 1.0
