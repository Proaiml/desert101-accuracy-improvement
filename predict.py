"""
Kayıtlı desert_model.pth ile tek görüntü ya da klasör sınıflandırma.

    py -3.11 predict.py resim.jpg
    py -3.11 predict.py klasor/ --classes sinif1 sinif2 sinif3 sinif4

desert_model.pth, BatchNorm'lu bir sürümle eğitildi (Conv-BN-ReLU-Conv-BN-ReLU-MaxPool x2).
Bu dosya o ağırlıklarla birebir eşleşen modeli kurar. Sınıf adları eğitimde ImageFolder'ın
alfabetik klasör sırasıdır; --classes verilmezse "sinif_0..3" yazılır.
GPU gerekmez; varsayılan olarak CPU'da çalışır.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class DesertClassifierBN(nn.Module):
    def __init__(self, hidden_units=64, output_shape=4):
        super().__init__()
        self.conv_block_1 = conv_block(3, hidden_units)
        self.conv_block_2 = conv_block(hidden_units, hidden_units)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
                                        nn.Linear(hidden_units * 16 * 16, output_shape))

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def load_model(path=ROOT / "desert_model.pth", device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model = DesertClassifierBN(output_shape=state["classifier.2.weight"].shape[0])
    model.load_state_dict(state)
    return model.to(device).eval()


TRANSFORM = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])   # main.py'deki test dönüşümü


@torch.no_grad()
def predict(model, image_path, classes):
    x = TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0)
    probs = model(x).softmax(dim=1)[0]
    k = int(probs.argmax())
    return classes[k], float(probs[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="görüntü dosyası ya da klasör")
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--weights", default=str(ROOT / "desert_model.pth"))
    args = ap.parse_args()
    model = load_model(args.weights)
    n = model.classifier[2].out_features
    classes = args.classes or [f"sinif_{i}" for i in range(n)]
    p = Path(args.path)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")) if p.is_dir() else [p]
    for f in files:
        name, conf = predict(model, f, classes)
        print(f"{f.name}: {name} (%{100 * conf:.1f})")


if __name__ == "__main__":
    main()
