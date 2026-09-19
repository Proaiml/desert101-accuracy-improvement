import torch
import torch.nn as nn

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================================================
# MODEL
# =========================================================

class DesertClassifier(nn.Module):

    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(
                in_features=hidden_units * 16 * 16,
                out_features=output_shape
            )
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x


# =========================================================
# TRAIN
# =========================================================

def train_step(model, dataloader, loss_fn, optimizer, device):

    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward
        pred = model(images)

        # Loss
        loss = loss_fn(pred, labels)

        # Gradientleri temizle
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Ağırlıkları güncelle
        optimizer.step()

        # Loss hesapla
        train_loss += loss.item() * images.size(0)

        # Accuracy hesapla
        predicted_classes = pred.argmax(dim=1)

        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

    train_loss = train_loss / total
    train_accuracy = correct / total

    return train_loss, train_accuracy


# =========================================================
# EVALUATION
# =========================================================

def evaluate_step(model, dataloader, loss_fn, device):

    model.eval()

    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in dataloader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            pred = model(images)

            loss = loss_fn(pred, labels)

            eval_loss += loss.item() * images.size(0)

            predicted_classes = pred.argmax(dim=1)

            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    eval_loss = eval_loss / total
    eval_accuracy = correct / total

    return eval_loss, eval_accuracy


# =========================================================
# MAIN
# =========================================================

def main():

    # Tekrarlanabilirlik
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # GPU varsa kullan
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Kullanılan cihaz:", device)

    # =====================================================
    # DATASET PATH
    # =====================================================

    image_path = Path(
        r"C:\Users\İlhan\Desktop\ödev4\desert101"
    )

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    print("Train yolu:", train_dir)
    print("Test yolu:", test_dir)

    print("Train var mı:", train_dir.exists())
    print("Test var mı:", test_dir.exists())

    if not train_dir.exists() or not test_dir.exists():
        print("HATA: Dataset yolu bulunamadı.")
        return

    # =====================================================
    # TRAIN TRANSFORM
    # Augmentation sadece train tarafında
    # =====================================================

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),

        transforms.RandomHorizontalFlip(
            p=0.4
        ),



        transforms.ToTensor()
    ])

    # =====================================================
    # TEST TRANSFORM
    # Testte augmentation yok
    # =====================================================

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # =====================================================
    # DATASET
    # =====================================================

    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    class_names = train_data.classes

    print("Sınıflar:", class_names)
    print("Train görüntü sayısı:", len(train_data))
    print("Test görüntü sayısı:", len(test_data))

    # =====================================================
    # DATALOADER
    # =====================================================

    BATCH_SIZE = 32
    NUM_WORKERS = 4

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = DesertClassifier(
        input_shape=3,
        hidden_units=64,
        output_shape=len(class_names)
    )

    model = model.to(device)

    # =====================================================
    # LOSS + OPTIMIZER
    # =====================================================

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.3
    )


    NUM_EPOCHS = 24

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = evaluate_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )
        scheduler.step()

    # =====================================================
    # MODEL SAVE
    # =====================================================

    save_path = Path(
        r"C:\Users\İlhan\Desktop\ödev4\desert_model.pth"
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print()
    print("Model kaydedildi:", save_path)



if __name__ == "__main__":
    main()