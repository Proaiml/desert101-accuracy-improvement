# Desert101 — görüntü sınıflandırma ve doğruluk iyileştirme

Çöl görüntülerini 4 sınıfa ayıran küçük bir evrişimli ağ (PyTorch). 64×64 girdi, iki evrişim bloğu (her blokta iki 3×3 evrişim ve MaxPool), Dropout 0.4 ile doğrusal sınıflandırıcı. Eğitimde yalnızca yatay çevirme ile veri çoğaltılır. Optimizasyon Adam (lr 0.001) ve StepLR (10 epokta bir ×0.3) ile yapılır, eğitim 24 epok sürer.

## Kurulum

```bash
pip install -r requirements.txt
```

## Eğitim

Veri klasörü `ImageFolder` düzenindedir:

```
desert101/
├── train/<sınıf_adı>/*.jpg
└── test/<sınıf_adı>/*.jpg
```

`main.py` içindeki `image_path` ve `save_path` satırlarını kendi klasörünüze göre ayarlayıp çalıştırın:

```bash
python main.py
```

Her epokta eğitim/test kaybı ve doğruluğu yazılır; sonunda ağırlıklar `desert_model.pth` olarak kaydedilir.

## Hazır modelle tahmin

Depodaki `desert_model.pth`, evrişimlerden sonra BatchNorm kullanan sürümle eğitildi. `predict.py` bu ağırlıklarla birebir eşleşen modeli kurar (GPU gerekmez):

```bash
python predict.py resim.jpg --classes <sınıf1> <sınıf2> <sınıf3> <sınıf4>
python predict.py klasor/
```

Sınıf adları, eğitimdeki `train/` klasör adlarının alfabetik sırasıdır.
