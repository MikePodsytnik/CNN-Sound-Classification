# UrbanSound8K Audio Classification

Классификация городских звуков из датасета **UrbanSound8K** с использованием log-mel спектрограмм и сверточной нейросети (CNN).

Проект реализует полный ML-пайплайн:  
загрузка аудио → аугментации → извлечение признаков → обучение модели → валидация → early stopping → сохранение чекпоинтов → финальная оценка на тесте.

---

## Датасет

**UrbanSound8K**
- 8732 аудиоклипа
- 10 классов (air_conditioner, car_horn, dog_bark и др.)
- 10 фолдов

---

## Архитектура

### Признаки
- Resampling до 22050 Hz
- Фиксированная длина 4 секунды
- Log-Mel Spectrogram (256 mel-бинов)
- Per-sample z-normalization

### Модель
`UrbanSoundCNN`
- 4 сверточных блока (Conv → BN → ReLU ×2 → MaxPool → Dropout)
- Global Average Pooling
- Linear head (64 hidden units)
- Dropout = 0.6
- ~591k параметров

---

## Регуляризация и обучение

- Optimizer: AdamW
- weight_decay: 3e-4
- label_smoothing: 0.15
- Gradient clipping (max_norm=1.0)
- LR scheduler: ReduceLROnPlateau
- Early stopping (patience=10)

---

## Аугментации (train only)

### Audio-level
- Random crop (позиционная инвариантность)
- Time shift с zero-padding
- Random gain
- Gaussian noise

### SpecAugment
- Time masking
- Frequency masking
