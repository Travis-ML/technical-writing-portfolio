# Neural Network Training Pipeline: A Complete Reference Guide

**Author:** Travis Lelle ([travis@travisml.ai](mailto:travis@travisml.ai))  
**Published:** November 2025

## Overview

This guide breaks down every step of training a neural network, with current industry-standard tools, algorithms, and methods. Each section includes common choices, when to use them, and practical considerations.

---

## 1. Data Loading & Preprocessing

### What Happens Here
Transform raw data into tensors that can be fed into neural networks. Handle batching, shuffling, and efficient data pipeline creation.

### Common Tools & Methods

#### **Data Loading Frameworks**
- **PyTorch `DataLoader`** (most common in research)
  - Handles batching, shuffling, multiprocessing
  - Works with custom `Dataset` classes
  ```python
  from torch.utils.data import DataLoader, Dataset
  
  loader = DataLoader(
      dataset,
      batch_size=32,
      shuffle=True,
      num_workers=4,
      pin_memory=True  # faster GPU transfer
  )
  ```

- **TensorFlow `tf.data`** (common in production)
  - More complex but powerful pipeline
  - Good for large-scale data
  ```python
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
  ```

- **Hugging Face `datasets`** (NLP/multimodal)
  - Built-in caching and memory mapping
  - Streaming for large datasets
  ```python
  from datasets import load_dataset
  
  dataset = load_dataset("imdb", streaming=True)
  ```

- **WebDataset** (large-scale vision)
  - Tar-based format for massive datasets
  - Efficient for cloud storage

#### **Data Preprocessing Libraries**

**Computer Vision:**
- `torchvision.transforms` - standard image transformations
- `albumentations` - fast, extensive augmentation library (used in competitions)
- `opencv-cv2` - low-level image processing
- `PIL/Pillow` - basic image operations

**NLP:**
- `transformers.AutoTokenizer` - pre-built tokenizers for models
- `sentencepiece` - unsupervised tokenization
- `tiktoken` - OpenAI's tokenizer (for GPT models)
- `tokenizers` (Hugging Face) - fast custom tokenizers

**Tabular:**
- `pandas` - data manipulation
- `sklearn.preprocessing` - StandardScaler, MinMaxScaler, LabelEncoder
- `category_encoders` - advanced categorical encoding

**Audio:**
- `torchaudio` - audio loading and processing
- `librosa` - audio analysis
- `soundfile` - audio I/O

#### **Normalization Methods**
- **Image:** Mean/std normalization (ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Tabular:** StandardScaler (z-score), MinMaxScaler (0-1 range), RobustScaler (outlier-resistant)
- **Text:** Already handled by tokenizers (word embeddings, BPE tokens)

#### **Data Validation**
- Check for NaN/inf values
- Verify tensor shapes
- Visualize sample batches
- Check class distribution (for classification)

---

## 2. Data Augmentation

### What Happens Here
Apply transformations to training data to improve model generalization and reduce overfitting.

### Common Methods by Domain

#### **Computer Vision**

**Basic Augmentations:**
- Horizontal/vertical flips
- Random rotations (±15°)
- Random crops and resizing
- Color jittering (brightness, contrast, saturation, hue)
- Gaussian blur/noise

**Advanced Augmentations:**
- **MixUp** - blend two images and their labels
- **CutMix** - cut and paste patches between images
- **RandAugment** - automated augmentation policy
- **AutoAugment** - learned augmentation policies
- **Cutout/Random Erasing** - remove random patches

**Libraries:**
```python
# Albumentations (most powerful)
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
])

# TorchVision (simpler)
from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomResizedCrop(224),
])
```

#### **NLP**

**Text Augmentation:**
- Back-translation
- Synonym replacement (using WordNet)
- Random insertion/deletion/swap
- Paraphrasing with models
- **EDA (Easy Data Augmentation)**

**Libraries:**
- `nlpaug` - comprehensive NLP augmentation
- `textattack` - adversarial augmentation

#### **Audio**

- Time stretching
- Pitch shifting
- Adding background noise
- SpecAugment (for spectrograms)

**Library:**
- `audiomentations` - audio augmentation

#### **Tabular**

- SMOTE (Synthetic Minority Over-sampling)
- Gaussian noise addition
- Feature shuffling
- Mixup for tabular data

---

## 3. Model Architecture Selection

### What Happens Here
Choose or design the neural network architecture for your task.

### Common Architectures by Domain

#### **Computer Vision**

**Classification:**
- **ResNet** (ResNet50, ResNet101) - residual connections, very reliable
- **EfficientNet** (EfficientNet-B0 to B7) - excellent accuracy/efficiency trade-off
- **Vision Transformer (ViT)** - transformer for images, needs lots of data
- **ConvNeXt** - modernized CNN, competitive with transformers
- **DeiT** - data-efficient image transformer
- **Swin Transformer** - hierarchical vision transformer

**Object Detection:**
- **YOLO** (YOLOv8, YOLOv9) - real-time detection
- **Faster R-CNN** - two-stage detector, high accuracy
- **RetinaNet** - single-stage with focal loss
- **DETR** - transformer-based detection

**Segmentation:**
- **U-Net** - medical imaging, semantic segmentation
- **Mask R-CNN** - instance segmentation
- **DeepLab** (v3, v3+) - semantic segmentation
- **Segment Anything Model (SAM)** - foundation model for segmentation

#### **NLP**

**Language Models:**
- **BERT** (bert-base, bert-large) - bidirectional, good for understanding
- **RoBERTa** - improved BERT training
- **DistilBERT** - faster, smaller BERT
- **DeBERTa** - enhanced BERT with disentangled attention
- **GPT-2/GPT-3** - autoregressive, good for generation
- **T5** - text-to-text framework
- **LLaMA** (LLaMA 2, LLaMA 3) - open-source LLMs
- **Mistral/Mixtral** - mixture of experts, efficient

**Specialized:**
- **Sentence-BERT** - sentence embeddings
- **Longformer/BigBird** - long document understanding
- **CodeBERT/CodeT5** - code understanding

#### **Multimodal**

- **CLIP** - image-text understanding
- **BLIP/BLIP-2** - vision-language tasks
- **LLaVA** - visual instruction tuning
- **Flamingo** - few-shot multimodal learning

#### **Audio**

- **Wav2Vec 2.0** - speech representation learning
- **Whisper** - speech recognition
- **HuBERT** - self-supervised audio
- **AudioCraft** - music generation

#### **Tabular**

- **TabNet** - deep learning for tabular data
- **FT-Transformer** - transformer for tabular
- **Simple MLPs** - often still competitive
- **Gradient Boosting** (XGBoost, LightGBM, CatBoost) - often better than neural nets for tabular!

#### **Where to Find Models**

```python
# Hugging Face Hub (30,000+ models)
from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained("microsoft/resnet-50")

# Timm (PyTorch Image Models) - 1000+ vision models
import timm

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
available_models = timm.list_models('*efficient*')

# TorchVision
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
```

---

## 4. Loss Function Selection

### What Happens Here
Define the objective that the model should minimize during training.

### Common Loss Functions

#### **Classification**

- **CrossEntropyLoss** (PyTorch) / **categorical_crossentropy** (TF)
  - Multi-class classification
  - Combines softmax + negative log likelihood
  ```python
  loss_fn = nn.CrossEntropyLoss()
  ```

- **BCEWithLogitsLoss** (binary classification)
  - Binary classification
  - Combines sigmoid + binary cross entropy
  ```python
  loss_fn = nn.BCEWithLogitsLoss()
  ```

- **Focal Loss** (for imbalanced datasets)
  - Down-weights easy examples
  - From RetinaNet paper
  ```python
  # From torchvision or custom implementation
  from torchvision.ops import sigmoid_focal_loss
  ```

- **Label Smoothing**
  - Prevents overconfident predictions
  - Built into CrossEntropyLoss in PyTorch
  ```python
  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
  ```

#### **Regression**

- **MSELoss** (Mean Squared Error)
  - L2 loss, penalizes large errors heavily
  ```python
  loss_fn = nn.MSELoss()
  ```

- **MAELoss** (Mean Absolute Error / L1 Loss)
  - More robust to outliers
  ```python
  loss_fn = nn.L1Loss()
  ```

- **Huber Loss** (Smooth L1)
  - Combination of L1 and L2
  - L2 for small errors, L1 for large
  ```python
  loss_fn = nn.SmoothL1Loss()
  ```

#### **Object Detection**

- **IoU Loss** (Intersection over Union)
- **GIoU, DIoU, CIoU** - improved IoU variants
- **YOLO Loss** - combination of classification, localization, objectness

#### **Segmentation**

- **Dice Loss**
  - Overlap between prediction and ground truth
  - Good for imbalanced segmentation
  
- **Tversky Loss**
  - Generalization of Dice loss
  - Controls false positive/negative balance

- **Combined Losses**
  ```python
  total_loss = dice_loss + crossentropy_loss
  ```

#### **Contrastive Learning**

- **Contrastive Loss** (SimCLR)
- **Triplet Loss** (face recognition)
- **NT-Xent Loss** (normalized temperature-scaled cross entropy)
- **SupCon Loss** (supervised contrastive)

#### **Generative Models**

- **GAN Losses:**
  - Vanilla GAN loss
  - WGAN (Wasserstein)
  - LSGAN (Least Squares)
  
- **VAE Loss:**
  - Reconstruction loss + KL divergence
  
- **Diffusion Loss:**
  - MSE on noise prediction

#### **NLP Specific**

- **Masked Language Modeling (MLM)**
  - Used in BERT training
  
- **Causal Language Modeling (CLM)**
  - Used in GPT training
  
- **Sequence-to-Sequence Loss**
  - CrossEntropy for each token position

---

## 5. Optimizer Selection

### What Happens Here
Choose the algorithm that updates model weights based on gradients.

### Common Optimizers

#### **Most Popular (2024)**

**Adam (Adaptive Moment Estimation)**
- Default choice for most tasks
- Adaptive learning rates per parameter
- Works well out of the box
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

**AdamW (Adam with Weight Decay)**
- Adam with decoupled weight decay
- Current best practice for transformers
- Better generalization than Adam
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

**SGD (Stochastic Gradient Descent) with Momentum**
- Still used for vision models (ResNets)
- Requires careful LR tuning
- Can achieve better final performance than Adam with proper scheduling
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```

#### **Advanced Optimizers**

**AdaFactor**
- Memory-efficient (stores less state)
- Good for large models with limited memory
- Used in T5 training

**LAMB (Layer-wise Adaptive Moments)**
- Large batch training (BERT with batch size 64k)
- Layer-wise adaptive learning rates

**Lion Optimizer**
- Recently published (2023)
- More memory efficient than Adam
- Competitive performance

**8-bit Optimizers**
- From `bitsandbytes` library
- Dramatically reduces optimizer memory
```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
```

#### **Parameter Groups**

Different learning rates for different parts of model:
```python
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},  # frozen/pretrained parts
    {'params': model.head.parameters(), 'lr': 1e-3}       # new parts
], weight_decay=0.01)
```

#### **Gradient Accumulation**

Train with larger effective batch sizes:
```python
accumulation_steps = 4

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # normalize
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. Learning Rate Scheduling

### What Happens Here
Adjust learning rate during training to improve convergence and final performance.

### Common LR Schedules

#### **Warmup**

Start with low LR, gradually increase to target:
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000
)
```

#### **Cosine Annealing**

LR follows cosine curve, gradually decreasing:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # number of epochs
    eta_min=1e-6
)
```

#### **Cosine Annealing with Warm Restarts**

Cosine with periodic restarts:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # epochs until first restart
    T_mult=2,    # factor to increase T_0 after each restart
    eta_min=1e-6
)
```

#### **Step Decay**

Reduce LR by factor at specific epochs:
```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1  # multiply LR by 0.1 at each milestone
)
```

#### **Reduce on Plateau**

Reduce LR when validation metric stops improving:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    verbose=True
)

# In training loop
scheduler.step(val_loss)
```

#### **One Cycle Policy**

LR increases then decreases in one cycle (fast.ai approach):
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=10
)
```

#### **Exponential Decay**

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95  # multiply LR by 0.95 each epoch
)
```

#### **Transformer-style Schedule**

Warmup + inverse square root decay:
```python
# From transformers library
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

#### **Learning Rate Finder**

Find optimal LR before training:
```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # suggests optimal LR
lr_finder.reset()
```

---

## 7. Regularization Techniques

### What Happens Here
Prevent overfitting and improve model generalization.

### Common Regularization Methods

#### **Dropout**

Randomly zero neurons during training:
```python
# In model definition
self.dropout = nn.Dropout(p=0.5)  # 50% dropout rate

# Usage
x = self.dropout(x)
```

**Variants:**
- **DropConnect** - drop weights instead of activations
- **DropBlock** - drop contiguous regions (for CNNs)
- **Stochastic Depth** - drop entire layers (for ResNets)

#### **Weight Decay (L2 Regularization)**

Penalize large weights:
```python
# In optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

#### **Batch Normalization**

Normalize layer inputs, also acts as regularization:
```python
self.bn = nn.BatchNorm2d(num_features=64)
```

**Alternatives:**
- **Layer Normalization** - better for transformers, RNNs
- **Group Normalization** - better for small batches
- **Instance Normalization** - for style transfer

#### **Data Augmentation**

(See Section 2) - most effective regularization!

#### **Early Stopping**

Stop training when validation performance degrades:
```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping")
        break
```

#### **Gradient Clipping**

Prevent exploding gradients:
```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

#### **Label Smoothing**

Prevent overconfident predictions:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

#### **Mixup / CutMix**

Blend training samples (see Data Augmentation)

#### **Noise Injection**

Add noise to weights or gradients:
```python
# Add noise to gradients
for param in model.parameters():
    if param.grad is not None:
        noise = torch.randn_like(param.grad) * 0.01
        param.grad += noise
```

#### **Ensemble Methods**

Train multiple models and average predictions:
- Snapshot ensembles
- Model averaging (SWA - Stochastic Weight Averaging)
```python
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
```

---

## 8. Training Loop

### What Happens Here
The core iteration: forward pass, loss computation, backward pass, weight update.

### Standard Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """One training epoch with mixed precision support"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YourModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = torch.cuda.amp.GradScaler()  # for mixed precision

best_acc = 0.0
for epoch in range(epochs):
    print(f'\nEpoch {epoch+1}/{epochs}')
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    scheduler.step()
    
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pth')
```

### Mixed Precision Training

Faster training with lower memory:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # automatic mixed precision
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Distributed Training

**PyTorch DDP (DistributedDataParallel):**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])
```

**PyTorch Lightning (simpler):**
```python
import pytorch_lightning as pl

trainer = pl.Trainer(
    accelerator='gpu',
    devices=4,
    strategy='ddp'
)
trainer.fit(model, train_dataloader, val_dataloader)
```

**Hugging Face Accelerate:**
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# Training loop looks the same!
outputs = model(inputs)
loss = criterion(outputs, targets)
accelerator.backward(loss)
```

### Gradient Accumulation

Simulate larger batch sizes:
```python
accumulation_steps = 4

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. Validation & Evaluation

### What Happens Here
Assess model performance on held-out data to monitor training and prevent overfitting.

### Evaluation Metrics

#### **Classification**

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Precision, Recall, F1
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# ROC-AUC (for binary/multiclass)
auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Full report
report = classification_report(y_true, y_pred)
```

**Top-k Accuracy:**
```python
def topk_accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
```

#### **Regression**

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# MSE / RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R² score
r2 = r2_score(y_true, y_pred)

# MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
```

#### **Object Detection**

```python
# mAP (mean Average Precision)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

map_metric = MeanAveragePrecision()
map_metric.update(preds, targets)
map_value = map_metric.compute()
```

#### **Segmentation**

```python
# IoU (Intersection over Union) / Dice Score
from torchmetrics import JaccardIndex, Dice

iou = JaccardIndex(task='multiclass', num_classes=num_classes)
dice = Dice(num_classes=num_classes)
```

#### **NLP**

```python
# BLEU (machine translation)
from nltk.translate.bleu_score import sentence_bleu

# ROUGE (summarization)
from rouge_score import rouge_scorer

# Perplexity (language modeling)
perplexity = torch.exp(loss)

# Hugging Face evaluate library
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")
```

### Cross-Validation

```python
from sklearn.model_selection import KFold, StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f'Fold {fold + 1}')
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    # Train model on this fold
    # ...
```

### Test Time Augmentation (TTA)

Average predictions over augmented versions:
```python
def predict_with_tta(model, image, transforms, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for transform in transforms:
            augmented = transform(image).unsqueeze(0).to(device)
            pred = model(augmented)
            predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

---

## 10. Experiment Tracking

### What Happens Here
Log metrics, hyperparameters, and artifacts to track experiments and enable reproducibility.

### Popular Tools

#### **Weights & Biases (W&B)**

Most popular in ML research:
```python
import wandb

# Initialize
wandb.init(
    project="my-project",
    config={
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "architecture": "ResNet50"
    }
)

# Log during training
for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss, val_acc = validate()
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# Log images, plots, models
wandb.log({"predictions": wandb.Image(img)})
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(...)})
wandb.save("model.pth")
```

#### **TensorBoard**

Built into PyTorch:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# Log scalars
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)

# Log images
writer.add_image('predictions', img_grid, epoch)

# Log model graph
writer.add_graph(model, sample_input)

# Log hyperparameters
writer.add_hparams(
    {'lr': 1e-3, 'batch_size': 32},
    {'accuracy': val_acc, 'loss': val_loss}
)

writer.close()

# View: tensorboard --logdir=runs
```

#### **MLflow**

Good for production ML:
```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", 32)
    
    # Train model
    for epoch in range(epochs):
        train_loss = train_epoch()
        mlflow.log_metric("train_loss", train_loss, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

#### **Neptune.ai**

Similar to W&B, metadata-focused:
```python
import neptune.new as neptune

run = neptune.init_run(project="workspace/project")

run["config"] = {"learning_rate": 1e-3, "batch_size": 32}
run["train/loss"].log(train_loss)
run["val/accuracy"].log(val_acc)
```

#### **Hugging Face Trainer Integration**

Automatic logging with Trainer:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # or "tensorboard", "mlflow"
    logging_steps=100,
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

---

## 11. Model Checkpointing

### What Happens Here
Save model states during training to enable resumption and preserve best models.

### Checkpointing Strategies

#### **Basic Checkpointing**

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc,
}, f'checkpoint_epoch_{epoch}.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint_epoch_10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

#### **Save Best Model Only**

```python
best_val_loss = float('inf')

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    # Save if best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pth')
        print(f'Saved best model at epoch {epoch}')
```

#### **Keep Top-K Checkpoints**

```python
import heapq
from pathlib import Path

class CheckpointManager:
    def __init__(self, save_dir, keep_top_k=3, metric='val_acc', mode='max'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.checkpoints = []  # heap of (metric, epoch, path)
    
    def save(self, epoch, model, optimizer, metrics):
        metric_value = metrics[self.metric]
        if self.mode == 'min':
            metric_value = -metric_value
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **metrics
        }, checkpoint_path)
        
        heapq.heappush(self.checkpoints, (metric_value, epoch, checkpoint_path))
        
        # Remove worst checkpoint if exceeded keep_top_k
        if len(self.checkpoints) > self.keep_top_k:
            _, _, path_to_remove = heapq.heappop(self.checkpoints)
            path_to_remove.unlink()
```

#### **Periodic Checkpointing**

```python
# Save every N epochs
if (epoch + 1) % save_every == 0:
    torch.save(state, f'checkpoint_epoch_{epoch}.pth')
```

#### **PyTorch Lightning Callbacks**

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    save_last=True
)

trainer = pl.Trainer(callbacks=[checkpoint_callback])
```

#### **Hugging Face Trainer**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # keep only 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
```

#### **Model Versioning**

```python
# Include git commit hash in checkpoint
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return None

torch.save({
    'model_state_dict': model.state_dict(),
    'git_commit': get_git_commit(),
    'timestamp': datetime.now().isoformat(),
}, 'model.pth')
```

---

## 12. Hyperparameter Tuning

### What Happens Here
Systematically search for optimal hyperparameters to improve model performance.

### Search Strategies

#### **Grid Search**

Exhaustive search over parameter grid:
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'dropout': [0.1, 0.3, 0.5]
}

for params in ParameterGrid(param_grid):
    print(f"Training with {params}")
    model = create_model(params['dropout'])
    train(model, lr=params['learning_rate'], bs=params['batch_size'])
```

#### **Random Search**

Sample random combinations:
```python
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, loguniform

param_distributions = {
    'learning_rate': loguniform(1e-5, 1e-2),
    'batch_size': [16, 32, 64],
    'dropout': uniform(0.1, 0.5)
}

for params in ParameterSampler(param_distributions, n_iter=20):
    train(model, **params)
```

#### **Bayesian Optimization**

**Optuna (most popular):**
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Train model
    model = create_model(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    val_acc = train_and_evaluate(model, optimizer, batch_size)
    
    return val_acc

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

**Ray Tune:**
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = create_model(config['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(10):
        train_loss = train_epoch(model, optimizer)
        val_acc = validate(model)
        tune.report(val_acc=val_acc)

config = {
    'lr': tune.loguniform(1e-5, 1e-2),
    'dropout': tune.uniform(0.1, 0.5),
    'batch_size': tune.choice([16, 32, 64])
}

scheduler = ASHAScheduler(max_t=10, grace_period=1)

analysis = tune.run(
    train_model,
    config=config,
    num_samples=50,
    scheduler=scheduler
)

print("Best config:", analysis.best_config)
```

#### **Hyperband / ASHA**

Early stopping for poorly performing trials:
```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    max_t=100,          # max epochs
    grace_period=10,    # min epochs before stopping
    reduction_factor=3  # fraction of trials to keep
)
```

#### **Population Based Training (PBT)**

Evolve hyperparameters during training:
```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    perturbation_interval=5,
    hyperparam_mutations={
        'lr': lambda: tune.loguniform(1e-5, 1e-2),
        'dropout': lambda: tune.uniform(0.1, 0.5)
    }
)
```

#### **Weights & Biases Sweeps**

```python
import wandb

sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': -9.21,  # ln(1e-4)
            'max': -4.61   # ln(1e-2)
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")

def train():
    wandb.init()
    config = wandb.config
    
    # Train with config
    model = create_model(dropout=config.dropout)
    # ...
    
    wandb.log({'val_acc': val_acc})

wandb.agent(sweep_id, train, count=50)
```

---

## 13. Model Inference & Deployment

### What Happens Here
Use trained model for predictions in production or evaluation environments.

### Inference Optimization

#### **Model Export Formats**

**TorchScript (PyTorch):**
```python
# Trace model
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Load and use
loaded = torch.jit.load("model_traced.pt")
output = loaded(input_tensor)
```

**ONNX (Cross-framework):**
```python
# Export to ONNX
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Run with ONNX Runtime (faster inference)
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'input': input_array})
```

**TensorRT (NVIDIA GPUs):**
```python
# Convert ONNX to TensorRT for maximum speed
import tensorrt as trt

# Build engine (one-time conversion)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("model.onnx")

# Optimize and save
engine = builder.build_cuda_engine(network)
```

#### **Quantization**

Reduce model size and increase speed:

**Post-Training Quantization (PyTorch):**
```python
# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (more accurate)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Calibrate with representative data
for data in calibration_data:
    model(data)
torch.quantization.convert(model, inplace=True)
```

**ONNX Quantization:**
```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic("model.onnx", "model_quantized.onnx")
```

#### **Batch Inference**

Process multiple samples efficiently:
```python
def batch_predict(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())
    
    return torch.cat(predictions, dim=0)
```

### Deployment Platforms

#### **Hugging Face Inference API**

```python
from transformers import pipeline

# Easy deployment with pipelines
classifier = pipeline("text-classification", model="your-model")
result = classifier("This is great!")
```

#### **FastAPI**

```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("model.pth")
model.eval()

@app.post("/predict")
async def predict(data: dict):
    input_tensor = preprocess(data['image'])
    with torch.no_grad():
        output = model(input_tensor)
    return {"prediction": output.tolist()}
```

#### **TorchServe**

Official PyTorch serving:
```bash
# Archive model
torch-model-archiver --model-name my_model \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pth \
    --handler image_classifier

# Start server
torchserve --start --model-store model_store --models my_model=my_model.mar

# Inference
curl -X POST http://localhost:8080/predictions/my_model -T image.jpg
```

#### **AWS SageMaker**

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1
)

# Make predictions
result = predictor.predict(data)
```

#### **Docker Deployment**

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pth /app/
COPY app.py /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Edge Deployment

**TensorFlow Lite:**
```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**CoreML (iOS):**
```python
import coremltools as ct

model = ct.convert(
    torch_model,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))]
)
model.save("model.mlmodel")
```

---

## Bonus: Modern Training Frameworks

### PyTorch Lightning

Reduces boilerplate:
```python
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Train
trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, train_dataloader, val_dataloader)
```

### Hugging Face Trainer

For transformers:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

### Hugging Face Accelerate

Low-level distributed training:
```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    outputs = model(batch)
    loss = criterion(outputs, labels)
    accelerator.backward(loss)
    optimizer.step()
```

---

## Summary: The Complete Pipeline

```
1. Data Loading → PyTorch DataLoader / HF datasets
2. Augmentation → albumentations / torchvision
3. Model → timm / Hugging Face / torchvision
4. Loss → CrossEntropyLoss / Focal / Custom
5. Optimizer → AdamW / SGD + momentum
6. LR Schedule → Cosine / Warmup + decay
7. Regularization → Dropout / Weight decay / Augmentation
8. Training → Mixed precision / Gradient accumulation
9. Validation → Accuracy / F1 / mAP / Perplexity
10. Tracking → W&B / TensorBoard
11. Checkpointing → Save best / Top-K
12. Tuning → Optuna / Ray Tune / W&B Sweeps
13. Deployment → ONNX / TorchScript / API
```

This guide covers 95% of what you'll encounter in modern ML/DL training. Bookmark it, reference it during projects, and gradually build your intuition for which tools fit which situations. Good luck!