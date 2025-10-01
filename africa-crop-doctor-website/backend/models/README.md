# Africa Crop Doctor AI Model
**Enhanced Multi-Task Crop Disease Classification System**

## Model Information
- **Model Name**: Africa Crop Doctor AI v1.0
- **Architecture**: EfficientNet-B0 with Enhanced Multi-task heads
- **Backbone**: EfficientNet-B0 (pretrained)
- **Enhancement**: Attention mechanism + Feature enhancer
- **Parameters**: 5.30M (5,297,968 total)
- **Model Size**: 20.2 MB
- **Disease Classes**: 35
- **Crop Classes**: 13
- **Severity Levels**: 4 (Healthy, Mild, Moderate, Severe)
- **Input Size**: 224x224 RGB
- **Framework**: PyTorch
- **Created**: 2025-10-01 16:51:27
- **Purpose**: African crop disease diagnosis for mobile deployment

## Architecture Details
```
Input (224x224x3)
    ↓
EfficientNet-B0 Backbone (2.5M params)
    ↓
Feature Enhancer (512 units) + Attention
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Disease Head   │   Crop Head     │  Severity Head  │
│  (Multi-layer)  │  (Multi-layer)  │  (Multi-layer)  │
│  → 35 classes   │  → 13 classes   │  → 4 levels     │
└─────────────────┴─────────────────┴─────────────────┘
```

## Usage
```python
import torch
from pathlib import Path

# Load model
model = CropDiseaseClassifier(num_diseases=35, num_crops=13)
model.load_state_dict(torch.load('pytorch_model.bin', map_location='cpu'))
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image_tensor)
    disease_pred = outputs['disease'].argmax(dim=1)
    crop_pred = outputs['crop'].argmax(dim=1)
    severity_pred = outputs['severity'].argmax(dim=1)
```

## Files
- `pytorch_model.bin`: PyTorch model weights (recommended)
- `model.safetensors`: SafeTensors format (secure)
- `traced_model.pt`: TorchScript traced model (deployment)
- `model_onnx.onnx`: ONNX format (cross-platform)
- `config.json`: Model configuration
- `vocab.txt`: Class mappings
- `tokenizer_config.json`: Preprocessing config
- `model_info.json`: Detailed model metadata

## Performance
- Target: 60-80% disease classification accuracy
- Inference: <300ms on mobile CPU
- Memory: ~40.4MB runtime
