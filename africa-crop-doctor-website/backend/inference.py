#!/usr/bin/env python3
"""
Africa Crop Doctor AI - Model Inference Script
Loads and runs inference with the already trained PyTorch model
"""

import sys
import json
import os
import logging
import torch
from torchvision import transforms
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model classes (from training - these ARE important for mapping predictions to labels)
CROP_CLASSES = [
    'Apple', 'Bell Pepper', 'Cherry', 'Citrus', 'Grape',
    'Maize', 'Cassava', 'Peach', 'Potato', 'Rice',
    'Soybean', 'Strawberry', 'Tomato'
]

DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Bell_pepper___Bacterial_spot', 'Bell_pepper___healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Citrus___Citrus_canker', 'Citrus___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Maize___Common_rust', 'Maize___Northern_Leaf_Blight', 'Maize___healthy',
    'Cassava___Bacterial_blight', 'Cassava___Brown_streak_disease', 'Cassava___Green_mottle', 'Cassava___Mosaic_disease', 'Cassava___healthy',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Rice___Brown_spot', 'Rice___Hispa', 'Rice___Leaf_blast', 'Rice___healthy',
    'Soybean___healthy',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

class ModelInference:
    """Handles loading and running inference with the trained model"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_transforms()
        logger.info(f"Using device: {self.device}")

    def _get_transforms(self):
        """Get image preprocessing transforms (same as training)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self):
        """Load the trained model from saved files"""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Try loading traced model first (optimized for inference)
            traced_model_path = os.path.join(self.model_path, 'traced_model.pt')
            if os.path.exists(traced_model_path):
                logger.info("Loading traced TorchScript model...")
                self.model = torch.jit.load(traced_model_path, map_location=self.device)
                self.model.eval()
                logger.info("✅ Traced model loaded successfully")
                return

            # Fall back to regular PyTorch model
            model_file = os.path.join(self.model_path, 'pytorch_model.bin')
            if os.path.exists(model_file):
                logger.info("Loading PyTorch model weights...")
                # For pytorch_model.bin, we need to know the architecture
                # This should be the same as the trained model
                self.model = self._load_pytorch_model(model_file)
                logger.info("✅ PyTorch model loaded successfully")
                return

            raise FileNotFoundError("No model file found (tried traced_model.pt and pytorch_model.bin)")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _load_pytorch_model(self, model_file):
        """Load PyTorch model with the correct architecture"""
        # This is a simplified loader - assumes the model was saved with state_dict
        # In a real scenario, you'd load the exact same architecture used during training
        try:
            # Try to load as state_dict first
            state_dict = torch.load(model_file, map_location=self.device, weights_only=True)

            # We need to create the model architecture that matches the saved weights
            # This is based on the config.json we saw earlier
            model = self._create_model_from_config()
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise

    def _create_model_from_config(self):
        """Create model architecture based on config"""
        # This recreates the same architecture that was used during training
        import torch.nn as nn
        from torchvision import models

        class TrainedModel(nn.Module):
            def __init__(self):
                super(TrainedModel, self).__init__()
                # EfficientNet-B0 backbone
                self.backbone = models.efficientnet_b0(weights=None)
                backbone_out_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()

                # Feature enhancement
                self.feature_enhancer = nn.Sequential(
                    nn.Linear(backbone_out_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                )

                # Attention
                self.attention = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.Sigmoid()
                )

                # Heads
                self.disease_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 35)  # 35 diseases
                )

                self.crop_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 13)  # 13 crops
                )

                self.severity_head = nn.Sequential(
                    nn.Linear(512, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 4)  # 4 severity levels
                )

            def forward(self, x):
                features = self.backbone(x)
                enhanced = self.feature_enhancer(features)
                attention_weights = self.attention(enhanced)
                attended = enhanced * attention_weights

                return {
                    'disease': self.disease_head(attended),
                    'crop': self.crop_head(attended),
                    'severity': self.severity_head(attended)
                }

        return TrainedModel()

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"❌ Failed to preprocess image: {e}")
            raise

    def predict(self, image_path):
        """Run inference on an image"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            # Preprocess image
            input_tensor = self.preprocess_image(image_path)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)

                # Handle different output formats
                if isinstance(self.model, torch.jit.TracedModule):
                    # For traced model - it returns a tuple instead of dict
                    # Traced model returns tuple in order: (disease, crop, severity)
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        disease_logits, crop_logits, severity_logits = outputs
                    else:
                        raise ValueError(f"Unexpected traced model output format: {type(outputs)}, length: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}")
                elif isinstance(self.model, torch.jit.ScriptModule):
                    # For TorchScript model (compiled, not traced) - also returns tuple
                    # TorchScript model returns tuple in order: (disease, crop, severity)
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        disease_logits, crop_logits, severity_logits = outputs
                    else:
                        raise ValueError(f"Unexpected TorchScript model output format: {type(outputs)}, length: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}")
                else:
                    # For regular model
                    disease_logits = outputs['disease']
                    crop_logits = outputs['crop']
                    severity_logits = outputs['severity']

                # Convert to probabilities
                disease_probs = torch.softmax(disease_logits, dim=1)
                crop_probs = torch.softmax(crop_logits, dim=1)
                severity_probs = torch.softmax(severity_logits, dim=1)

                # Get predictions
                disease_conf, disease_idx = torch.max(disease_probs, dim=1)
                crop_conf, crop_idx = torch.max(crop_probs, dim=1)
                severity_conf, severity_idx = torch.max(severity_probs, dim=1)

                return {
                    'disease_class': DISEASE_CLASSES[disease_idx.item()],
                    'disease_confidence': disease_conf.item(),
                    'crop_class': CROP_CLASSES[crop_idx.item()],
                    'crop_confidence': crop_conf.item(),
                    'severity_level': severity_idx.item(),
                    'severity_confidence': severity_conf.item()
                }

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

def main():
    """Main inference function called from Node.js"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path provided'}))
        sys.exit(1)

    image_path = sys.argv[1]
    # Model directory is passed as second argument, or default to script directory's parent
    model_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Africa Crop Doctor AI v1.0')

    try:
        # Initialize model inference
        inference = ModelInference(model_dir)

        # Load model
        inference.load_model()

        # Run prediction
        result = inference.predict(image_path)

        # Format result for Node.js
        response = {
            'success': True,
            'prediction': {
                'crop': result['crop_class'],
                'disease': result['disease_class'].split('___')[1],
                'fullClass': result['disease_class'],
                'confidence': round(result['disease_confidence'] * 100, 2),
                'crop_confidence': round(result['crop_confidence'] * 100, 2),
                'severity_level': result['severity_level']
            }
        }

        print(json.dumps(response))

    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_response))
        sys.exit(1)

if __name__ == '__main__':
    main()