import torch
import os
from typing import Dict, Tuple
from ml.encoder import Encoder
from ml.classifier import PrototypeClassifier
from ml.prototypes import compute_prototypes
from ml.threshold import compute_open_set_threshold
from backend.config import settings

class MLService:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.device = settings.DEVICE
            if self.device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, using CPU")
                self.device = "cpu"
            
            self.classifier: PrototypeClassifier = None
            self.prototypes: Dict = None
            self.class_names: Tuple = None
            self.threshold: float = None
            self._initialized = True
    
    def initialize(self):
        if self.classifier is not None:
            return
        
        encoder_path = settings.ENCODER_PATH
        train_dir = settings.TRAIN_DATA_DIR
        
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder model not found at {encoder_path}")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training data directory not found at {train_dir}")
        
        self.threshold = compute_open_set_threshold(
            encoder_path, train_dir, self.device, percentile=0.5
        )
        print(f"Open-set threshold: {self.threshold:.3f}")
        
        self.prototypes, self.class_names = compute_prototypes(
            encoder_path, train_dir, self.device
        )
        print(f"Loaded {len(self.class_names)} disease classes")
        
        self.classifier = PrototypeClassifier(
            encoder_path, self.prototypes, self.class_names, self.device
        )
        
    def get_classifier(self) -> PrototypeClassifier:
        if self.classifier is None:
            self.initialize()
        return self.classifier
    
    def get_threshold(self) -> float:
        if self.threshold is None:
            self.initialize()
        return self.threshold

    def retrain_model(self):
        self.classifier = None
        encoder_path = settings.ENCODER_PATH
        train_dir = settings.TRAIN_DATA_DIR
        self.prototypes, self.class_names = compute_prototypes(
            encoder_path, train_dir, self.device
        )
        print(f"Reloaded {len(self.class_names)} disease classes")
        
        self.classifier = PrototypeClassifier(
            encoder_path, self.prototypes, self.class_names, self.device
        )

ml_service = MLService()
