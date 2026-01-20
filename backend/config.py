import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:    
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "AgroAI - Plant Disease Detection System"
    VERSION: str = "1.0.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    ENCODER_PATH: str = os.getenv("ENCODER_PATH", "backend/ml/encoder_supcon.pth")
    TRAIN_DATA_DIR: str = os.getenv("TRAIN_DATA_DIR", "data/fewshot/train")
    OPEN_SET_THRESHOLD: Optional[float] = None
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/processed")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    GRADCAM_OUTPUT_DIR: str = os.getenv("GRADCAM_OUTPUT_DIR", "data/processed/gradcam")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000",
    ]
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_SENDER: Optional[str] = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD: Optional[str] = os.getenv("EMAIL_PASSWORD")

settings = Settings()
