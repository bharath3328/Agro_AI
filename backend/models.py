from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    phone = Column(String(20))
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    image_filename = Column(String(255))
    disease_name = Column(String(255))
    confidence_score = Column(Float)
    is_unknown = Column(Boolean, default=False)
    disease_stage = Column(String(50))
    recommended_action = Column(String(100))
    estimated_yield_loss = Column(String(50))
    gradcam_path = Column(String(500))
    cam_coverage = Column(Float)
    ai_advisory = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="predictions")


class DiseaseHistory(Base):
    __tablename__ = "disease_history"

    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String(255), index=True, nullable=False)
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float)
    early_stage_count = Column(Integer, default=0)
    mid_stage_count = Column(Integer, default=0)
    late_stage_count = Column(Integer, default=0)
    first_detected = Column(DateTime(timezone=True))
    last_detected = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ReportedCase(Base):
    __tablename__ = "reported_cases"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="PENDING", index=True)
    proposed_label = Column(String(255))
    description = Column(Text)
    admin_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    prediction = relationship("Prediction")
    user = relationship("User")
