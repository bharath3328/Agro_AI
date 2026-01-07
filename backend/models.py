from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base


class User(Base):
    """User model for farmers and admins"""
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
    verification_codes = relationship("VerificationCode", back_populates="user", cascade="all, delete-orphan")


class VerificationCode(Base):
    """Store verification codes for email/phone verification and password reset"""
    __tablename__ = "verification_codes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    code = Column(String(10), nullable=False)
    type = Column(String(20), nullable=False)  # "EMAIL_VERIFICATION", "PHONE_VERIFICATION", "PASSWORD_RESET"
    via = Column(String(10), nullable=False)  # "EMAIL", "SMS"
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_used = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="verification_codes")


class Prediction(Base):
    """Disease prediction records"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Image info
    image_path = Column(String(500), nullable=False)
    image_filename = Column(String(255))
    
    # Prediction results
    disease_name = Column(String(255))
    confidence_score = Column(Float)
    is_unknown = Column(Boolean, default=False)
    
    # Disease intelligence
    disease_stage = Column(String(50))  # Early, Mid, Late
    recommended_action = Column(String(100))  # Monitor, Preventive, Curative, Immediate Treatment
    estimated_yield_loss = Column(String(50))  # e.g., "5-10%"
    
    # Grad-CAM
    gradcam_path = Column(String(500))
    cam_coverage = Column(Float)  # Percentage of image covered by heatmap
    
    # AI Advisory
    ai_advisory = Column(Text)  # Full AI-generated advisory text
    
    # Metadata
    crop_type = Column(String(100))
    location = Column(String(255))
    notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="predictions")


class DiseaseHistory(Base):
    """Aggregated disease statistics for analytics"""
    __tablename__ = "disease_history"

    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String(255), index=True, nullable=False)
    crop_type = Column(String(100), index=True)
    
    # Statistics
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float)
    early_stage_count = Column(Integer, default=0)
    mid_stage_count = Column(Integer, default=0)
    late_stage_count = Column(Integer, default=0)
    
    # Timestamps
    first_detected = Column(DateTime(timezone=True))
    last_detected = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ReportedCase(Base):
    """Cases reported by users for unknown diseases"""
    __tablename__ = "reported_cases"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Status: PENDING, APPROVED, REJECTED
    status = Column(String(20), default="PENDING", index=True)
    
    # User input
    proposed_label = Column(String(255))
    description = Column(Text)
    
    # Admin feedback
    admin_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    prediction = relationship("Prediction")
    user = relationship("User")
