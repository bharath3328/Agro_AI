"""
Diagnosis Router - Main endpoint for disease detection
Integrates all ML components: classification, Grad-CAM, AI reasoning, intelligence
"""
import os
import uuid
import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image
from pydantic import BaseModel
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

from backend.database import get_db
from backend.models import Prediction, DiseaseHistory
from backend.auth_utils import get_current_active_user
from backend.ml_service import ml_service
from backend.config import settings
from ml.gradcam import GradCAM
from ml.encoder import Encoder
from ml.transforms import inference_transform
from ml.agro_intelligence import assess_disease_intelligence
from ml.api_reasoner import generate_ai_advisory

router = APIRouter(prefix="/diagnosis", tags=["Disease Diagnosis"])


class DiagnosisRequest(BaseModel):
    notes: Optional[str] = None


class DiagnosisResponse(BaseModel):
    prediction_id: int
    disease_name: str
    confidence_score: float
    is_unknown: bool
    disease_stage: str
    recommended_action: str
    estimated_yield_loss: str
    gradcam_path: str
    cam_coverage: float
    ai_advisory: str
    image_filename: str


def save_uploaded_file(file: UploadFile, user_id: int) -> tuple[str, str]:
    """Save uploaded file and return (file_path, filename)"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext.lower() not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid file extension: {file_ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    filename = f"{user_id}_{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    try:
        # Save file directly
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            if len(content) > settings.MAX_UPLOAD_SIZE:
                logger.warning("File too large upload attempt")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
                )
            buffer.write(content)
            
        # Verify it's a real image using PIL
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file. The file is corrupted or not a valid image."
            )
            
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e
    
    logger.info(f"File saved successfully: {filename}")
    return file_path, filename


def generate_gradcam(image_path: str, prototype: torch.Tensor, user_id: int) -> tuple[str, float]:
    """Generate Grad-CAM visualization and return (gradcam_path, cam_coverage)"""
    os.makedirs(settings.GRADCAM_OUTPUT_DIR, exist_ok=True)
    
    device = ml_service.device
    
    # Load model
    model = Encoder()
    model.load_state_dict(torch.load(settings.ENCODER_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Get target layer
    target_layer = model.feature_extractor[-1]
    cam_generator = GradCAM(model, target_layer)
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    # Generate CAM using prototype similarity
    cam = cam_generator.generate_from_prototype(input_tensor, prototype)
    
    # Filter background noise (thresholding)
    cam[cam < 0.4] = 0
    
    # Calculate coverage (percentage of image with high activation)
    cam_coverage = float((cam > 0.0).sum() / cam.size)
    
    # Create visualization
    img_array = np.array(image.resize((224, 224)))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    
    # Save Grad-CAM
    gradcam_filename = f"{user_id}_{uuid.uuid4()}_gradcam.jpg"
    gradcam_path = os.path.join(settings.GRADCAM_OUTPUT_DIR, gradcam_filename)
    cv2.imwrite(gradcam_path, overlay)
    
    return gradcam_path, cam_coverage


@router.post("/predict", response_model=DiagnosisResponse, status_code=status.HTTP_201_CREATED)
async def predict_disease(
    file: UploadFile = File(...),
    notes: Optional[str] = None,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Main disease prediction endpoint
    Integrates: classification, open-set detection, Grad-CAM, AI reasoning, intelligence
    """
    
    # Initialize ML service if needed
    classifier = ml_service.get_classifier()
    threshold = ml_service.get_threshold()
    
    # Save uploaded file
    image_path, filename = save_uploaded_file(file, current_user.id)
    
    try:
        # 1. Disease Classification
        disease_name, confidence_score = classifier.predict(image_path, threshold)
        print(f"DEBUG: Prediction: {disease_name}, Confidence: {confidence_score:.4f}, Threshold: {threshold:.4f}")
        is_unknown = (disease_name == "UNKNOWN")
        
        # 2. Generate Grad-CAM (only if not unknown)
        gradcam_path = None
        cam_coverage = 0.0
        
        if not is_unknown:
            # Find disease label index
            disease_label = None
            for idx, name in enumerate(classifier.class_names):
                if name == disease_name:
                    disease_label = idx
                    break
            
            if disease_label is not None:
                # Retrieve the specific prototype for this class
                prototype = ml_service.prototypes[disease_label]
                
                gradcam_path, cam_coverage = generate_gradcam(
                    image_path, prototype, current_user.id
                )
        
        # 3. Disease Intelligence Assessment
        if is_unknown:
            disease_stage = "Unknown"
            recommended_action = "Monitor Only"
            estimated_yield_loss = "Unknown"
        else:
            disease_stage, recommended_action, estimated_yield_loss = assess_disease_intelligence(
                confidence_score, cam_coverage
            )
        
        # 4. AI Advisory (only if not unknown)
        ai_advisory = ""
        if not is_unknown:
            try:
                ai_advisory = generate_ai_advisory(
                    disease=disease_name,
                    crop="Unknown",
                    confidence=confidence_score,
                    severity=disease_stage
                )
            except Exception as e:
                ai_advisory = f"AI advisory generation failed: {str(e)}"
        
        # 5. Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            image_path=image_path,
            image_filename=filename,
            disease_name=disease_name if not is_unknown else None,
            confidence_score=confidence_score,
            is_unknown=is_unknown,
            disease_stage=disease_stage,
            recommended_action=recommended_action,
            estimated_yield_loss=estimated_yield_loss,
            gradcam_path=gradcam_path,
            cam_coverage=cam_coverage,
            ai_advisory=ai_advisory,
            notes=notes
        )
        
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        # 6. Update disease history (if not unknown)
        if not is_unknown:
            history = db.query(DiseaseHistory).filter(
                DiseaseHistory.disease_name == disease_name
            ).first()
            
            if history:
                history.total_detections += 1
                history.avg_confidence = (
                    (history.avg_confidence * (history.total_detections - 1) + confidence_score) 
                    / history.total_detections
                )
                if disease_stage == "Early":
                    history.early_stage_count += 1
                elif disease_stage == "Mid":
                    history.mid_stage_count += 1
                elif disease_stage == "Late":
                    history.late_stage_count += 1
            else:
                history = DiseaseHistory(
                    disease_name=disease_name,
                    total_detections=1,
                    avg_confidence=confidence_score,
                    early_stage_count=1 if disease_stage == "Early" else 0,
                    mid_stage_count=1 if disease_stage == "Mid" else 0,
                    late_stage_count=1 if disease_stage == "Late" else 0
                )
                db.add(history)
            
            db.commit()
        
        return {
            "prediction_id": prediction.id,
            "disease_name": disease_name,
            "confidence_score": confidence_score,
            "is_unknown": is_unknown,
            "disease_stage": disease_stage,
            "recommended_action": recommended_action,
            "estimated_yield_loss": estimated_yield_loss,
            "gradcam_path": gradcam_path or "",
            "cam_coverage": cam_coverage,
            "ai_advisory": ai_advisory,
            "image_filename": filename
        }
    
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/history")
def get_prediction_history(
    skip: int = 0,
    limit: int = 20,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's prediction history"""
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
    
    return predictions


@router.get("/history/{prediction_id}")
def get_prediction_detail(
    prediction_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed prediction information"""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    return prediction


@router.get("/gradcam/{prediction_id}")
def get_gradcam_image(
    prediction_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get Grad-CAM visualization image"""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    if not prediction.gradcam_path or not os.path.exists(prediction.gradcam_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grad-CAM image not available"
        )
    
    return FileResponse(prediction.gradcam_path, media_type="image/jpeg")


@router.get("/image/{prediction_id}")
def get_uploaded_image(
    prediction_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get original uploaded image"""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    if not prediction.image_path or not os.path.exists(prediction.image_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not available"
        )
    
    
    return FileResponse(prediction.image_path, media_type="image/jpeg")


class ReportRequest(BaseModel):
    prediction_id: int
    proposed_label: Optional[str] = None
    description: Optional[str] = None


@router.post("/report", status_code=status.HTTP_201_CREATED)
def report_unknown_case(
    report: ReportRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Report an incorrect or unknown prediction to admin.
    """
    # 1. Verify prediction exists and belongs to user
    prediction = db.query(Prediction).filter(
        Prediction.id == report.prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found or access denied"
        )
        
    # 2. Check if already reported
    from backend.models import ReportedCase
    existing = db.query(ReportedCase).filter(
        ReportedCase.prediction_id == report.prediction_id
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This prediction has already been reported"
        )
        
    # 3. Create Report
    new_report = ReportedCase(
        prediction_id=prediction.id,
        user_id=current_user.id,
        proposed_label=report.proposed_label,
        description=report.description,
        status="PENDING"
    )
    
    db.add(new_report)
    db.commit()
    
    return {"message": "Report submitted successfully. An expert will review it soon."}
