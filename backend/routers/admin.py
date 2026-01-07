"""
Admin Router - Admin-only endpoints for user management and analytics
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional

from backend.database import get_db
from backend.models import User, Prediction, DiseaseHistory
from backend.auth_utils import get_current_admin_user

router = APIRouter(prefix="/admin", tags=["Admin"])


class UserStats(BaseModel):
    total_users: int
    active_users: int
    admin_users: int
    total_predictions: int
    unique_diseases_detected: int


class DiseaseStats(BaseModel):
    disease_name: str
    total_detections: int
    avg_confidence: float
    early_stage_count: int
    mid_stage_count: int
    late_stage_count: int
    crop_type: Optional[str] = None


class UserInfo(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_admin: bool
    is_active: bool
    prediction_count: int
    created_at: str

    class Config:
        from_attributes = True


@router.get("/stats", response_model=UserStats)
def get_system_stats(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get overall system statistics"""
    
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    admin_users = db.query(func.count(User.id)).filter(User.is_admin == True).scalar()
    total_predictions = db.query(func.count(Prediction.id)).scalar()
    unique_diseases = db.query(func.count(func.distinct(Prediction.disease_name))).filter(
        Prediction.is_unknown == False
    ).scalar() or 0
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "admin_users": admin_users,
        "total_predictions": total_predictions,
        "unique_diseases_detected": unique_diseases
    }


@router.get("/diseases", response_model=List[DiseaseStats])
def get_disease_statistics(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get statistics for all detected diseases"""
    
    diseases = db.query(DiseaseHistory).order_by(
        DiseaseHistory.total_detections.desc()
    ).all()
    
    return diseases


@router.get("/users", response_model=List[UserInfo])
def get_all_users(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all users with their prediction counts"""
    
    users = db.query(
        User,
        func.count(Prediction.id).label('prediction_count')
    ).outerjoin(
        Prediction, User.id == Prediction.user_id
    ).group_by(User.id).offset(skip).limit(limit).all()
    
    result = []
    for user, pred_count in users:
        result.append({
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
            "is_active": user.is_active,
            "prediction_count": pred_count or 0,
            "created_at": user.created_at.isoformat() if user.created_at else None
        })
    
    return result


@router.get("/users/{user_id}", response_model=UserInfo)
def get_user_details(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific user"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    pred_count = db.query(func.count(Prediction.id)).filter(
        Prediction.user_id == user_id
    ).scalar()
    
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "is_admin": user.is_admin,
        "is_active": user.is_active,
        "prediction_count": pred_count or 0,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }


@router.put("/users/{user_id}/activate")
def activate_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Activate a user account"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = True
    db.commit()
    
    return {"message": "User activated successfully"}


@router.put("/users/{user_id}/deactivate")
def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Deactivate a user account"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    user.is_active = False
    db.commit()
    
    return {"message": "User deactivated successfully"}


@router.put("/users/{user_id}/make-admin")
def make_admin(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Grant admin privileges to a user"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_admin = True
    db.commit()
    
    return {"message": "Admin privileges granted"}


@router.get("/predictions")
def get_all_predictions(
    skip: int = 0,
    limit: int = 50,
    disease_name: Optional[str] = None,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all predictions (admin view)"""
    
    query = db.query(Prediction)
    
    if disease_name:
        query = query.filter(Prediction.disease_name == disease_name)
    
    predictions = query.order_by(
        Prediction.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return predictions


@router.post("/train")
async def train_new_disease(
    disease_name: str = Form(...),
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Train the model on a new disease class (Admin Only).
    Workflow:
    1. Save uploaded images to data/fewshot/train/{disease_name}
    2. Trigger MLService to re-compute prototypes
    """
    import os
    import shutil
    from backend.config import settings
    from backend.ml_service import ml_service

    # 1. Prepare Directory
    # Sanitize disease name (simple alphanumeric check or replace spaces)
    safe_name = "".join(c for c in disease_name if c.isalnum() or c in (' ', '_', '-')).strip()
    class_dir = os.path.join(settings.TRAIN_DATA_DIR, safe_name)
    
    if os.path.exists(class_dir):
        # Optional: Decide if we want to add to existing or error out
        # For now, let's allow adding to existing
        pass
    else:
        os.makedirs(class_dir)

    saved_count = 0
    for file in files:
        if file.filename:
            file_path = os.path.join(class_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
    
    if saved_count == 0:
        raise HTTPException(status_code=400, detail="No valid files saved.")

    # 2. Trigger Retraining
    try:
        ml_service.retrain_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    return {
        "message": f"Successfully trained new disease: {safe_name}",
        "images_added": saved_count,
        "total_classes": len(ml_service.class_names)
    }
    return {
        "message": f"Successfully trained new disease: {safe_name}",
        "images_added": saved_count,
        "total_classes": len(ml_service.class_names)
    }


class ReportResponse(BaseModel):
    id: int
    prediction_id: int
    user_id: int
    status: str
    proposed_label: Optional[str]
    description: Optional[str]
    image_url: str
    created_at: str

    class Config:
        from_attributes = True


@router.get("/reports", response_model=List[ReportResponse])
def get_pending_reports(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all pending reported cases"""
    from backend.models import ReportedCase
    
    reports = db.query(ReportedCase).filter(
        ReportedCase.status == "PENDING"
    ).order_by(ReportedCase.created_at.desc()).all()
    
    # Enrich with image URL (relative path for frontend)
    results = []
    for r in reports:
        # Construct image URL from prediction
        image_url = f"/api/v1/diagnosis/image/{r.prediction_id}"
        results.append({
            "id": r.id,
            "prediction_id": r.prediction_id,
            "user_id": r.user_id,
            "status": r.status,
            "proposed_label": r.proposed_label,
            "description": r.description,
            "image_url": image_url,
            "created_at": r.created_at.isoformat() if r.created_at else None
        })
        
    return results


class ApproveReportRequest(BaseModel):
    correct_label: str
    admin_notes: Optional[str] = None


@router.post("/reports/{report_id}/approve")
def approve_report(
    report_id: int,
    approval: ApproveReportRequest,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Approve a report:
    1. Move image to few-shot training set (under correct_label)
    2. Retrain model (update prototypes)
    3. Update report status
    """
    import os
    import shutil
    from backend.config import settings
    from backend.models import ReportedCase
    from backend.ml_service import ml_service
    
    report = db.query(ReportedCase).filter(ReportedCase.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    if report.status != "PENDING":
        raise HTTPException(status_code=400, detail="Report is not pending")
        
    # Get original image path
    if not report.prediction or not os.path.exists(report.prediction.image_path):
        raise HTTPException(status_code=404, detail="Original image file not found")
        
    source_path = report.prediction.image_path
    
    # Prepare destination
    safe_label = "".join(c for c in approval.correct_label if c.isalnum() or c in (' ', '_', '-')).strip()
    dest_dir = os.path.join(settings.TRAIN_DATA_DIR, safe_label)
    os.makedirs(dest_dir, exist_ok=True)
    
    # We copy instead of move to keep the prediction record valid
    filename = os.path.basename(source_path)
    dest_path = os.path.join(dest_dir, filename)
    
    try:
        shutil.copy2(source_path, dest_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy image: {str(e)}")
        
    # Update Status
    report.status = "APPROVED"
    report.admin_notes = approval.admin_notes
    
    # Use the approved label as the proposed label if not provided previously, or just for record
    if not report.proposed_label:
        report.proposed_label = safe_label
        
    db.commit()
    
    # Trigger Retraining
    try:
        ml_service.retrain_model()
    except Exception as e:
        # Note: We don't rollback DB here because the manual work was done, 
        # but we warn the admin.
        return {
            "message": "Report approved and image saved, BUT model update failed.",
            "error": str(e)
        }
        
    return {"message": f"Report approved. Image added to '{safe_label}' and model updated."}


@router.post("/reports/{report_id}/reject")
def reject_report(
    report_id: int,
    admin_notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Reject a report"""
    from backend.models import ReportedCase
    
    report = db.query(ReportedCase).filter(ReportedCase.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    report.status = "REJECTED"
    report.admin_notes = admin_notes
    db.commit()
    
    return {"message": "Report rejected."}
