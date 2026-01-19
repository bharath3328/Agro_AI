from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional

from backend.database import get_db
from backend.models import User
from backend.auth_utils import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_active_user,
    generate_verification_code
)
from backend.config import settings
from backend.notification_service import NotificationService

router = APIRouter(prefix="/auth", tags=["Authentication"])


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    is_admin: bool
    is_active: bool
    is_verified: bool

    class Config:
        from_attributes = True

class RegisterResponse(BaseModel):
    message: str
    verification_token: str 
    email: str 

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class VerifyAccountRequest(BaseModel):
    email: EmailStr
    code: str
    verification_token: str


class ForgotPasswordRequest(BaseModel):
    email: Optional[EmailStr] = None


class ForgotPasswordResponse(BaseModel):
    message: str
    verification_token: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    email: Optional[EmailStr] = None
    verification_token: str
    code: str
    new_password: str


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    
    # Check if user already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    try:
        # Prepare user data (stateless)
        hashed_password = get_password_hash(user_data.password)
        
        # We store these details in the token to persist them until verification
        registration_data = {
            "reg_username": user_data.username,
            "reg_password_hash": hashed_password,
            "reg_full_name": user_data.full_name,
            "reg_phone": user_data.phone
        }
        
        # Generate verification code
        code = generate_verification_code()
        
        # Create stateless token with user data embedded
        from backend.auth_utils import create_verification_token
        verification_token = create_verification_token(
            email=user_data.email, 
            code=code, 
            purpose="REGISTRATION",
            extra_data=registration_data
        )
        
        # Send email in background
        background_tasks.add_task(
            NotificationService.send_verification_code, 
            user_data.email, 
            code, 
            "EMAIL"
        )
        
        return {
            "message": "Verification code sent",
            "verification_token": verification_token,
            "email": user_data.email
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/verify")
def verify_account(request: VerifyAccountRequest, db: Session = Depends(get_db)):
    from backend.auth_utils import verify_code_token

    # Verify logic (stateless) - returns payload if valid
    payload = verify_code_token(request.verification_token, request.code, request.email, purpose="REGISTRATION")
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        if existing_user.is_verified:
            return {"message": "Account already verified"}
        else:
            # Legacy support: User exists but not verified.
            # Mark as verified since token was valid
            existing_user.is_verified = True
            db.commit()
            return {"message": "Account verified successfully"}
        
    # Extract data from payload
    try:
        username = payload.get("reg_username")
        password_hash = payload.get("reg_password_hash")
        full_name = payload.get("reg_full_name")
        phone = payload.get("reg_phone")
        
        if not username or not password_hash:
             raise HTTPException(status_code=400, detail="Invalid token data: Missing registration info")

        # Create user now
        db_user = User(
            email=request.email,
            username=username,
            hashed_password=password_hash,
            full_name=full_name,
            phone=phone,
            is_verified=True, # Verified immediately
            is_active=True
        )
        
        db.add(db_user)
        db.commit()
        
    except Exception as e:
        db.rollback()
        print(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user account")
    
    return {"message": "Account verified and created successfully"}


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    if not user.is_verified:
        raise HTTPException(status_code=400, detail="Account not verified. Please verify your email.")
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = None
    
    if request.email:
        user = db.query(User).filter(User.email == request.email).first()
    
    # Check if user exists but don't reveal it if not
    if not user:
        return {"message": "If an account exists, a code has been sent."}
    
    # Generate code
    code = generate_verification_code()
    
    # Create stateless token
    from backend.auth_utils import create_verification_token
    verification_token = create_verification_token(user.email, code, purpose="PASSWORD_RESET")
    
    # Send notification
    background_tasks.add_task(
        NotificationService.send_verification_code,
        user.email,
        code,
        "EMAIL"
    )
     
    return {
        "message": "Code sent successfully",
        "verification_token": verification_token
    }


@router.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    if not request.email:
         raise HTTPException(status_code=400, detail="Email is required")

    # Verify the token
    from backend.auth_utils import verify_code_token
    verify_code_token(request.verification_token, request.code, request.email, purpose="PASSWORD_RESET")

    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Update password
    user.hashed_password = get_password_hash(request.new_password)
    db.commit()
    
    return {"message": "Password reset successfully"}


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return current_user
