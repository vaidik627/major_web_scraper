from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import bcrypt
from pydantic import BaseModel, EmailStr
import os

from database import get_db
from models import User
from services.email_service import email_service

router = APIRouter()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Utility functions
def verify_password(plain_password, hashed_password):
    # Bcrypt has a 72 byte limit, so we need to truncate the password
    truncated = plain_password[:72] if len(plain_password) > 72 else plain_password
    # Use bcrypt directly to avoid passlib backend detection issues on some platforms
    return bcrypt.checkpw(truncated.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    # Bcrypt has a 72 byte limit, so we need to truncate the password
    truncated = password[:72] if len(password) > 72 else password
    # Use bcrypt directly to avoid passlib backend detection issues on some platforms
    hashed = bcrypt.hashpw(truncated.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user

# Routes
@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    if get_user_by_username(db, user.username):
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Send welcome email (non-blocking - don't fail registration if email fails)
    try:
        email_sent = email_service.send_welcome_email(user.email, user.username)
        if email_sent:
            print(f"Welcome email sent successfully to {user.email}")
        else:
            print(f"Failed to send welcome email to {user.email}")
    except Exception as e:
        print(f"Error sending welcome email to {user.email}: {str(e)}")
        # Don't raise the exception - registration should succeed even if email fails
    
    return db_user

@router.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    authenticated_user = authenticate_user(db, user.username, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user.username}, expires_delta=access_token_expires
    )
    
    # Send login notification (non-blocking)
    try:
        email_service.send_login_notification(authenticated_user.email, authenticated_user.username)
    except Exception as e:
        print(f"Error sending login notification to {authenticated_user.email}: {str(e)}")
        # Don't raise the exception - login should succeed even if notification fails
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/verify-token")
async def verify_token(current_user: User = Depends(get_current_user)):
    return {"valid": True, "user": current_user.username}

@router.delete("/delete-account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Forcefully delete user account and all associated data
    """
    try:
        user_id = current_user.id
        username = current_user.username
        
        # Import models here to avoid circular imports
        from models import (
            ScrapingJob, ScrapedData, ExtractedEntity, ContentCategory, 
            AIInsight, ScrapingTemplate, APIKey, TrendData
        )
        
        # Force delete all related data in the correct order
        # 1. Delete AI insights
        ai_insights = db.query(AIInsight).join(ScrapedData).join(ScrapingJob).filter(ScrapingJob.user_id == user_id).all()
        for insight in ai_insights:
            db.delete(insight)
        
        # 2. Delete extracted entities
        entities = db.query(ExtractedEntity).join(ScrapedData).join(ScrapingJob).filter(ScrapingJob.user_id == user_id).all()
        for entity in entities:
            db.delete(entity)
        
        # 3. Delete content categories
        categories = db.query(ContentCategory).join(ScrapedData).join(ScrapingJob).filter(ScrapingJob.user_id == user_id).all()
        for category in categories:
            db.delete(category)
        
        # 4. Delete scraped data
        scraped_data = db.query(ScrapedData).join(ScrapingJob).filter(ScrapingJob.user_id == user_id).all()
        for data in scraped_data:
            db.delete(data)
        
        # 5. Delete scraping jobs
        jobs = db.query(ScrapingJob).filter(ScrapingJob.user_id == user_id).all()
        for job in jobs:
            db.delete(job)
        
        # 6. Delete scraping templates
        templates = db.query(ScrapingTemplate).filter(ScrapingTemplate.user_id == user_id).all()
        for template in templates:
            db.delete(template)
        
        # 7. Delete API keys
        api_keys = db.query(APIKey).filter(APIKey.user_id == user_id).all()
        for key in api_keys:
            db.delete(key)
        
        # 8. Delete trend data
        trend_data = db.query(TrendData).filter(TrendData.user_id == user_id).all()
        for trend in trend_data:
            db.delete(trend)
        
        # 9. Finally delete the user
        db.delete(current_user)
        
        # Commit all deletions
        db.commit()
        
        return {
            "message": "Account successfully deleted",
            "deleted_user": username,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
    except Exception as e:
        db.rollback()
        print(f"Account deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete account: {str(e)}"
        )