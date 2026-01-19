from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.database import init_db
from backend.ml_service import ml_service
from backend.routers import auth, diagnosis, admin
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AgroAI Backend")
    init_db()
    print("Database initialized")
    try:
        ml_service.initialize()
        print("ML models loaded successfully")
    except Exception as e:
        print(f"Warning: ML models failed to initialize: {e}")    
    yield
    print("Shutting down AgroAI Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered plant disease detection and advisory system",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(diagnosis.router, prefix=settings.API_V1_PREFIX)
app.include_router(admin.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {
        "message": "Welcome to AgroAI - Plant Disease Detection System",
        "version": settings.VERSION,
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX
    }


@app.get("/health")
def health_check():
    ml_status = "ready" if ml_service.classifier is not None else "not_initialized"
    return {
        "status": "healthy",
        "ml_models": ml_status,
        "version": settings.VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
