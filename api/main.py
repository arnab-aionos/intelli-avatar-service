"""
FastAPI application for IntelliAvatar Service REST API.

Provides endpoints for:
- Session management (WebRTC)
- Avatar configuration
- Asynchronous video generation
- Authentication
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from api.routes import sessions, avatars, generation, auth

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting IntelliAvatar Service API")
    logger.info(f"Running on port {settings.api_port}")
    logger.info(f"Device mode: {settings.musetalk_device}")
    
    if settings.musetalk_device == "cpu":
        logger.warning(
            "Running in CPU mode. Avatar generation will be slow. "
            "GPU is recommended for production."
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down IntelliAvatar Service API")


# Create FastAPI app
app = FastAPI(
    title="IntelliAvatar Service",
    description="Production-ready real-time streaming avatar API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
app.include_router(avatars.router, prefix="/api/v1/avatars", tags=["Avatars"])
app.include_router(generation.router, prefix="/api/v1/generate", tags=["Generation"])


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "IntelliAvatar Service",
        "version": "1.0.0",
        "status": "running",
        "device": settings.musetalk_device,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": settings.musetalk_device,
        "models_path": str(settings.musetalk_models_full_path),
        "models_exist": settings.musetalk_models_full_path.exists()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
