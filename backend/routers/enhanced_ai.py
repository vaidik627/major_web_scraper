"""
Enhanced AI API Router
New endpoints for the enhanced AI functionality
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
import logging

from database import get_db
from services.ai_service import AIService
from services.enhanced_ai_integration import EnhancedAIIntegrationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enhanced-ai", tags=["Enhanced AI"])

# Pydantic models for request/response
class EnhancedAnalysisRequest(BaseModel):
    content: str
    title: str = ""
    url: str = ""
    user_id: Optional[str] = None
    max_length: int = 500
    enable_personalization: bool = True

class FeedbackRequest(BaseModel):
    user_id: str
    summary_id: str
    rating: int  # 1-5 scale
    feedback_text: str = ""
    domain: str = ""
    technology: str = ""
    metadata: Optional[Dict[str, Any]] = None

class TechScrapingRequest(BaseModel):
    urls: List[str]
    tech_domain: str  # ai_ml, web_dev, mobile_dev, etc.
    user_id: Optional[str] = None
    enable_personalization: bool = True

class LearningPathRequest(BaseModel):
    start_technology: str
    target_technology: str
    max_steps: int = 5

class TechSuggestionRequest(BaseModel):
    current_technologies: List[str]
    target_domain: Optional[str] = None
    max_suggestions: int = 5

@router.post("/analyze")
async def analyze_content_enhanced(
    request: EnhancedAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Analyze content using enhanced AI with multi-model ensemble and domain awareness"""
    
    try:
        ai_service = AIService(db)
        
        # Use enhanced AI analysis
        result = await ai_service.analyze_content_with_enhanced_ai(
            content=request.content,
            title=request.title,
            url=request.url,
            user_id=request.user_id,
            max_length=request.max_length,
            enable_personalization=request.enable_personalization
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content analyzed successfully with enhanced AI"
        }
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/feedback")
async def collect_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Collect user feedback for continuous improvement"""
    
    try:
        ai_service = AIService(db)
        
        # Collect feedback
        await ai_service.collect_user_feedback(
            user_id=request.user_id,
            summary_id=request.summary_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            domain=request.domain,
            technology=request.technology,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "message": "Feedback collected successfully"
        }
        
    except Exception as e:
        logger.error(f"Feedback collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback collection failed: {str(e)}")

@router.get("/user-insights/{user_id}")
async def get_user_insights(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get personalized insights for a user"""
    
    try:
        ai_service = AIService(db)
        
        insights = await ai_service.get_user_insights(user_id)
        
        return {
            "success": True,
            "data": insights,
            "message": "User insights retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get user insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user insights: {str(e)}")

@router.post("/scrape-tech")
async def scrape_tech_content(
    request: TechScrapingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Scrape technology-specific content with domain-aware analysis"""
    
    try:
        ai_service = AIService(db)
        
        if not ai_service.enhanced_integration_service:
            raise HTTPException(status_code=503, detail="Enhanced AI service not available")
        
        # Import TechDomain enum
        from services.tech_specialized_scraper import TechDomain
        
        # Map string to TechDomain enum
        tech_domain_mapping = {
            'ai_ml': TechDomain.AI_ML,
            'web_dev': TechDomain.WEB_DEVELOPMENT,
            'mobile_dev': TechDomain.MOBILE_DEVELOPMENT,
            'devops': TechDomain.DEVOPS,
            'cybersecurity': TechDomain.CYBERSECURITY,
            'data_science': TechDomain.DATA_SCIENCE,
            'blockchain': TechDomain.BLOCKCHAIN,
            'cloud': TechDomain.CLOUD_COMPUTING,
            'game_dev': TechDomain.GAME_DEVELOPMENT,
            'embedded': TechDomain.EMBEDDED_SYSTEMS
        }
        
        tech_domain = tech_domain_mapping.get(request.tech_domain)
        if not tech_domain:
            raise HTTPException(status_code=400, detail=f"Invalid tech domain: {request.tech_domain}")
        
        # Scrape and analyze content
        results = await ai_service.enhanced_integration_service.scrape_and_analyze_tech_content(
            urls=request.urls,
            tech_domain=tech_domain,
            user_id=request.user_id,
            enable_personalization=request.enable_personalization
        )
        
        return {
            "success": True,
            "data": [
                {
                    "url": result.summary,
                    "domain": result.domain,
                    "confidence": result.confidence,
                    "technical_elements": result.technical_elements,
                    "related_technologies": result.related_technologies,
                    "personalized_for_user": result.personalized_for_user,
                    "metadata": result.metadata
                }
                for result in results
            ],
            "message": f"Scraped and analyzed {len(results)} URLs successfully"
        }
        
    except Exception as e:
        logger.error(f"Tech scraping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tech scraping failed: {str(e)}")

@router.post("/learning-path")
async def find_learning_path(
    request: LearningPathRequest,
    db: Session = Depends(get_db)
):
    """Find learning path between technologies"""
    
    try:
        ai_service = AIService(db)
        
        learning_path = await ai_service.find_learning_path(
            start_technology=request.start_technology,
            target_technology=request.target_technology,
            max_steps=request.max_steps
        )
        
        return {
            "success": True,
            "data": learning_path,
            "message": "Learning path found successfully"
        }
        
    except Exception as e:
        logger.error(f"Learning path search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Learning path search failed: {str(e)}")

@router.post("/suggest-technologies")
async def suggest_technologies(
    request: TechSuggestionRequest,
    db: Session = Depends(get_db)
):
    """Suggest technologies based on current tech stack"""
    
    try:
        ai_service = AIService(db)
        
        suggestions = await ai_service.suggest_technologies(
            current_technologies=request.current_technologies,
            target_domain=request.target_domain,
            max_suggestions=request.max_suggestions
        )
        
        return {
            "success": True,
            "data": suggestions,
            "message": "Technology suggestions generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Technology suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Technology suggestion failed: {str(e)}")

@router.get("/technology-ecosystem/{technology}")
async def get_technology_ecosystem(
    technology: str,
    db: Session = Depends(get_db)
):
    """Get technology ecosystem information"""
    
    try:
        ai_service = AIService(db)
        
        ecosystem = await ai_service.get_technology_ecosystem(technology)
        
        return {
            "success": True,
            "data": ecosystem,
            "message": "Technology ecosystem retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get technology ecosystem: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get technology ecosystem: {str(e)}")

@router.get("/system-health")
async def get_system_health(
    db: Session = Depends(get_db)
):
    """Get enhanced AI system health and performance metrics"""
    
    try:
        ai_service = AIService(db)
        
        if not ai_service.enhanced_integration_service:
            return {
                "success": True,
                "data": {"enhanced_ai_available": False},
                "message": "Enhanced AI service not available"
            }
        
        health = await ai_service.enhanced_integration_service.get_system_health()
        
        return {
            "success": True,
            "data": health,
            "message": "System health retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/domains")
async def get_supported_domains():
    """Get list of supported content domains"""
    
    domains = [
        {"id": "tech", "name": "Technology", "description": "General technology content"},
        {"id": "ai", "name": "AI/ML", "description": "Artificial Intelligence and Machine Learning"},
        {"id": "webdev", "name": "Web Development", "description": "Web development frameworks and tools"},
        {"id": "mobile", "name": "Mobile Development", "description": "Mobile app development"},
        {"id": "devops", "name": "DevOps", "description": "DevOps tools and practices"},
        {"id": "cybersecurity", "name": "Cybersecurity", "description": "Security and cybersecurity"},
        {"id": "finance", "name": "Finance", "description": "Financial and business content"},
        {"id": "medical", "name": "Medical", "description": "Medical and healthcare content"},
        {"id": "legal", "name": "Legal", "description": "Legal and compliance content"},
        {"id": "academic", "name": "Academic", "description": "Academic and research content"},
        {"id": "business", "name": "Business", "description": "Business and strategy content"},
        {"id": "general", "name": "General", "description": "General purpose content"}
    ]
    
    return {
        "success": True,
        "data": domains,
        "message": "Supported domains retrieved successfully"
    }

@router.get("/tech-domains")
async def get_supported_tech_domains():
    """Get list of supported technology domains"""
    
    tech_domains = [
        {"id": "ai_ml", "name": "AI/ML", "description": "Machine Learning and AI frameworks"},
        {"id": "web_dev", "name": "Web Development", "description": "Web frameworks and libraries"},
        {"id": "mobile_dev", "name": "Mobile Development", "description": "Mobile app frameworks"},
        {"id": "devops", "name": "DevOps", "description": "DevOps tools and platforms"},
        {"id": "cybersecurity", "name": "Cybersecurity", "description": "Security tools and practices"},
        {"id": "data_science", "name": "Data Science", "description": "Data analysis and visualization"},
        {"id": "blockchain", "name": "Blockchain", "description": "Blockchain and cryptocurrency"},
        {"id": "cloud", "name": "Cloud Computing", "description": "Cloud platforms and services"},
        {"id": "game_dev", "name": "Game Development", "description": "Game development frameworks"},
        {"id": "embedded", "name": "Embedded Systems", "description": "Embedded and IoT development"}
    ]
    
    return {
        "success": True,
        "data": tech_domains,
        "message": "Supported tech domains retrieved successfully"
    }
