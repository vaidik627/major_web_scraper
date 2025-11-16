from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import uuid

from database import get_db
from models import User, ScrapingJob, ScrapedData
from routers.auth import get_current_user
from services.scraper_service import ScraperService
from services.ai_service import AIService
from services.enhanced_ai_integration import EnhancedAIIntegrationService

router = APIRouter()

# Initialize services
scraper_service = ScraperService()
ai_service = AIService()
enhanced_ai_service = EnhancedAIIntegrationService()

# Pydantic models
class ScrapingConfig(BaseModel):
    css_selector: Optional[str] = None
    xpath: Optional[str] = None
    keywords: Optional[List[str]] = None
    data_type: str = "text"  # text, images, links, prices, etc.
    use_playwright: bool = False
    wait_time: int = 3
    max_pages: int = 1
    follow_links: bool = False
    extract_images: bool = False
    extract_links: bool = False
    custom_headers: Optional[Dict[str, str]] = None
    enhanced_summarization: Optional[Dict] = None  # Enhanced AI summarization config
    use_enhanced_ai: bool = False  # Use multi-model AI system
    ai_models: Optional[List[str]] = None  # Specific AI models to use

class ScrapingRequest(BaseModel):
    name: str
    urls: List[HttpUrl]
    config: ScrapingConfig
    use_ai: bool = True
    ai_prompt: Optional[str] = None

class EnhancedSummaryRequest(BaseModel):
    content: str
    title: Optional[str] = ""
    url: Optional[str] = ""
    summary_type: str = "balanced"  # brief, balanced, detailed, executive
    detail_level: str = "medium"    # short, medium, long, comprehensive
    output_format: str = "mixed"    # paragraph, bullet_points, mixed
    focus_areas: Optional[List[str]] = None  # main_content, key_facts, actionable_items, etc.
    highlight_relevant_text: bool = True
    include_keywords: bool = True
    max_length: Optional[int] = None
    user_query: Optional[str] = None

class EnhancedAIScrapingRequest(BaseModel):
    name: str
    urls: List[HttpUrl]
    config: ScrapingConfig
    ai_prompt: Optional[str] = None
    use_multi_model: bool = True  # Use multiple AI models
    models: Optional[List[str]] = None  # Specific models: ["gpt4", "claude", "bart"]

class EnhancedSummaryResponse(BaseModel):
    enhanced_summary: Dict[str, Any]
    key_points: List[str]
    highlights: List[Dict[str, Any]]
    enhanced_keywords: List[Dict[str, Any]]
    confidence_score: float
    model: str

class ScrapingJobResponse(BaseModel):
    id: int
    name: str
    status: str
    total_urls: int
    processed_urls: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class ScrapedDataResponse(BaseModel):
    id: int
    url: str
    title: Optional[str]
    content: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    ai_analysis: Optional[Dict[str, Any]]
    status: str
    scraped_at: datetime
    processing_time: Optional[float]
    
    class Config:
        from_attributes = True

# Initialize services
scraper_service = ScraperService()
ai_service = AIService()

@router.post("/scrape", response_model=ScrapingJobResponse)
async def create_scraping_job(
    request: ScrapingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new scraping job"""
    # Create job record
    job = ScrapingJob(
        user_id=current_user.id,
        name=request.name,
        urls=[str(url) for url in request.urls],
        config=request.config.dict(),
        total_urls=len(request.urls),
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start scraping in background
    background_tasks.add_task(
        run_scraping_job,
        job.id,
        request.urls,
        request.config,
        request.use_ai,
        request.ai_prompt
    )
    
    return job

@router.post("/enhanced-ai-scrape", response_model=ScrapingJobResponse)
async def create_enhanced_ai_scraping_job(
    request: EnhancedAIScrapingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new scraping job with enhanced multi-model AI analysis"""
    
    # Enable enhanced AI in config
    request.config.use_enhanced_ai = True
    
    # Create job
    job = ScrapingJob(
        user_id=current_user.id,
        name=request.name,
        urls=[str(url) for url in request.urls],
        config=request.config.dict(),
        total_urls=len(request.urls),
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start enhanced AI scraping in background
    background_tasks.add_task(
        run_enhanced_ai_scraping_job,
        job.id,
        request.urls,
        request.config,
        request.ai_prompt,
        request.use_multi_model,
        request.models
    )
    
    return job

@router.get("/jobs", response_model=List[ScrapingJobResponse])
async def get_user_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get user's scraping jobs"""
    jobs = db.query(ScrapingJob).filter(
        ScrapingJob.user_id == current_user.id
    ).offset(skip).limit(limit).all()
    return jobs

@router.get("/jobs/{job_id}", response_model=ScrapingJobResponse)
async def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific job details"""
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

@router.get("/jobs/{job_id}/data", response_model=List[ScrapedDataResponse])
async def get_job_data(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get scraped data for a specific job"""
    # Verify job ownership
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    data = db.query(ScrapedData).filter(
        ScrapedData.job_id == job_id
    ).offset(skip).limit(limit).all()
    
    return data

@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a scraping job and its data"""
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete associated data
    db.query(ScrapedData).filter(ScrapedData.job_id == job_id).delete()
    db.delete(job)
    db.commit()
    
    return {"message": "Job deleted successfully"}

@router.post("/quick-scrape")
async def quick_scrape(
    url: HttpUrl,
    config: Optional[ScrapingConfig] = None,
    use_ai: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Quick scrape a single URL without saving to database"""
    if not config:
        config = ScrapingConfig()
    
    try:
        # Scrape the URL
        result = await scraper_service.scrape_url(str(url), config.dict())
        
        # Apply AI analysis if requested
        if use_ai and result.get("content"):
            ai_analysis = await ai_service.analyze_content(
                result["content"],
                result.get("title", ""),
                str(url)
            )
            result["ai_analysis"] = ai_analysis
        
        return {
            "url": str(url),
            "data": result,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_scraping_job(
    job_id: int,
    urls: List[HttpUrl],
    config: ScrapingConfig,
    use_ai: bool,
    ai_prompt: Optional[str]
):
    """Background task to run scraping job"""
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        # Update job status
        job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        processed = 0
        for url in urls:
            try:
                # Scrape URL
                result = await scraper_service.scrape_url(str(url), config.dict())
                
                # Apply AI analysis if requested
                ai_analysis = None
                enhanced_summary = None
                
                if use_ai and result.get("content"):
                    # Check if enhanced AI is requested
                    if config.use_enhanced_ai:
                        try:
                            # Use enhanced multi-model AI system
                            enhanced_analysis = await enhanced_ai_service.analyze_content_with_enhanced_ai(
                                content=result["content"],
                                title=result.get("title", ""),
                                url=str(url),
                                user_id=job.user_id,
                                custom_prompt=ai_prompt
                            )
                            ai_analysis = enhanced_analysis
                            print(f"✅ Enhanced AI analysis completed for {url}")
                        except Exception as e:
                            print(f"❌ Enhanced AI failed for {url}: {str(e)}")
                            # Fallback to standard AI
                            ai_analysis = await ai_service.analyze_content(
                                result["content"],
                                result.get("title", ""),
                                str(url),
                                custom_prompt=ai_prompt
                            )
                    else:
                        # Standard AI analysis
                        ai_analysis = await ai_service.analyze_content(
                            result["content"],
                            result.get("title", ""),
                            str(url),
                            custom_prompt=ai_prompt
                        )
                    
                    # Enhanced summarization if enabled
                    enhanced_config = config.dict().get("enhanced_summarization")
                    if enhanced_config:
                        try:
                            enhanced_summary = ai_service.generate_enhanced_summary(
                                content=result["content"],
                                title=result.get("title", ""),
                                url=str(url),
                                summary_type=enhanced_config.get("summaryType", "balanced"),
                                detail_level=enhanced_config.get("detailLevel", "medium"),
                                output_format=enhanced_config.get("outputFormat", "paragraph"),
                                focus_areas=enhanced_config.get("focusAreas", []),
                                highlight_relevant_text=enhanced_config.get("highlightRelevantText", True),
                                include_keywords=enhanced_config.get("includeKeywords", True),
                                max_length=enhanced_config.get("maxLength"),
                                user_query=enhanced_config.get("userQuery")
                            )
                            
                            # Add enhanced summary to AI analysis
                            if ai_analysis:
                                ai_analysis["enhanced_summary"] = enhanced_summary
                            else:
                                ai_analysis = {"enhanced_summary": enhanced_summary}
                                
                        except Exception as e:
                            print(f"Enhanced summarization failed for {url}: {str(e)}")
                            # Continue with standard analysis if enhanced fails
                
                # Save scraped data
                # Determine status and processing time
                has_error = bool(result.get("error"))
                has_content = bool(result.get("content"))
                status = "failed" if has_error or not has_content else "success"
                processing_time = result.get("metadata", {}).get("processing_time") or result.get("processing_time")

                scraped_data = ScrapedData(
                    job_id=job_id,
                    url=str(url),
                    title=result.get("title"),
                    content=result.get("content"),
                    extracted_data=result.get("extracted_data"),
                    ai_analysis=ai_analysis if status == "success" else None,
                    status=status,
                    error_message=result.get("error") if status == "failed" else None,
                    processing_time=processing_time
                )
                db.add(scraped_data)
                processed += 1
                
            except Exception as e:
                # Save error data
                scraped_data = ScrapedData(
                    job_id=job_id,
                    url=str(url),
                    status="failed",
                    error_message=str(e)
                )
                db.add(scraped_data)
            
            # Update progress
            job.processed_urls = processed
            db.commit()
        
        # Mark job as completed
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        # Mark job as failed
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

@router.post("/enhanced-summary", response_model=EnhancedSummaryResponse)
async def generate_enhanced_summary(
    request: EnhancedSummaryRequest
):
    """Generate enhanced summary with customization options and text highlighting"""
    try:
        if not request.content or len(request.content.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Content is required for summary generation"
            )
        
        # Generate enhanced summary using the enhanced summarization service
        from services.enhanced_summarization_service import (
            EnhancedSummarizationService, 
            SummaryCustomization, 
            SummaryType, 
            DetailLevel, 
            OutputFormat, 
            FocusArea
        )
        enhanced_service = EnhancedSummarizationService()
        
        # Convert string values to enums
        summary_type_enum = SummaryType(request.summary_type) if request.summary_type else SummaryType.BALANCED
        detail_level_enum = DetailLevel(request.detail_level) if request.detail_level else DetailLevel.MEDIUM
        output_format_enum = OutputFormat(request.output_format) if request.output_format else OutputFormat.MIXED
        
        # Convert focus areas to enums if provided
        focus_areas_enums = []
        if request.focus_areas:
            for area in request.focus_areas:
                try:
                    focus_areas_enums.append(FocusArea(area))
                except ValueError:
                    # Skip invalid focus areas
                    pass
        
        # Create customization object
        customization = SummaryCustomization(
            summary_type=summary_type_enum,
            detail_level=detail_level_enum,
            output_format=output_format_enum,
            focus_areas=focus_areas_enums,
            highlight_relevant_text=request.highlight_relevant_text,
            include_keywords=request.include_keywords,
            max_length=request.max_length,
            user_query=request.user_query
        )
        
        summary_result = enhanced_service.generate_enhanced_summary(
            content=request.content,
            title=request.title or "",
            url=request.url or "",
            customization=customization
        )
        
        # Convert EnhancedSummary dataclass to response dict matching schema
        highlights = [
            {
                "text": h.text,
                "relevance": h.relevance,
                "context": h.context
            } for h in (summary_result.highlights or [])
        ]
        
        model_name = (
            "openai" if getattr(enhanced_service, "openai_client", None)
            else ("anthropic" if getattr(enhanced_service, "anthropic_client", None) else "enhanced_local")
        )

        return {
            "enhanced_summary": {
                "text": summary_result.text,
                "type": request.summary_type,
                "word_count": len((summary_result.text or "").split()),
                "metadata": summary_result.metadata or {}
            },
            "key_points": summary_result.key_points or [],
            "highlights": highlights,
            "enhanced_keywords": summary_result.keywords or [],
            "confidence_score": summary_result.confidence_score or 0.0,
            "model": model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate enhanced summary: {str(e)}"
        )



@router.post("/quick-enhanced-summary")
async def quick_enhanced_summary(
    url: str,
    summary_type: str = "balanced",
    detail_level: str = "medium",
    user_query: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Quick enhanced summary generation for a single URL"""
    try:
        # Scrape the URL first
        config = ScrapingConfig()
        scrape_result = await scraper_service.scrape_url(url, config.dict())
        
        if not scrape_result.get("content"):
            raise HTTPException(
                status_code=400,
                detail="Failed to extract content from URL"
            )
        
        # Generate enhanced summary
        summary_result = ai_service.generate_enhanced_summary(
            content=scrape_result["content"],
            title=scrape_result.get("title", ""),
            url=url,
            summary_type=summary_type,
            detail_level=detail_level,
            user_query=user_query,
            highlight_relevant_text=True,
            include_keywords=True
        )
        
        return {
            "url": url,
            "title": scrape_result.get("title", ""),
            "content_preview": scrape_result["content"][:200] + "..." if len(scrape_result["content"]) > 200 else scrape_result["content"],
            "enhanced_summary": summary_result,
            "scraping_metadata": scrape_result.get("metadata", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate quick enhanced summary: {str(e)}"
        )

async def run_enhanced_ai_scraping_job(
    job_id: int,
    urls: List[HttpUrl],
    config: ScrapingConfig,
    ai_prompt: Optional[str],
    use_multi_model: bool,
    models: Optional[List[str]]
):
    """Background task to run enhanced AI scraping job"""
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        # Update job status
        job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        processed = 0
        for url in urls:
            try:
                # Scrape URL
                result = await scraper_service.scrape_url(str(url), config.dict())
                
                # Apply enhanced AI analysis
                enhanced_analysis = None
                
                if result.get("content"):
                    try:
                        # Use enhanced multi-model AI system
                        enhanced_analysis = await enhanced_ai_service.analyze_content_with_enhanced_ai(
                            content=result["content"],
                            title=result.get("title", ""),
                            url=str(url),
                            user_id=job.user_id,
                            custom_prompt=ai_prompt
                        )
                        print(f"🚀 Enhanced AI analysis completed for {url}")
                        print(f"📊 Analysis includes: {list(enhanced_analysis.keys())}")
                        
                    except Exception as e:
                        print(f"❌ Enhanced AI failed for {url}: {str(e)}")
                        # Fallback to standard AI
                        enhanced_analysis = await ai_service.analyze_content(
                            result["content"],
                            result.get("title", ""),
                            str(url),
                            custom_prompt=ai_prompt
                        )
                
                # Save scraped data
                has_error = bool(result.get("error"))
                has_content = bool(result.get("content"))
                status = "failed" if has_error or not has_content else "success"
                processing_time = result.get("metadata", {}).get("processing_time") or result.get("processing_time")

                scraped_data = ScrapedData(
                    job_id=job_id,
                    url=str(url),
                    title=result.get("title"),
                    content=result.get("content"),
                    extracted_data=result.get("extracted_data"),
                    ai_analysis=enhanced_analysis if status == "success" else None,
                    status=status,
                    error_message=result.get("error") if status == "failed" else None,
                    processing_time=processing_time
                )
                db.add(scraped_data)
                processed += 1
                
            except Exception as e:
                # Save error data
                scraped_data = ScrapedData(
                    job_id=job_id,
                    url=str(url),
                    status="failed",
                    error_message=str(e)
                )
                db.add(scraped_data)
            
            # Update progress
            job.processed_urls = processed
            db.commit()
        
        # Mark job as completed
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        print(f"Enhanced AI Job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
    finally:
        db.close()