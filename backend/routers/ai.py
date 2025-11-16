from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime

from database import get_db
from models import User
from routers.auth import get_current_user
from services.ai_service import AIService
from services.scraper_service import ScraperService

router = APIRouter()

# Initialize services - will be updated with database session in endpoints
ai_service = None
scraper_service = ScraperService()

# Pydantic models
class SmartAnalysisRequest(BaseModel):
    url: HttpUrl
    analysis_type: str = "comprehensive"  # comprehensive, summary, extraction, sentiment
    custom_prompt: Optional[str] = None

class ContentAnalysisRequest(BaseModel):
    content: str
    title: Optional[str] = ""
    url: Optional[str] = ""
    analysis_type: str = "comprehensive"

class ComparisonRequest(BaseModel):
    urls: List[HttpUrl]
    comparison_criteria: List[str] = ["prices", "features", "reviews"]
    
class TrendAnalysisRequest(BaseModel):
    job_ids: List[int]
    metric: str = "sentiment"  # sentiment, content_length, processing_time

class EnhancedAnalysisRequest(BaseModel):
    url: HttpUrl
    analysis_types: List[str] = ["entities", "categories", "trends", "insights"]
    custom_prompt: Optional[str] = None

class UserTrendsRequest(BaseModel):
    domain: Optional[str] = None
    days: int = 30
    trend_types: Optional[List[str]] = None

class DomainComparisonRequest(BaseModel):
    domains: List[str]
    metric: str = "sentiment"

class ComprehensiveAnalysisRequest(BaseModel):
    content: str
    title: Optional[str] = ""
    url: Optional[str] = ""
    analysis_features: List[str] = ["summary", "keywords", "topics", "sentiment", "entities", "insights"]
    days: int = 30

class ContentEnhancedAnalysisRequest(BaseModel):
    content: str
    title: Optional[str] = ""
    url: Optional[str] = ""
    analysis_types: List[str] = ["entities", "categories", "trends", "insights"]
    custom_prompt: Optional[str] = None

class TargetedExtractRequest(BaseModel):
    url: HttpUrl
    extract_text: bool = True
    extract_links: bool = True
    extract_images: bool = True
    extract_emails: bool = True
    extract_phones: bool = False
    use_playwright: bool = True
    wait_time: int = 3
    custom_prompt: Optional[str] = None

class TargetedExtractResponse(BaseModel):
    url: str
    title: Optional[str]
    content_length: int
    links_found: int
    images_found: int
    emails_found: int
    phones_found: int
    extracted_data: Dict[str, Any]
    ai_insights: Dict[str, Any]

@router.post("/smart-analysis")
async def smart_analysis(
    request: SmartAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform smart AI analysis on a URL"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        
        # First scrape the URL
        scraping_config = {
            "use_playwright": True,
            "wait_time": 3,
            "extract_links": True,
            "extract_images": True
        }
        
        scraped_data = await scraper_service.scrape_url(str(request.url), scraping_config)
        
        if scraped_data.get("error"):
            raise HTTPException(status_code=400, detail=f"Scraping failed: {scraped_data['error']}")
        
        # Perform AI analysis
        ai_analysis = await ai_service_instance.analyze_content(
            content=scraped_data.get("content", ""),
            title=scraped_data.get("title", ""),
            url=str(request.url),
            custom_prompt=request.custom_prompt
        )
        
        # Combine results based on analysis type
        if request.analysis_type == "summary":
            return {
                "url": str(request.url),
                "summary": ai_analysis.get("summary", ""),
                "key_points": ai_analysis.get("smart_extraction", {}).get("key_information", [])
            }
        elif request.analysis_type == "extraction":
            return {
                "url": str(request.url),
                "extracted_data": scraped_data.get("extracted_data", {}),
                "smart_extraction": ai_analysis.get("smart_extraction", {}),
                "entities": ai_analysis.get("entities", {})
            }
        elif request.analysis_type == "sentiment":
            return {
                "url": str(request.url),
                "sentiment": ai_analysis.get("sentiment", {}),
                "content_classification": ai_analysis.get("content_type", ""),
                "readability_score": ai_analysis.get("readability_score", 0)
            }
        else:  # comprehensive
            return {
                "url": str(request.url),
                "scraped_data": scraped_data,
                "ai_analysis": ai_analysis,
                "smart_insights": {
                    "content_type": ai_analysis.get("content_type", ""),
                    "key_entities": ai_analysis.get("entities", {}),
                    "actionable_insights": ai_analysis.get("smart_extraction", {}).get("actionable_insights", [])
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-content")
async def analyze_content(
    request: ContentAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze provided content with AI"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        
        analysis = await ai_service_instance.analyze_content(
            content=request.content,
            title=request.title,
            url=request.url
        )
        
        if request.analysis_type == "summary":
            return {
                "summary": analysis.get("summary", ""),
                "keywords": analysis.get("keywords", [])
            }
        elif request.analysis_type == "sentiment":
            return {
                "sentiment": analysis.get("sentiment", {}),
                "language": analysis.get("language", ""),
                "readability_score": analysis.get("readability_score", 0)
            }
        else:
            return analysis
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-sites")
async def compare_sites(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare multiple websites using AI"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        comparison_results = []
        
        for url in request.urls:
            # Scrape each URL
            scraping_config = {
                "use_playwright": True,
                "wait_time": 3,
                "data_type": "prices" if "prices" in request.comparison_criteria else "text"
            }
            
            scraped_data = await scraper_service.scrape_url(str(url), scraping_config)
            
            if not scraped_data.get("error"):
                # Analyze with AI
                ai_analysis = await ai_service_instance.analyze_content(
                    content=scraped_data.get("content", ""),
                    title=scraped_data.get("title", ""),
                    url=str(url)
                )
                
                comparison_results.append({
                    "url": str(url),
                    "title": scraped_data.get("title", ""),
                    "data": scraped_data,
                    "analysis": ai_analysis
                })
        
        # Generate comparison insights
        comparison_insights = await _generate_comparison_insights(comparison_results, request.comparison_criteria)
        
        return {
            "comparison_results": comparison_results,
            "insights": comparison_insights,
            "criteria": request.comparison_criteria
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trend-analysis")
async def trend_analysis(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze trends across multiple scraping jobs"""
    from models import ScrapingJob, ScrapedData
    
    try:
        # Verify job ownership and get data
        jobs = db.query(ScrapingJob).filter(
            ScrapingJob.id.in_(request.job_ids),
            ScrapingJob.user_id == current_user.id
        ).all()
        
        if len(jobs) != len(request.job_ids):
            raise HTTPException(status_code=404, detail="Some jobs not found or not accessible")
        
        trend_data = []
        
        for job in jobs:
            scraped_items = db.query(ScrapedData).filter(ScrapedData.job_id == job.id).all()
            
            job_metrics = {
                "job_id": job.id,
                "job_name": job.name,
                "created_at": job.created_at,
                "data_points": []
            }
            
            for item in scraped_items:
                data_point = {"url": item.url, "scraped_at": item.scraped_at}
                
                if request.metric == "sentiment" and item.ai_analysis:
                    sentiment = item.ai_analysis.get("sentiment", {})
                    data_point["value"] = sentiment.get("polarity", 0)
                    data_point["label"] = sentiment.get("label", "neutral")
                elif request.metric == "content_length":
                    data_point["value"] = len(item.content or "")
                elif request.metric == "processing_time":
                    data_point["value"] = item.processing_time or 0
                
                job_metrics["data_points"].append(data_point)
            
            trend_data.append(job_metrics)
        
        # Generate trend insights
        trend_insights = _analyze_trends(trend_data, request.metric)
        
        return {
            "trend_data": trend_data,
            "insights": trend_insights,
            "metric": request.metric
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/smart-suggestions/{url:path}")
async def get_smart_suggestions(
    url: str,
    current_user: User = Depends(get_current_user)
):
    """Get smart suggestions for scraping a URL"""
    try:
        # Quick analysis to determine best scraping strategy
        scraping_config = {"use_playwright": False, "wait_time": 1}
        quick_scrape = await scraper_service.scrape_url(url, scraping_config)
        
        suggestions = {
            "recommended_config": {},
            "detected_content_type": "",
            "suggested_selectors": [],
            "estimated_complexity": "low"
        }
        
        if quick_scrape.get("content"):
            # Analyze content to provide suggestions
            ai_analysis = await ai_service.analyze_content(
                content=quick_scrape["content"][:2000],  # Limit for quick analysis
                url=url
            )
            
            content_type = ai_analysis.get("content_type", "general")
            suggestions["detected_content_type"] = content_type
            
            # Recommend configuration based on content type
            if content_type == "ecommerce":
                suggestions["recommended_config"] = {
                    "use_playwright": True,
                    "wait_time": 5,
                    "data_type": "prices",
                    "extract_images": True
                }
                suggestions["suggested_selectors"] = [
                    ".price", ".product-price", "[data-price]",
                    ".product-title", "h1", ".product-name"
                ]
                suggestions["estimated_complexity"] = "medium"
            elif content_type == "news":
                suggestions["recommended_config"] = {
                    "use_playwright": False,
                    "data_type": "text",
                    "keywords": ["breaking", "news", "report"]
                }
                suggestions["suggested_selectors"] = [
                    "article", ".article-content", ".news-body",
                    "h1", ".headline", ".title"
                ]
            elif content_type == "blog":
                suggestions["recommended_config"] = {
                    "use_playwright": False,
                    "data_type": "text"
                }
                suggestions["suggested_selectors"] = [
                    ".post-content", ".entry-content", "article",
                    ".blog-post", "main"
                ]
            else:
                suggestions["recommended_config"] = {
                    "use_playwright": True,
                    "wait_time": 3
                }
                suggestions["estimated_complexity"] = "high"
        
        return suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_comparison_insights(results: List[Dict], criteria: List[str]) -> Dict[str, Any]:
    """Generate insights from comparison results"""
    insights = {
        "summary": "",
        "key_differences": [],
        "recommendations": []
    }
    
    if "prices" in criteria:
        prices = []
        for result in results:
            extracted_prices = result["data"].get("extracted_data", {}).get("prices", [])
            if extracted_prices:
                # Extract numeric values from price strings
                numeric_prices = []
                for price in extracted_prices:
                    import re
                    numbers = re.findall(r'[\d.]+', price)
                    if numbers:
                        numeric_prices.append(float(numbers[0]))
                if numeric_prices:
                    prices.append({
                        "url": result["url"],
                        "min_price": min(numeric_prices),
                        "max_price": max(numeric_prices),
                        "avg_price": sum(numeric_prices) / len(numeric_prices)
                    })
        
        if prices:
            cheapest = min(prices, key=lambda x: x["min_price"])
            insights["recommendations"].append(f"Lowest price found at: {cheapest['url']}")
    
    if "sentiment" in criteria:
        sentiments = []
        for result in results:
            sentiment = result["analysis"].get("sentiment", {})
            if sentiment:
                sentiments.append({
                    "url": result["url"],
                    "sentiment": sentiment.get("label", "neutral"),
                    "polarity": sentiment.get("polarity", 0)
                })
        
        if sentiments:
            most_positive = max(sentiments, key=lambda x: x["polarity"])
            insights["recommendations"].append(f"Most positive content: {most_positive['url']}")
    
    return insights

@router.post("/enhanced-analysis")
async def enhanced_analysis(
    request: EnhancedAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform enhanced AI analysis with entity extraction, categorization, and trend analysis"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        
        # First scrape the URL
        scraping_config = {
            "use_playwright": True,
            "wait_time": 3,
            "extract_links": True,
            "extract_images": True
        }
        
        scraped_data = await scraper_service.scrape_url(str(request.url), scraping_config)
        
        if scraped_data.get("error"):
            raise HTTPException(status_code=400, detail=f"Scraping failed: {scraped_data['error']}")
        
        # Perform enhanced AI analysis
        enhanced_analysis = await ai_service_instance.analyze_content_enhanced(
            content=scraped_data.get("content", ""),
            title=scraped_data.get("title", ""),
            url=str(request.url),
            user_id=current_user.id,
            analysis_types=request.analysis_types,
            custom_prompt=request.custom_prompt
        )
        
        return {
            "success": True,
            "url": str(request.url),
            "scraped_data": {
                "title": scraped_data.get("title", ""),
                "content_length": len(scraped_data.get("content", "")),
                "links_found": len(scraped_data.get("links", [])),
                "images_found": len(scraped_data.get("images", []))
            },
            "enhanced_analysis": enhanced_analysis,
            "analysis_types": request.analysis_types
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@router.post("/enhanced-analysis-content")
async def enhanced_analysis_content(
    request: ContentEnhancedAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform enhanced AI analysis on provided content with entity extraction, categorization, and trend analysis"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        
        # Perform enhanced AI analysis on the provided content
        enhanced_analysis = await ai_service_instance.analyze_content_enhanced(
            content=request.content,
            title=request.title,
            url=request.url,
            user_id=current_user.id,
            analysis_types=request.analysis_types,
            custom_prompt=request.custom_prompt
        )
        
        return {
            "success": True,
            "content_length": len(request.content),
            "enhanced_analysis": enhanced_analysis,
            "analysis_types": request.analysis_types
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content enhanced analysis failed: {str(e)}")

@router.post("/targeted-extract", response_model=TargetedExtractResponse)
async def targeted_extract(
    request: TargetedExtractRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Scrape and extract specific content types (text, images, links, emails, phones) and produce focused AI insights."""
    try:
        ai_service_instance = AIService(db=db)
        scraping_config = {
            "use_playwright": request.use_playwright,
            "wait_time": request.wait_time,
            "extract_links": request.extract_links,
            "extract_images": request.extract_images,
            # targeted flags (independent of data_type)
            "extract_emails": request.extract_emails,
            "extract_phones": request.extract_phones,
            "data_type": "text",
        }

        scraped = await scraper_service.scrape_url(str(request.url), scraping_config)
        if scraped.get("error"):
            raise HTTPException(status_code=400, detail=f"Scraping failed: {scraped['error']}")

        content_for_ai = scraped.get("content", "") if request.extract_text else ""

        # Focused AI insights prompt
        custom_prompt = request.custom_prompt or (
            "Provide concise, actionable insights tailored to the user's requested content types. "
            "Summarize the main text (if provided), list key links with purpose, describe image topics (if any), "
            "and extract any contact signals (emails/phones). Return JSON with sections: summary, links, images, contacts, risks."
        )

        ai_result = await ai_service_instance.analyze_content(
            content=content_for_ai,
            title=scraped.get("title", ""),
            url=str(request.url),
            custom_prompt=custom_prompt,
        )

        return {
            "url": str(request.url),
            "title": scraped.get("title"),
            "content_length": len(scraped.get("content", "")),
            "links_found": len(scraped.get("links", [])),
            "images_found": len(scraped.get("images", [])),
            "emails_found": len(scraped.get("extracted_data", {}).get("emails", [])) if request.extract_emails else 0,
            "phones_found": len(scraped.get("extracted_data", {}).get("phones", [])) if request.extract_phones else 0,
            "extracted_data": {
                "links": scraped.get("links", []) if request.extract_links else [],
                "images": scraped.get("images", []) if request.extract_images else [],
                "emails": scraped.get("extracted_data", {}).get("emails", []) if request.extract_emails else [],
                "phones": scraped.get("extracted_data", {}).get("phones", []) if request.extract_phones else [],
            },
            "ai_insights": ai_result or {}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Targeted extract failed: {str(e)}")

@router.get("/user-trends")
async def get_user_trends(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    domain: Optional[str] = None,
    days: int = 30,
    trend_types: Optional[str] = None
):
    """Get comprehensive trend analysis for the current user"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        
        # Parse trend types if provided
        trend_types_list = None
        if trend_types:
            trend_types_list = [t.strip() for t in trend_types.split(",")]
        
        # Get user trends
        trends = await ai_service_instance.get_user_trends(
            user_id=current_user.id,
            domain=domain,
            days=days,
            trend_types=trend_types_list
        )
        
        return {
            "success": True,
            "user_id": current_user.id,
            "domain": domain,
            "days": days,
            "trend_types": trend_types_list,
            "trends": trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User trends analysis failed: {str(e)}")

@router.post("/compare-domains")
async def compare_domains(
    request: DomainComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare trends across different domains"""
    try:
        # Initialize AI service with database session
        ai_service_instance = AIService(db=db)
        
        # Compare domains
        comparison = await ai_service_instance.compare_domains(
            user_id=current_user.id,
            domains=request.domains,
            metric=request.metric,
            days=request.days
        )
        
        return {
            "success": True,
            "user_id": current_user.id,
            "domains": request.domains,
            "metric": request.metric,
            "days": request.days,
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Domain comparison failed: {str(e)}")

@router.get("/entities/{job_id}")
async def get_extracted_entities(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get extracted entities for a specific scraping job"""
    try:
        from models import ScrapedData, ExtractedEntity
        
        # Verify job belongs to user
        scraped_data = db.query(ScrapedData).filter(
            ScrapedData.id == job_id,
            ScrapedData.user_id == current_user.id
        ).first()
        
        if not scraped_data:
            raise HTTPException(status_code=404, detail="Scraping job not found")
        
        # Get entities
        entities = db.query(ExtractedEntity).filter(
            ExtractedEntity.scraped_data_id == job_id
        ).all()
        
        entities_data = []
        for entity in entities:
            entities_data.append({
                "id": entity.id,
                "entity_type": entity.entity_type,
                "entity_text": entity.entity_text,
                "confidence_score": entity.confidence_score,
                "context": entity.context,
                "created_at": entity.created_at.isoformat()
            })
        
        return {
            "success": True,
            "job_id": job_id,
            "entities_count": len(entities_data),
            "entities": entities_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entities: {str(e)}")

@router.get("/categories/{job_id}")
async def get_content_categories(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get content categories for a specific scraping job"""
    try:
        from models import ScrapedData, ContentCategory
        
        # Verify job belongs to user
        scraped_data = db.query(ScrapedData).filter(
            ScrapedData.id == job_id,
            ScrapedData.user_id == current_user.id
        ).first()
        
        if not scraped_data:
            raise HTTPException(status_code=404, detail="Scraping job not found")
        
        # Get categories
        categories = db.query(ContentCategory).filter(
            ContentCategory.scraped_data_id == job_id
        ).all()
        
        categories_data = []
        for category in categories:
            categories_data.append({
                "id": category.id,
                "category": category.category,
                "confidence_score": category.confidence_score,
                "subcategory": category.subcategory,
                "created_at": category.created_at.isoformat()
            })
        
        return {
            "success": True,
            "job_id": job_id,
            "categories_count": len(categories_data),
            "categories": categories_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@router.post("/comprehensive-analysis")
async def comprehensive_content_analysis(
    request: ComprehensiveAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Comprehensive content analysis including summarization, keywords, topics, sentiment, entities, and insights
    """
    try:
        # Initialize AI service with database session
        from services.enhanced_ai_service import EnhancedAIService
        ai_service = EnhancedAIService(db)
        
        # Perform comprehensive analysis
        analysis_result = await ai_service.comprehensive_content_analysis(
            content=request.content,
            title=request.title,
            url=request.url,
            features=request.analysis_features
        )
        
        return {
            "success": True,
            "analysis": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

def _analyze_trends(trend_data: List[Dict], metric: str) -> Dict[str, Any]:
    """Analyze trends in the data"""
    insights = {
        "trend_direction": "stable",
        "key_observations": [],
        "recommendations": []
    }
    
    # Simple trend analysis
    all_values = []
    for job in trend_data:
        for point in job["data_points"]:
            if "value" in point:
                all_values.append(point["value"])
    
    if len(all_values) > 1:
        # Calculate trend
        first_half = all_values[:len(all_values)//2]
        second_half = all_values[len(all_values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_second > avg_first * 1.1:
            insights["trend_direction"] = "increasing"
        elif avg_second < avg_first * 0.9:
            insights["trend_direction"] = "decreasing"
        
        insights["key_observations"].append(
            f"Average {metric} changed from {avg_first:.2f} to {avg_second:.2f}"
        )
    
    return insights