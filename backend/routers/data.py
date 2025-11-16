from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import json
import csv
import io
from datetime import datetime

from database import get_db
from models import User, ScrapingJob, ScrapedData
from routers.auth import get_current_user

router = APIRouter()

@router.get("/export/{job_id}/csv")
async def export_csv(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export scraped data as CSV"""
    # Verify job ownership
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get scraped data
    data = db.query(ScrapedData).filter(ScrapedData.job_id == job_id).all()
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for this job")
    
    # Convert to DataFrame
    records = []
    for item in data:
        record = {
            "url": item.url,
            "title": item.title or "",
            "content": item.content or "",
            "status": item.status,
            "scraped_at": item.scraped_at.isoformat() if item.scraped_at else "",
            "processing_time": item.processing_time or 0
        }
        
        # Add extracted data fields
        if item.extracted_data:
            for key, value in item.extracted_data.items():
                if isinstance(value, list):
                    record[f"extracted_{key}"] = "; ".join(map(str, value))
                else:
                    record[f"extracted_{key}"] = str(value)
        
        # Add AI analysis fields
        if item.ai_analysis:
            record["ai_summary"] = item.ai_analysis.get("summary", "")
            record["ai_sentiment"] = item.ai_analysis.get("sentiment", {}).get("label", "")
            record["ai_keywords"] = "; ".join(item.ai_analysis.get("keywords", []))
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=scraped_data_{job_id}.csv"}
    )

@router.get("/export/{job_id}/json")
async def export_json(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export scraped data as JSON"""
    # Verify job ownership
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get scraped data
    data = db.query(ScrapedData).filter(ScrapedData.job_id == job_id).all()
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for this job")
    
    # Convert to JSON-serializable format
    export_data = {
        "job_info": {
            "id": job.id,
            "name": job.name,
            "created_at": job.created_at.isoformat(),
            "status": job.status,
            "total_urls": job.total_urls,
            "processed_urls": job.processed_urls
        },
        "scraped_data": []
    }
    
    for item in data:
        item_data = {
            "id": item.id,
            "url": item.url,
            "title": item.title,
            "content": item.content,
            "extracted_data": item.extracted_data,
            "ai_analysis": item.ai_analysis,
            "metadata": item.extra_metadata,
            "status": item.status,
            "error_message": item.error_message,
            "scraped_at": item.scraped_at.isoformat() if item.scraped_at else None,
            "processing_time": item.processing_time
        }
        export_data["scraped_data"].append(item_data)
    
    # Create JSON response
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    return StreamingResponse(
        io.BytesIO(json_str.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=scraped_data_{job_id}.json"}
    )

@router.get("/export/{job_id}/excel")
async def export_excel(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export scraped data as Excel"""
    # Verify job ownership
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get scraped data
    data = db.query(ScrapedData).filter(ScrapedData.job_id == job_id).all()
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for this job")
    
    # Convert to DataFrame
    records = []
    for item in data:
        record = {
            "URL": item.url,
            "Title": item.title or "",
            "Content": item.content or "",
            "Status": item.status,
            "Scraped At": item.scraped_at.isoformat() if item.scraped_at else "",
            "Processing Time (s)": item.processing_time or 0
        }
        
        # Add extracted data fields
        if item.extracted_data:
            for key, value in item.extracted_data.items():
                if isinstance(value, list):
                    record[f"Extracted {key.title()}"] = "; ".join(map(str, value))
                else:
                    record[f"Extracted {key.title()}"] = str(value)
        
        # Add AI analysis fields
        if item.ai_analysis:
            record["AI Summary"] = item.ai_analysis.get("summary", "")
            record["AI Sentiment"] = item.ai_analysis.get("sentiment", {}).get("label", "")
            record["AI Keywords"] = "; ".join(item.ai_analysis.get("keywords", []))
            record["Content Type"] = item.ai_analysis.get("content_type", "")
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Create Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Scraped Data', index=False)
        
        # Add job info sheet
        job_info = pd.DataFrame([{
            "Job ID": job.id,
            "Job Name": job.name,
            "Created At": job.created_at.isoformat(),
            "Status": job.status,
            "Total URLs": job.total_urls,
            "Processed URLs": job.processed_urls,
            "Config": json.dumps(job.config, indent=2)
        }])
        job_info.to_excel(writer, sheet_name='Job Info', index=False)
    
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=scraped_data_{job_id}.xlsx"}
    )

@router.get("/stats/{job_id}")
async def get_job_stats(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get statistics for a scraping job"""
    # Verify job ownership
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get scraped data
    data = db.query(ScrapedData).filter(ScrapedData.job_id == job_id).all()
    
    # Calculate statistics
    total_items = len(data)
    successful_items = len([item for item in data if item.status == "success"])
    failed_items = len([item for item in data if item.status == "failed"])
    
    # Processing time stats
    processing_times = [item.processing_time for item in data if item.processing_time]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Content length stats
    content_lengths = [len(item.content or "") for item in data]
    avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    
    # Sentiment distribution (if AI analysis available)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    content_types = {}
    
    for item in data:
        if item.ai_analysis:
            sentiment = item.ai_analysis.get("sentiment", {}).get("label", "neutral")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            content_type = item.ai_analysis.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
    
    return {
        "job_info": {
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "created_at": job.created_at,
            "completed_at": job.completed_at
        },
        "statistics": {
            "total_items": total_items,
            "successful_items": successful_items,
            "failed_items": failed_items,
            "success_rate": (successful_items / total_items * 100) if total_items > 0 else 0,
            "avg_processing_time": round(avg_processing_time, 2),
            "avg_content_length": round(avg_content_length, 0),
            "sentiment_distribution": sentiment_counts,
            "content_type_distribution": content_types
        }
    }

@router.get("/dashboard")
async def get_dashboard_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard data for the user"""
    # Get user's jobs
    jobs = db.query(ScrapingJob).filter(ScrapingJob.user_id == current_user.id).all()
    
    # Calculate overall statistics
    total_jobs = len(jobs)
    completed_jobs = len([job for job in jobs if job.status == "completed"])
    running_jobs = len([job for job in jobs if job.status == "running"])
    failed_jobs = len([job for job in jobs if job.status == "failed"])
    
    # Get total scraped items
    total_scraped_items = db.query(ScrapedData).join(ScrapingJob).filter(
        ScrapingJob.user_id == current_user.id
    ).count()
    
    # Recent activity (last 10 jobs)
    recent_jobs = db.query(ScrapingJob).filter(
        ScrapingJob.user_id == current_user.id
    ).order_by(ScrapingJob.created_at.desc()).limit(10).all()
    
    return {
        "overview": {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "running_jobs": running_jobs,
            "failed_jobs": failed_jobs,
            "total_scraped_items": total_scraped_items
        },
        "recent_jobs": [
            {
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "created_at": job.created_at,
                "processed_urls": job.processed_urls,
                "total_urls": job.total_urls
            }
            for job in recent_jobs
        ]
    }