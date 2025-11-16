from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    scraping_jobs = relationship("ScrapingJob", back_populates="user")

class ScrapingJob(Base):
    __tablename__ = "scraping_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    urls = Column(JSON, nullable=False)  # List of URLs
    config = Column(JSON, nullable=False)  # Scraping configuration
    status = Column(String, default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    total_urls = Column(Integer, default=0)
    processed_urls = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="scraping_jobs")
    scraped_data = relationship("ScrapedData", back_populates="job")

class ScrapedData(Base):
    __tablename__ = "scraped_data"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id"), nullable=False)
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    extracted_data = Column(JSON, nullable=True)  # Structured extracted data
    ai_analysis = Column(JSON, nullable=True)  # AI-generated insights
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata
    status = Column(String, default="success")  # success, failed, partial
    error_message = Column(Text, nullable=True)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time = Column(Float, nullable=True)  # Time taken in seconds
    
    # Relationships
    job = relationship("ScrapingJob", back_populates="scraped_data")
    entities = relationship("ExtractedEntity", back_populates="scraped_data")
    categories = relationship("ContentCategory", back_populates="scraped_data")
    ai_insights = relationship("AIInsight", back_populates="scraped_data")

class ScrapingTemplate(Base):
    __tablename__ = "scraping_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    config = Column(JSON, nullable=False)  # Template configuration
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    usage_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User")

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User")

class ExtractedEntity(Base):
    __tablename__ = "extracted_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    scraped_data_id = Column(Integer, ForeignKey("scraped_data.id"), nullable=False)
    entity_type = Column(String, nullable=False)  # PERSON, ORG, GPE, PRODUCT, etc.
    entity_text = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=True)
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    context = Column(Text, nullable=True)  # Surrounding text for context
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    scraped_data = relationship("ScrapedData", back_populates="entities")

class ContentCategory(Base):
    __tablename__ = "content_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    scraped_data_id = Column(Integer, ForeignKey("scraped_data.id"), nullable=False)
    category = Column(String, nullable=False)  # news, tech, finance, politics, etc.
    subcategory = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=False)
    keywords = Column(JSON, nullable=True)  # Keywords that led to this categorization
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    scraped_data = relationship("ScrapedData", back_populates="categories")

class TrendData(Base):
    __tablename__ = "trend_data"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    domain = Column(String, nullable=False)  # Website domain for trend tracking
    trend_type = Column(String, nullable=False)  # price, sentiment, mentions, etc.
    metric_name = Column(String, nullable=False)  # Specific metric being tracked
    metric_value = Column(Float, nullable=False)
    trend_metadata = Column(JSON, nullable=True)  # Additional trend metadata
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")

class AIInsight(Base):
    __tablename__ = "ai_insights"
    
    id = Column(Integer, primary_key=True, index=True)
    scraped_data_id = Column(Integer, ForeignKey("scraped_data.id"), nullable=False)
    insight_type = Column(String, nullable=False)  # summary, sentiment, trend, prediction
    insight_data = Column(JSON, nullable=False)  # Structured insight data
    confidence_score = Column(Float, nullable=True)
    generated_by = Column(String, nullable=False)  # AI model used
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    scraped_data = relationship("ScrapedData", back_populates="ai_insights")