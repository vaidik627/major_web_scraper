
import os
import json
import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from textblob import TextBlob
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import nltk
from collections import Counter

# OpenAI SDK v1 compatibility (graceful fallback if unavailable)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Import enhanced AI service
try:
    from .enhanced_ai_service import EnhancedAIService
except ImportError:
    EnhancedAIService = None

# Import enhanced summarization service
try:
    from .enhanced_summarization_service import (
        EnhancedSummarizationService, 
        SummaryCustomization, 
        SummaryType, 
        DetailLevel, 
        OutputFormat, 
        FocusArea
    )
except ImportError:
    EnhancedSummarizationService = None
    SummaryCustomization = None

# Import new enhanced AI integration service
try:
    from .enhanced_ai_integration import EnhancedAIIntegrationService
except ImportError:
    EnhancedAIIntegrationService = None

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, db: Session = None):
        self.db = db
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.openai_api_key and OpenAI:
            # Initialize modern OpenAI client; avoid deprecated interfaces
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception:
                self.client = None
        
        # Initialize enhanced AI service if available
        self.enhanced_service = None
        if EnhancedAIService and self.db:
            try:
                self.enhanced_service = EnhancedAIService(db=self.db, openai_api_key=self.openai_api_key)
            except Exception as e:
                print(f"Could not initialize enhanced AI service: {e}")
                self.enhanced_service = None
        
        # Initialize enhanced summarization service
        self.enhanced_summarization_service = None
        if EnhancedSummarizationService:
            try:
                self.enhanced_summarization_service = EnhancedSummarizationService()
                logging.info("Enhanced summarization service initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize enhanced summarization service: {e}")
                self.enhanced_summarization_service = None
        else:
            logging.warning("Enhanced summarization service not available")
        
        # Initialize new enhanced AI integration service
        self.enhanced_integration_service = None
        if EnhancedAIIntegrationService:
            try:
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                self.enhanced_integration_service = EnhancedAIIntegrationService(
                    openai_api_key=self.openai_api_key,
                    anthropic_api_key=anthropic_api_key
                )
                logging.info("Enhanced AI integration service initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize enhanced AI integration service: {e}")
                self.enhanced_integration_service = None
        else:
            logging.warning("Enhanced AI integration service not available")
    
    async def analyze_content(
        self, 
        content: str, 
        title: str = "", 
        url: str = "",
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive AI analysis of scraped content"""
        
        # Preprocess content to remove boilerplate/navigation and normalize text
        preprocessed = self._preprocess_content(content, url)

        analysis = {
            "summary": "",
            "sentiment": {},
            "keywords": [],
            "entities": {},
            "smart_extraction": {},
            "content_type": "",
            "language": "en",
            "readability_score": 0
        }
        
        try:
            # Basic text analysis
            analysis["sentiment"] = self._analyze_sentiment(preprocessed)
            analysis["keywords"] = self._extract_keywords(preprocessed)
            analysis["language"] = self._detect_language(preprocessed)
            analysis["readability_score"] = self._calculate_readability(preprocessed)
            analysis["content_type"] = self._classify_content_type(preprocessed, title, url)
            
            # AI-powered analysis if client is available
            if self.client is not None:
                ai_analysis = await self._openai_analysis(preprocessed, title, url, custom_prompt)
                analysis.update(ai_analysis)
            else:
                # Fallback to rule-based analysis
                summary, bullets = self._summarize_with_chunks(preprocessed, title)
                analysis["summary"] = summary
                analysis["bullets"] = bullets
                analysis["entities"] = self._extract_entities(content)
                analysis["smart_extraction"] = self._smart_extract_fallback(preprocessed, url)
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis

    async def analyze_content_with_enhanced_ai(
        self,
        content: str,
        title: str = "",
        url: str = "",
        user_id: str = None,
        max_length: int = 500,
        enable_personalization: bool = True
    ) -> Dict[str, Any]:
        """Analyze content using the new enhanced AI integration service"""
        
        if not self.enhanced_integration_service:
            # Fallback to regular analysis
            return await self.analyze_content(content, title, url)
        
        try:
            # Use enhanced AI integration service
            result = await self.enhanced_integration_service.analyze_content_enhanced(
                content=content,
                title=title,
                url=url,
                user_id=user_id,
                max_length=max_length,
                enable_personalization=enable_personalization
            )
            
            # Convert to expected format
            return {
                "summary": result.summary,
                "domain": result.domain,
                "confidence": result.confidence,
                "technical_elements": result.technical_elements,
                "related_technologies": result.related_technologies,
                "learning_insights": result.learning_insights,
                "personalized_for_user": result.personalized_for_user,
                "metadata": result.metadata,
                "enhanced_analysis": True
            }
            
        except Exception as e:
            logger.error(f"Enhanced AI analysis failed: {e}")
            # Fallback to regular analysis
            return await self.analyze_content(content, title, url)

    async def collect_user_feedback(
        self,
        user_id: str,
        summary_id: str,
        rating: int,
        feedback_text: str = "",
        domain: str = "",
        technology: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Collect user feedback for continuous improvement"""
        
        if self.enhanced_integration_service:
            await self.enhanced_integration_service.collect_user_feedback(
                user_id=user_id,
                summary_id=summary_id,
                rating=rating,
                feedback_text=feedback_text,
                domain=domain,
                technology=technology,
                metadata=metadata
            )

    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get personalized insights for a user"""
        
        if self.enhanced_integration_service:
            return await self.enhanced_integration_service.get_user_insights(user_id)
        return {}

    async def get_technology_ecosystem(self, technology: str) -> Dict[str, Any]:
        """Get technology ecosystem information"""
        
        if self.enhanced_integration_service:
            return await self.enhanced_integration_service.get_technology_ecosystem(technology)
        return {}

    async def find_learning_path(
        self,
        start_technology: str,
        target_technology: str,
        max_steps: int = 5
    ) -> List[Dict[str, Any]]:
        """Find learning path between technologies"""
        
        if self.enhanced_integration_service:
            return await self.enhanced_integration_service.find_learning_path(
                start_technology, target_technology, max_steps
            )
        return []

    async def suggest_technologies(
        self,
        current_technologies: List[str],
        target_domain: str = None,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Suggest technologies based on current stack"""
        
        if self.enhanced_integration_service:
            return await self.enhanced_integration_service.suggest_technologies(
                current_technologies, target_domain, max_suggestions
            )
        return []

    async def analyze_content_enhanced(
        self, 
        content: str, 
        title: str = "", 
        url: str = "",
        scraped_data = None,
        custom_prompt: Optional[str] = None,
        user_id: int = None,
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """Enhanced AI analysis with entity extraction, categorization, and trend analysis"""
        
        # Start with basic analysis
        analysis = await self.analyze_content(content, title, url, custom_prompt)
        
        # Add enhanced features if available
        if self.enhanced_service:
            try:
                # Get enhanced analysis
                enhanced_analysis = await self.enhanced_service.enhanced_analyze_content(
                    content=content,
                    title=title,
                    url=url,
                    scraped_data=scraped_data
                )
                
                # Merge enhanced features
                analysis.update({
                    "advanced_entities": enhanced_analysis.get("advanced_entities", {}),
                    "content_category": enhanced_analysis.get("content_category", {}),
                    "trend_indicators": enhanced_analysis.get("trend_indicators", {}),
                    "ai_insights": enhanced_analysis.get("ai_insights", []),
                    "content_classification": enhanced_analysis.get("content_classification", {}),
                    "enhanced_keywords": enhanced_analysis.get("enhanced_keywords", []),
                    "content_quality": enhanced_analysis.get("content_quality", {}),
                    "data_enrichment": {
                        "entity_extraction": enhanced_analysis.get("advanced_entities", {}),
                        "categorization": enhanced_analysis.get("content_category", {}),
                        "trend_analysis": enhanced_analysis.get("trend_indicators", {})
                    }
                })
                
                # Store trend data if scraped_data is provided
                if scraped_data:
                    await self.enhanced_service.store_analysis_trends(scraped_data, enhanced_analysis)
                
            except Exception as e:
                print(f"Error in enhanced analysis: {e}")
                analysis["enhanced_error"] = str(e)
        
        return analysis

    async def get_user_trends(self, user_id: int, domain: str = None, days: int = 30, trend_types: str = None) -> Dict[str, Any]:
        """Get trend analysis for a user"""
        if self.enhanced_service:
            return await self.enhanced_service.get_user_trends(user_id, domain, days)
        else:
            return {"error": "Enhanced AI service not available"}

    async def compare_domains(self, user_id: int, domains: List[str], metric: str = "sentiment", days: int = 30) -> Dict[str, Any]:
        """Compare trends across different domains"""
        if self.enhanced_service and self.enhanced_service.trend_service:
            return await self.enhanced_service.trend_service.compare_domains(user_id, domains, metric, days)
        else:
            return {"error": "Trend analysis service not available"}
    
    async def _openai_analysis(
        self, 
        content: str, 
        title: str, 
        url: str,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use OpenAI (v1 SDK) for advanced, readable summaries and extraction with enhanced accuracy."""

        # Enhanced content preprocessing for better accuracy
        processed_content = self._enhanced_content_preprocessing(content, title, url)
        
        # Use larger context window for better accuracy
        max_content_length = 12000  # Increased from 8000
        if len(processed_content) > max_content_length:
            # Smart truncation preserving important sections
            processed_content = self._smart_truncate_content(processed_content, max_content_length)

        prompt = custom_prompt or self._build_enhanced_readable_summary_prompt(processed_content, title, url)

        try:
            # Use to_thread to avoid blocking the event loop with sync client
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # Upgraded to GPT-4o for better accuracy
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert content analyst and summarization specialist with advanced NLP capabilities. "
                            "Your task is to create highly accurate, comprehensive summaries that capture the essence and "
                            "nuances of the content. Focus on factual accuracy, logical coherence, and actionable insights. "
                            "Use advanced reasoning to identify the most important information and present it in a clear, "
                            "professional manner. Always output valid JSON with precise, well-structured data."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Lower temperature for more consistent, accurate results
                max_tokens=1500,  # Increased token limit for more comprehensive summaries
                top_p=0.9,  # Added for better quality control
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1,  # Encourage diverse content
            )

            result_text = response.choices[0].message.content
            try:
                parsed = json.loads(result_text)
            except json.JSONDecodeError:
                # Enhanced fallback parsing
                parsed = self._parse_fallback_response(result_text)

            # Enhanced validation and enrichment
            parsed = self._enhance_analysis_result(parsed, content, title, url)
            parsed.setdefault("model", os.getenv("OPENAI_MODEL", "gpt-4o"))
            parsed.setdefault("accuracy_score", self._calculate_accuracy_score(parsed, content))
            
            return parsed

        except Exception as e:
            # Enhanced fallback with better error handling
            logging.error(f"OpenAI analysis failed: {e}")
            summary, bullets = self._generate_user_friendly_summary(content, title)
            return {
                "summary": summary,
                "bullets": bullets,
                "key_points": self._extract_key_points_fallback(content),
                "smart_extraction": self._smart_extract_fallback(content, url),
                "ai_error": str(e),
                "fallback_mode": True,
                "accuracy_score": 0.6  # Lower confidence for fallback
            }
    
    def _enhanced_content_preprocessing(self, content: str, title: str, url: str) -> str:
        """Enhanced content preprocessing for better AI analysis accuracy."""
        
        # Remove common web noise
        noise_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<style[^>]*>.*?</style>',
            r'<nav[^>]*>.*?</nav>',
            r'<footer[^>]*>.*?</footer>',
            r'<header[^>]*>.*?</header>',
            r'<aside[^>]*>.*?</aside>',
            r'<div[^>]*class="[^"]*(?:ad|advertisement|banner|popup)[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*id="[^"]*(?:ad|advertisement|banner|popup)[^"]*"[^>]*>.*?</div>',
        ]
        
        processed = content
        for pattern in noise_patterns:
            processed = re.sub(pattern, '', processed, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up HTML tags but preserve structure
        processed = re.sub(r'<[^>]+>', ' ', processed)
        processed = re.sub(r'\s+', ' ', processed)
        processed = processed.strip()
        
        # Add context from title and URL
        context_info = f"Title: {title}\nURL: {url}\n\nContent: {processed}"
        
        return context_info
    
    def _smart_truncate_content(self, content: str, max_length: int) -> str:
        """Smart content truncation that preserves important sections."""
        
        if len(content) <= max_length:
            return content
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Prioritize sentences with important keywords
        important_keywords = ['important', 'key', 'main', 'primary', 'significant', 'critical', 'essential']
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            # Higher score for sentences with important keywords
            for keyword in important_keywords:
                if keyword.lower() in sentence.lower():
                    score += 2
            
            # Higher score for sentences near the beginning
            if i < len(sentences) * 0.3:
                score += 1
            
            # Higher score for longer sentences (more informative)
            if len(sentence) > 50:
                score += 1
                
            scored_sentences.append((score, i, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        result_sentences = []
        current_length = 0
        
        for score, original_index, sentence in scored_sentences:
            if current_length + len(sentence) + 1 <= max_length - 100:  # Leave room for "..."
                result_sentences.append((original_index, sentence))
                current_length += len(sentence) + 1
        
        # Sort back to original order
        result_sentences.sort(key=lambda x: x[0])
        result = '. '.join([s[1] for s in result_sentences])
        
        if len(result) < len(content):
            result += "..."
            
        return result
    
    def _build_enhanced_readable_summary_prompt(self, content: str, title: str, url: str) -> str:
        """Enhanced prompt for producing professional-grade, highly accurate summaries."""

        return (
            "You are a senior content analyst and professional summarization specialist with expertise in business intelligence, "
            "academic research, and technical documentation. Your task is to analyze the provided web content and create a "
            "professional-grade summary that meets enterprise standards for accuracy, completeness, and actionable insights.\n\n"
            
            "PROFESSIONAL ANALYSIS REQUIREMENTS:\n"
            "- Conduct comprehensive semantic analysis with deep contextual understanding\n"
            "- Perform factual verification and cross-reference information for accuracy\n"
            "- Identify hierarchical information structure (main topics, subtopics, supporting details)\n"
            "- Extract quantitative data, statistics, metrics, and concrete examples\n"
            "- Analyze tone, sentiment, credibility, and underlying business implications\n"
            "- Identify strategic insights, market trends, and competitive intelligence\n"
            "- Distinguish between primary content, supporting evidence, and contextual information\n"
            "- Evaluate information quality, source credibility, and potential biases\n"
            "- Extract actionable recommendations, next steps, and strategic implications\n\n"
            
            "PROFESSIONAL OUTPUT STANDARDS:\n"
            "- Use precise, professional business language with proper terminology\n"
            "- Ensure factual accuracy with zero speculation or assumptions\n"
            "- Maintain logical coherence, flow, and professional structure\n"
            "- Present information in order of strategic importance and relevance\n"
            "- Include specific data points, percentages, dates, and concrete examples\n"
            "- Maintain objectivity while highlighting critical business insights\n"
            "- Use active voice, clear sentence structure, and professional formatting\n"
            "- Ensure compliance with business communication standards\n\n"
            
            "Return ONLY valid JSON with these exact keys:\n"
            "{\n"
            "  \"summary\": \"A comprehensive 5-7 sentence professional overview that captures the main purpose, key findings, strategic implications, and business relevance\",\n"
            "  \"executive_summary\": \"A 2-3 sentence executive-level summary for C-level decision makers\",\n"
            "  \"bullets\": [\"8-10 specific, actionable bullet points with concrete details, metrics, and strategic insights\"],\n"
            "  \"key_points\": [\"6-8 critical facts, statistics, trends, or strategic takeaways that executives must know\"],\n"
            "  \"insights\": [\"3-4 deeper strategic insights, market implications, or competitive intelligence\"],\n"
            "  \"actionable_items\": [\"Specific strategic actions, recommendations, or business decisions mentioned\"],\n"
            "  \"quantitative_data\": [\"Specific numbers, percentages, dates, and metrics extracted from the content\"],\n"
            "  \"stakeholders\": [\"Key people, organizations, or entities mentioned and their relevance\"],\n"
            "  \"business_impact\": \"Assessment of potential business impact and strategic importance\",\n"
            "  \"confidence_score\": 0.95,\n"
            "  \"professional_grade\": true\n"
            "}\n\n"
            
            f"CONTENT TO ANALYZE:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content[:10000]}{'...' if len(content) > 10000 else ''}"
        )
    
    def _parse_fallback_response(self, result_text: str) -> Dict[str, Any]:
        """Enhanced fallback parsing for non-JSON responses."""
        
        # Try to extract structured information from text
        lines = result_text.split('\n')
        summary_lines = []
        bullets = []
        key_points = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'summary' in line.lower() and ':' in line:
                current_section = 'summary'
                continue
            elif 'bullet' in line.lower() or 'point' in line.lower():
                current_section = 'bullets'
                continue
            elif 'key' in line.lower() and 'point' in line.lower():
                current_section = 'key_points'
                continue
            
            # Add content to appropriate section
            if current_section == 'summary':
                summary_lines.append(line)
            elif current_section == 'bullets' and (line.startswith('-') or line.startswith('•')):
                bullets.append(line.lstrip('-• '))
            elif current_section == 'key_points' and (line.startswith('-') or line.startswith('•')):
                key_points.append(line.lstrip('-• '))
        
        return {
            "summary": ' '.join(summary_lines) if summary_lines else result_text[:500],
            "bullets": bullets,
            "key_points": key_points,
            "insights": [],
            "actionable_items": [],
            "confidence_score": 0.7
        }
    
    def _enhance_analysis_result(self, parsed: Dict[str, Any], content: str, title: str, url: str) -> Dict[str, Any]:
        """Enhance and validate the analysis result."""
        
        # Ensure all required fields exist with defaults
        parsed.setdefault("summary", "")
        parsed.setdefault("bullets", [])
        parsed.setdefault("key_points", [])
        parsed.setdefault("insights", [])
        parsed.setdefault("actionable_items", [])
        parsed.setdefault("confidence_score", 0.8)
        
        # Validate and clean summary
        if parsed["summary"]:
            parsed["summary"] = self._clean_text(parsed["summary"])
        
        # Validate and clean bullets
        if parsed["bullets"]:
            parsed["bullets"] = [self._clean_text(bullet) for bullet in parsed["bullets"] if bullet.strip()]
        
        # Validate and clean key points
        if parsed["key_points"]:
            parsed["key_points"] = [self._clean_text(point) for point in parsed["key_points"] if point.strip()]
        
        # Add metadata
        parsed["metadata"] = {
            "content_length": len(content),
            "title": title,
            "url": url,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "enhanced_analysis": True
        }
        
        return parsed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = re.sub(r'^["\']|["\']$', '', text)  # Remove quotes at start/end
        text = re.sub(r'^[-•]\s*', '', text)  # Remove bullet points
        
        return text
    
    def _calculate_accuracy_score(self, parsed: Dict[str, Any], content: str) -> float:
        """Calculate accuracy score based on analysis quality."""
        
        score = 0.5  # Base score
        
        # Check if summary exists and has reasonable length
        if parsed.get("summary") and len(parsed["summary"]) > 50:
            score += 0.2
        
        # Check if bullets exist and are meaningful
        bullets = parsed.get("bullets", [])
        if bullets and len(bullets) >= 3:
            score += 0.15
        
        # Check if key points exist
        key_points = parsed.get("key_points", [])
        if key_points and len(key_points) >= 2:
            score += 0.1
        
        # Check confidence score from AI
        ai_confidence = parsed.get("confidence_score", 0.8)
        score = (score + ai_confidence) / 2
        
        return min(score, 1.0)
    
    def _extract_key_points_fallback(self, content: str) -> List[str]:
        """Extract key points using fallback methods."""
        
        # Use TextBlob for basic extraction
        blob = TextBlob(content)
        sentences = blob.sentences
        
        # Score sentences by importance
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_text = str(sentence)
            
            # Higher score for sentences with numbers, important words
            if re.search(r'\d+', sentence_text):
                score += 1
            if any(word in sentence_text.lower() for word in ['important', 'key', 'main', 'primary']):
                score += 2
            if len(sentence_text) > 50:  # Longer sentences often more informative
                score += 1
                
            scored_sentences.append((score, sentence_text))
        
        # Return top 3-5 sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored_sentences[:5]]
    
    def _build_readable_summary_prompt(self, content: str, title: str, url: str) -> str:
        """Enhanced prompt for producing well-structured, grammatically correct summaries."""

        return (
            "You are an expert content analyst creating a professional summary for business users. "
            "Your task is to analyze the provided web content and create a clear, well-structured summary.\n\n"
            
            "REQUIREMENTS:\n"
            "- Use proper grammar, punctuation, and professional language\n"
            "- Write in complete, well-formed sentences\n"
            "- Ensure logical flow and coherent structure\n"
            "- Focus on the most important and actionable information\n"
            "- Use active voice when possible\n"
            "- Avoid technical jargon unless necessary\n\n"
            
            "Return ONLY valid JSON with these exact keys:\n"
            "{\n"
            "  \"summary\": \"A comprehensive 3-4 sentence overview that captures the main purpose and key information\",\n"
            "  \"bullets\": [\"5-7 specific, actionable bullet points highlighting important details\"],\n"
            "  \"key_points\": [\"3-5 critical facts or takeaways that users should remember\"]\n"
            "}\n\n"
            
            f"CONTENT TO ANALYZE:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content[:3000]}{'...' if len(content) > 3000 else ''}"
        )
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(content)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = "positive"
            elif polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "label": sentiment_label,
                "confidence": abs(polarity)
            }
        except:
            return {"label": "neutral", "polarity": 0, "subjectivity": 0, "confidence": 0}
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using simple frequency analysis"""
        try:
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Remove common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
                'those', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'can', 'are', 'is', 'am'
            }
            
            # Filter and count
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:max_keywords]]
            
        except:
            return []
    
    def _detect_language(self, content: str) -> str:
        """Detect content language"""
        try:
            blob = TextBlob(content)
            return blob.detect_language()
        except:
            return "en"
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate basic readability score"""
        try:
            sentences = content.split('.')
            words = content.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability score (lower is easier to read)
            score = (avg_sentence_length * 0.5) + (avg_word_length * 2)
            return min(100, max(0, 100 - score))
            
        except:
            return 50
    
    def _classify_content_type(self, content: str, title: str, url: str) -> str:
        """Classify the type of content"""
        content_lower = content.lower()
        title_lower = title.lower()
        url_lower = url.lower()
        
        # E-commerce indicators
        if any(word in content_lower for word in ['price', 'buy', 'cart', 'product', 'order', 'shipping']):
            return "ecommerce"
        
        # News indicators
        if any(word in content_lower for word in ['breaking', 'news', 'reported', 'according to', 'sources']):
            return "news"
        
        # Blog indicators
        if any(word in url_lower for word in ['blog', 'post', 'article']) or 'posted' in content_lower:
            return "blog"
        
        # Business/service indicators
        if any(word in content_lower for word in ['contact', 'service', 'company', 'business', 'about us']):
            return "business"
        
        # Research/academic indicators
        if any(word in content_lower for word in ['research', 'study', 'analysis', 'methodology', 'conclusion']):
            return "research"
        
        return "general"
    
    def _generate_user_friendly_summary(self, content: str, title: str = "", max_sentences: int = 4) -> (str, List[str]):
        """Enhanced extractive summary with improved structure and grammar (fallback without AI)."""
        try:
            # Normalize whitespace and split sentences
            text = re.sub(r"\s+", " ", content).strip()
            raw_sentences = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 25 and len(s.strip()) < 200]

            if not sentences:
                return (self._create_fallback_summary(content, title), [])

            # Enhanced keyword-based scoring with improved heuristics
            keywords = set(self._extract_keywords(text, 25))
            scores: Dict[int, float] = {}
            
            for i, s in enumerate(sentences):
                s_lower = s.lower()
                # Keyword relevance score
                kw_score = sum(s_lower.count(k) for k in keywords) * 2
                
                # Length scoring (prefer medium-length sentences)
                length = len(s.split())
                if 10 <= length <= 25:
                    length_score = 1.0
                elif 8 <= length <= 30:
                    length_score = 0.7
                else:
                    length_score = 0.3
                
                # Position bonus (first few sentences are often important)
                position_bonus = max(0, 1.0 - (i * 0.1))
                
                # Grammar and structure bonus
                structure_bonus = 0.5 if self._has_good_structure(s) else 0
                
                scores[i] = kw_score + length_score + position_bonus + structure_bonus

            # Select top sentences, maintain logical order
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            top_indices = sorted([i for i, _ in top])
            summary_sents = [sentences[i] for i in top_indices]

            # Create coherent summary with proper transitions
            summary_text = self._create_coherent_summary(summary_sents, title)

            # Build structured bullet points
            bullets = self._create_structured_bullets(sentences, keywords)

            return (summary_text, self._clean_bullets(bullets))
        except Exception:
            return (self._create_fallback_summary(content, title), [])

    def _has_good_structure(self, sentence: str) -> bool:
        """Check if sentence has good grammatical structure."""
        # Basic checks for well-formed sentences
        sentence = sentence.strip()
        if not sentence:
            return False
        
        # Should start with capital letter
        if not sentence[0].isupper():
            return False
        
        # Should end with proper punctuation
        if not sentence.endswith(('.', '!', '?')):
            return False
        
        # Should have reasonable word count
        word_count = len(sentence.split())
        if word_count < 5 or word_count > 40:
            return False
        
        # Should not be all caps or have excessive punctuation
        if sentence.isupper() or sentence.count('!') > 2:
            return False
        
        return True

    def _create_coherent_summary(self, sentences: List[str], title: str) -> str:
        """Create a coherent summary with proper flow and transitions."""
        if not sentences:
            return "No meaningful content could be extracted from this page."
        
        # Clean and improve sentences
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = sentence.strip()
            # Normalize spaces and commas
            cleaned = re.sub(r"\s+", " ", cleaned)
            cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
            # Ensure proper capitalization
            if cleaned and not cleaned[0].isupper():
                cleaned = cleaned[0].upper() + cleaned[1:]
            # Ensure proper ending punctuation
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
            # Avoid dangling conjunctions at the end
            cleaned = re.sub(r"\b(and|but|or)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned_sentences.append(cleaned)
        
        # Join sentences with proper spacing
        summary = ' '.join(cleaned_sentences)
        
        # Add title context if available
        if title and title.strip():
            title_clean = title.strip().rstrip(':.')
            if not summary.lower().startswith(title_clean.lower()[:20]):
                summary = f"This content about {title_clean} shows that {summary[0].lower() + summary[1:]}"
        
        return summary

    def _create_structured_bullets(self, sentences: List[str], keywords: set) -> List[str]:
        """Create well-structured bullet points from content."""
        bullets: List[str] = []
        number_pattern = re.compile(r"\b\d+[\d,./-]*\b")
        
        # Prioritize sentences with numbers, dates, or key information
        for s in sentences:
            if len(bullets) >= 7:
                break
            
            # Look for informative content
            has_numbers = bool(number_pattern.search(s))
            has_keywords = any(k in s.lower() for k in list(keywords)[:15])
            is_good_length = 20 <= len(s) <= 150
            
            if (has_numbers or has_keywords) and is_good_length and self._has_good_structure(s):
                # Clean and format bullet point
                cleaned = s.strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
                if cleaned.endswith('.'):
                    cleaned = cleaned[:-1]  # Remove period for bullet points
                
                # Ensure it starts with capital letter
                if cleaned and not cleaned[0].isupper():
                    cleaned = cleaned[0].upper() + cleaned[1:]
                # Keep to a single idea: cut at subordinate clause keywords or first comma
                cleaned = re.split(r"\b(which|that|because|however|although|whereas)\b", cleaned, maxsplit=1)[0].strip()
                cleaned = cleaned.split(',')[0].strip()
                
                bullets.append(cleaned)
        
        # If we don't have enough bullets, add some well-formed sentences
        if len(bullets) < 3:
            for s in sentences:
                if len(bullets) >= 5:
                    break
                if self._has_good_structure(s) and s not in [b + '.' for b in bullets]:
                    cleaned = s.strip().rstrip('.')
                    if cleaned and not cleaned[0].isupper():
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    bullets.append(cleaned)
        
        return bullets

    def _create_fallback_summary(self, content: str, title: str) -> str:
        """Create a basic fallback summary when extraction fails."""
        if not content:
            return "No content available for analysis."
        
        # Take first meaningful portion
        text = re.sub(r"\s+", " ", content).strip()
        if len(text) > 300:
            # Find a good breaking point
            break_point = text.rfind('.', 0, 300)
            if break_point > 100:
                text = text[:break_point + 1]
            else:
                text = text[:300] + "..."
        
        # Add title context
        if title and title.strip():
            title_clean = title.strip().rstrip(':.')
            return f"This page about {title_clean} contains the following information: {text}"
        
        return f"This page contains: {text}"

    def _preprocess_content(self, content: str, url: str = "") -> str:
        """Remove common boilerplate, navigation, and template text to improve summaries."""
        text = content or ""
        text = re.sub(r"\s+", " ", text)

        # Remove cookie consent / subscribe / sign-in boilerplate
        patterns = [
            r"cookie(s)? (policy|consent|settings)",
            r"subscribe now|sign in|log in|register|accept all",
            r"privacy policy|terms of service|use of cookies",
            r"^\s*(menu|navigation)\b",
            r"toggle the table of contents|table of contents",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

        # Wikipedia-specific cleanup if applicable
        if "wikipedia.org" in (url or ""):
            wiki_noise = [
                r"article talk", r"watch edit", r"language", r"learn how and when to remove this message",
                r"find sources", r"help improve this article", r"citation needed"
            ]
            for wn in wiki_noise:
                text = re.sub(wn, "", text, flags=re.IGNORECASE)

        # Collapse multiple spaces
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _chunk_sentences(self, text: str, max_chars: int = 1500) -> List[str]:
        """Chunk text by sentences to roughly target max_chars per chunk."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) + 1 <= max_chars:
                current = (current + " " + s).strip()
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)
        return chunks

    def _summarize_with_chunks(self, text: str, title: str = "") -> (str, List[str]):
        """Summarize large text by chunking then merging summaries and bullets."""
        chunks = self._chunk_sentences(text)
        if len(chunks) <= 1:
            return self._generate_user_friendly_summary(text, title)

        merged_summary_parts: List[str] = []
        merged_bullets: List[str] = []
        for ch in chunks:
            s, b = self._generate_user_friendly_summary(ch, "", max_sentences=3)
            if s:
                merged_summary_parts.append(s)
            merged_bullets.extend(b)

        # Compose overall summary: take top parts and keep concise
        overall_summary = "; ".join(merged_summary_parts[:4])
        cleaned_bullets = self._clean_bullets(merged_bullets)[:7]
        if title:
            overall_summary = f"{title.strip()}: " + overall_summary
        return overall_summary, cleaned_bullets

    def _clean_bullets(self, bullets: List[str]) -> List[str]:
        """Enhanced bullet cleaning with better formatting and structure."""
        seen = set()
        cleaned: List[str] = []
        
        # Expanded noise patterns for better filtering
        noise_patterns = [
            r"^\s*(menu|navigation|header|footer)\b",
            r"cookie|privacy|subscribe|sign in|log in|register",
            r"article talk|watch edit|language|citation needed",
            r"toggle the table of contents|table of contents",
            r"redirects here|see .*\.|click here|read more",
            r"^\s*(home|about|contact|search)\s*$",
            r"^\s*\d+\s*$",  # Just numbers
            r"^\s*[^\w\s]*\s*$",  # Just punctuation
        ]
        
        for b in bullets:
            if not b:
                continue
                
            candidate = b.strip()
            
            # Skip very short or very long bullets
            if len(candidate) < 10 or len(candidate) > 200:
                continue
            
            # Filter noise patterns
            if any(re.search(p, candidate, flags=re.IGNORECASE) for p in noise_patterns):
                continue
            
            # Ensure proper formatting
            candidate = self._format_bullet_point(candidate)
            
            # Skip if formatting failed
            if not candidate:
                continue
            
            # Deduplicate (case-insensitive, considering similar content)
            key = re.sub(r'[^\w\s]', '', candidate.lower())
            if key in seen or len(key) < 5:
                continue
            
            # Check for substantial overlap with existing bullets
            is_duplicate = False
            for existing in cleaned:
                existing_key = re.sub(r'[^\w\s]', '', existing.lower())
                if self._calculate_similarity(key, existing_key) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(key)
                cleaned.append(candidate)
        
        return cleaned[:7]  # Limit to 7 bullets max

    def _format_bullet_point(self, bullet: str) -> str:
        """Format a bullet point for better readability."""
        if not bullet:
            return ""
        
        # Clean up the bullet
        formatted = bullet.strip()
        
        # Remove bullet markers if present
        formatted = re.sub(r'^[-•*]\s*', '', formatted)
        
        # Ensure proper capitalization
        if formatted and not formatted[0].isupper():
            formatted = formatted[0].upper() + formatted[1:]
        
        # Remove trailing periods for bullet points
        if formatted.endswith('.') and not formatted.endswith('...'):
            formatted = formatted[:-1]
        
        # Keep concise by removing subordinate clauses and trimming after first comma
        formatted = re.split(r"\b(which|that|because|however|although|whereas)\b", formatted, maxsplit=1)[0].strip()
        formatted = formatted.split(',')[0].strip()
        
        # Ensure reasonable length
        if len(formatted) > 150:
            # Try to break at a natural point
            break_point = formatted.rfind(',', 0, 150)
            if break_point < 100:
                break_point = formatted.rfind(' ', 0, 150)
            if break_point > 50:
                formatted = formatted[:break_point] + "..."
            else:
                formatted = formatted[:150] + "..."
        
        # Basic validation
        if len(formatted) < 10 or not any(c.isalpha() for c in formatted):
            return ""
        
        return formatted

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities using pattern matching"""
        entities = {
            "emails": [],
            "phones": [],
            "urls": [],
            "dates": [],
            "prices": [],
            "organizations": [],
            "locations": []
        }
        
        try:
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities["emails"] = list(set(re.findall(email_pattern, content)))
            
            # Phone pattern
            phone_pattern = r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
            entities["phones"] = list(set(re.findall(phone_pattern, content)))
            
            # URL pattern
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            entities["urls"] = list(set(re.findall(url_pattern, content)))
            
            # Price pattern
            price_pattern = r'\$[\d,]+\.?\d*|€[\d,]+\.?\d*|£[\d,]+\.?\d*'
            entities["prices"] = list(set(re.findall(price_pattern, content)))
            
            # Date pattern (basic)
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            entities["dates"] = list(set(re.findall(date_pattern, content)))
            
        except:
            pass
        
        return entities
    
    def _smart_extract_fallback(self, content: str, url: str) -> Dict[str, Any]:
        """Fallback smart extraction without AI"""
        extraction = {}
        
        content_type = self._classify_content_type(content, "", url)
        
        if content_type == "ecommerce":
            extraction.update({
                "prices": self._extract_entities(content)["prices"],
                "product_indicators": self._find_product_info(content)
            })
        elif content_type == "news":
            extraction.update({
                "headline": self._extract_headline(content),
                "key_facts": self._extract_key_facts(content)
            })
        elif content_type == "business":
            extraction.update({
                "contact_info": self._extract_contact_info(content),
                "services": self._extract_services(content)
            })
        
        return extraction
    
    def generate_enhanced_summary(
        self, 
        content: str, 
        title: str = "", 
        url: str = "",
        summary_type: str = "balanced",
        detail_level: str = "medium",
        output_format: str = "mixed",
        focus_areas: List[str] = None,
        highlight_relevant_text: bool = True,
        include_keywords: bool = True,
        max_length: Optional[int] = None,
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate enhanced summary with customization options and text highlighting.
        
        Args:
            content: The content to summarize
            title: Title of the content
            url: URL of the content
            summary_type: Type of summary (brief, balanced, detailed, executive)
            detail_level: Level of detail (short, medium, long, comprehensive)
            output_format: Format of output (paragraph, bullet_points, mixed)
            focus_areas: List of focus areas (main_content, key_facts, etc.)
            highlight_relevant_text: Whether to highlight relevant text
            include_keywords: Whether to include keywords
            max_length: Maximum length of summary
            user_query: User's specific query or interest
            
        Returns:
            Dictionary containing enhanced summary with highlights and metadata
        """
        
        try:
            # Use enhanced summarization service if available
            if self.enhanced_summarization_service and SummaryCustomization:
                # Convert string parameters to enums
                summary_type_enum = getattr(SummaryType, summary_type.upper(), SummaryType.BALANCED)
                detail_level_enum = getattr(DetailLevel, detail_level.upper(), DetailLevel.MEDIUM)
                output_format_enum = getattr(OutputFormat, output_format.upper(), OutputFormat.MIXED)
                
                # Convert focus areas
                focus_area_enums = []
                if focus_areas:
                    for area in focus_areas:
                        focus_enum = getattr(FocusArea, area.upper(), None)
                        if focus_enum:
                            focus_area_enums.append(focus_enum)
                
                # Create customization object
                customization = SummaryCustomization(
                    summary_type=summary_type_enum,
                    detail_level=detail_level_enum,
                    output_format=output_format_enum,
                    focus_areas=focus_area_enums if focus_area_enums else None,
                    highlight_relevant_text=highlight_relevant_text,
                    include_keywords=include_keywords,
                    max_length=max_length,
                    user_query=user_query
                )
                
                # Generate enhanced summary
                enhanced_summary = self.enhanced_summarization_service.generate_enhanced_summary(
                    content, title, url, customization
                )
                
                # Convert to dictionary format
                return {
                    "enhanced_summary": {
                        "text": enhanced_summary.text,
                        "type": summary_type,
                        "word_count": len(enhanced_summary.text.split()),
                        "metadata": enhanced_summary.metadata
                    },
                    "key_points": enhanced_summary.key_points,
                    "highlights": [
                        {
                            "text": highlight.text,
                            "relevance": highlight.relevance,
                            "context": highlight.context
                        }
                        for highlight in enhanced_summary.highlights
                    ],
                    "enhanced_keywords": enhanced_summary.keywords,
                    "confidence_score": enhanced_summary.confidence_score,
                    "model": "enhanced_local" if not self.openai_client else "gpt-4-enhanced"
                }
            
            # Fallback to basic summarization
            else:
                logger.warning("Enhanced summarization service not available, using basic summarization")
                basic_summary, bullets = self._generate_user_friendly_summary(content, title)
                
                return {
                    "enhanced_summary": {
                        "text": basic_summary,
                        "type": summary_type,
                        "word_count": len(basic_summary.split()),
                        "metadata": {"method": "basic_fallback"}
                    },
                    "key_points": bullets,
                    "highlights": [],
                    "enhanced_keywords": self._extract_keywords(content),
                    "confidence_score": 0.6,
                    "model": "basic_local"
                }
                
        except Exception as e:
            logger.error(f"Enhanced summary generation failed: {e}")
            # Ultimate fallback
            basic_summary = content[:500] + "..." if len(content) > 500 else content
            return {
                "enhanced_summary": {
                    "text": basic_summary,
                    "type": "basic",
                    "word_count": len(basic_summary.split()),
                    "metadata": {"method": "emergency_fallback", "error": str(e)}
                },
                "key_points": ["Summary generation encountered an error"],
                "highlights": [],
                "enhanced_keywords": [],
                "confidence_score": 0.3,
                "model": "fallback"
            }
    
    def _find_product_info(self, content: str) -> Dict[str, Any]:
        """Extract product-related information"""
        product_keywords = ['product', 'item', 'model', 'brand', 'specification', 'feature']
        info = {}
        
        for keyword in product_keywords:
            pattern = rf'{keyword}[:\s]+([^.]+)'
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                info[keyword] = matches[:3]  # Limit to 3 matches
        
        return info
    
    def _extract_headline(self, content: str) -> str:
        """Extract potential headline from news content"""
        sentences = content.split('.')
        if sentences:
            return sentences[0].strip()
        return ""
    
    def _extract_key_facts(self, content: str) -> List[str]:
        """Extract key facts from news content"""
        fact_indicators = ['according to', 'reported', 'announced', 'confirmed', 'revealed']
        facts = []
        
        sentences = content.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in fact_indicators):
                facts.append(sentence.strip())
        
        return facts[:5]  # Limit to 5 facts
    
    def _extract_contact_info(self, content: str) -> Dict[str, Any]:
        """Extract contact information"""
        entities = self._extract_entities(content)
        return {
            "emails": entities["emails"],
            "phones": entities["phones"]
        }
    
    def _extract_services(self, content: str) -> List[str]:
        """Extract services from business content"""
        service_keywords = ['service', 'offer', 'provide', 'specialize', 'solution']
        services = []
        
        sentences = content.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in service_keywords):
                services.append(sentence.strip())
        
        return services[:5]  # Limit to 5 services
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response that's not in JSON format"""
        # Try to extract structured information from text response
        extraction = {}
        
        # Look for common patterns
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if ':' in line and len(line.split(':')) == 2:
                key, value = line.split(':', 1)
                extraction[key.strip().lower()] = value.strip()
        
        return extraction