import os
import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from models import ScrapedData, ExtractedEntity, ContentCategory, AIInsight, TrendData
from .trend_analysis_service import TrendAnalysisService
from .advanced_analytics import AdvancedAnalyticsService

# NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
except ImportError:
    nltk = None

try:
    import spacy
    # Try to load the English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model not found, use a basic tokenizer
        nlp = None
        print("Warning: spaCy model not available. Using NLTK for NLP features.")
except ImportError:
    spacy = None
    nlp = None
    print("Warning: spaCy not installed. Using NLTK for NLP features.")

# OpenAI SDK v1 compatibility
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

class EnhancedAIService:
    def __init__(self, db = None, openai_api_key: str = None):
        self.db = db
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.openai_api_key and OpenAI:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception:
                self.client = None
        
        # Initialize advanced analytics service
        self.advanced_analytics = AdvancedAnalyticsService(self.client)
        
        # Initialize trend analysis service
        if self.db:
            self.trend_service = TrendAnalysisService(db)
        else:
            self.trend_service = None
        
        # Category keywords for classification
        self.category_keywords = {
            "technology": ["tech", "software", "ai", "artificial intelligence", "machine learning", "programming", "coding", "computer", "digital", "app", "website", "internet", "cyber", "data", "algorithm"],
            "business": ["business", "company", "corporate", "enterprise", "startup", "entrepreneur", "finance", "investment", "market", "economy", "revenue", "profit", "sales", "marketing"],
            "news": ["news", "breaking", "report", "journalist", "media", "press", "announcement", "update", "story", "article", "headline"],
            "politics": ["politics", "government", "policy", "election", "vote", "politician", "congress", "senate", "president", "minister", "law", "legislation"],
            "health": ["health", "medical", "doctor", "hospital", "medicine", "treatment", "disease", "wellness", "fitness", "nutrition", "healthcare"],
            "science": ["science", "research", "study", "experiment", "discovery", "scientist", "laboratory", "analysis", "theory", "hypothesis"],
            "sports": ["sports", "game", "team", "player", "match", "tournament", "championship", "league", "athlete", "competition"],
            "entertainment": ["entertainment", "movie", "film", "music", "celebrity", "actor", "artist", "show", "concert", "theater", "gaming"],
            "education": ["education", "school", "university", "student", "teacher", "learning", "course", "degree", "academic", "study"],
            "finance": ["finance", "bank", "money", "investment", "stock", "market", "trading", "cryptocurrency", "bitcoin", "economy", "financial"]
        }

        # Preferred LLM model (overridable via env)
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Initialize stop words for fallback summarization
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }

    # -----------------------------
    # LLM helpers for accurate outputs
    # -----------------------------
    def _llm_available(self) -> bool:
        try:
            return self.client is not None
        except Exception:
            return False

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            # Try to recover JSON by trimming non-json leading/trailing content
            try:
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start:end+1])
            except Exception:
                pass
        return {}

    async def _llm_comprehensive_analysis(self, content: str, title: str, url: str, features: List[str]) -> Dict[str, Any]:
        """Use LLM to produce unique, accurate, and grounded analysis JSON.
        Falls back gracefully if the API errors."""
        # Limit content length to keep token usage reasonable
        src = (content or "").strip()
        truncated = src[:8000]

        system_msg = (
            "You are an elite content analyst with advanced comprehension and linguistic expertise. Your mission is to provide professional-grade analysis that rivals the best summarization models in the industry. "
            "CORE PRINCIPLES:\n"
            "1. FORMAT PRESERVATION: Maintain the input format structure (mathematical expressions, bullet points, code, tables) in your output\n"
            "2. GRAMMAR EXCELLENCE: Use impeccable grammar, proper sentence structure, and sophisticated vocabulary\n"
            "3. COMPREHENSIVE COVERAGE: Capture ALL essential information, not just highlights\n"
            "4. PROFESSIONAL QUALITY: Deliver analysis that meets enterprise-grade standards\n"
            "5. CONTEXTUAL INTELLIGENCE: Understand implicit meanings, relationships, and domain-specific nuances\n"
            "6. ACCURACY PRIORITY: Ensure factual correctness and avoid hallucinations\n"
            "7. STRUCTURAL COHERENCE: Organize information logically with smooth transitions\n"
            "8. DOMAIN EXPERTISE: Adapt language and focus based on content type (technical, business, academic, etc.)\n"
            "When processing mathematical content, preserve formulas and expressions exactly. "
            "When processing lists, maintain the list structure in summaries. "
            "When processing code, reference technical concepts appropriately. "
            "Always use proper grammar, complete sentences, and professional tone throughout all outputs."
        )

        # Detect content format for format-aware processing
        content_format = self._detect_content_format(content)
        format_instructions = self._generate_format_instructions(content_format)
        
        user_prompt = (
            "Input title: " + (title or "") + "\n"
            "Input URL: " + (url or "") + "\n"
            "Content format detected: " + str(content_format) + "\n\n"
            "Input content:\n" + truncated + "\n\n"
            f"FORMAT-SPECIFIC INSTRUCTIONS:\n{format_instructions}\n\n"
            "Return strict JSON with keys present only if requested in features: " + ", ".join(features) + "\n"
            "Schema: {\n"
            "  summary: { text: string, key_points: string[], word_count: number, format_preserved: boolean },\n"
            "  summary_views: { tldr: {text: string, style: string}, executive: {text: string, focus: string}, technical: {text: string, mentions: string[], complexity: string}, marketing: {text: string, hooks: string[], appeal: string} },\n"
            "  keywords: { primary: string[], secondary: string[], domain_specific: string[], technical_terms: string[] },\n"
            "  topics: { main_topics: string[], subtopics: string[], topic_distribution: { [topic]: number }, domain: string },\n"
            "  sentiment: { overall: { label: string, polarity: number, subjectivity: number, confidence: number }, emotional_tone: string, professional_tone: string },\n"
            "  insights: { key_insights: string[], recommendations: string[], content_quality: { score: number, level: string, grammar_score: number }, readability: number, gaps_filled: string[], details: [{ text: string, confidence: number, rationale: string, evidence: string }] }\n"
            "}\n"
            "PROFESSIONAL-GRADE REQUIREMENTS:\n"
            "GRAMMAR & LANGUAGE:\n"
            "- Use perfect grammar, punctuation, and sentence structure throughout\n"
            "- Employ sophisticated vocabulary appropriate to the content domain\n"
            "- Ensure smooth transitions between ideas and logical flow\n"
            "- Maintain consistent tense and voice\n"
            "- Use active voice where appropriate for clarity and impact\n\n"
            "SUMMARY EXCELLENCE:\n"
            "- Main summary.text: 250-600 words, comprehensive coverage of ALL content aspects\n"
            "- Include: background context, main arguments, supporting evidence, methodology (if applicable), implications, conclusions, and future considerations\n"
            "- Preserve original format elements (math formulas, lists, technical terms) exactly as they appear\n"
            "- Key_points: 10-20 detailed, well-structured bullet points covering distinct aspects\n"
            "- Each point should be a complete, grammatically correct sentence\n"
            "- Organize points logically (introduction → main content → conclusions)\n\n"
            "SUMMARY VIEWS - PROFESSIONAL STANDARDS:\n"
            "- TLDR: 120-180 words, conversational yet comprehensive, captures essence without losing critical details\n"
            "- Executive: 200-300 words, strategic business perspective, focus on impact, ROI, and decision-making insights\n"
            "- Technical: 250-400 words, detailed technical analysis, preserve all technical terms, formulas, and specifications\n"
            "- Marketing: 180-250 words, benefit-driven narrative, compelling value propositions, audience-focused messaging\n\n"
            "ENHANCED ANALYSIS:\n"
            "- Keywords: Include domain-specific terminology, avoid generic words, prioritize by relevance and frequency\n"
            "- Topics: Identify main themes, sub-themes, and their relationships with confidence scores\n"
            "- Sentiment: Provide nuanced analysis with professional tone assessment\n"
            "- Insights: Fill gaps in the original content, provide expert-level observations, suggest improvements or extensions\n"
            "- Content Quality: Assess grammar, structure, completeness, and professional presentation\n\n"
            "CRITICAL: Every output must demonstrate enterprise-grade quality that exceeds current industry standards for AI summarization."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                top_p=0.9
            )
            content_text = resp.choices[0].message.content if resp and resp.choices else ""
            data = self._parse_json_safely(content_text)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return {}

    async def enhanced_analyze_content(
        self, 
        content: str, 
        title: str = "", 
        url: str = "",
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced comprehensive AI analysis with new features"""
        
        # Preprocess content
        preprocessed = self._preprocess_content(content, url)
        
        # Base analysis
        analysis = {
            "summary": "",
            "sentiment": {},
            "keywords": [],
            "entities": {},
            "smart_extraction": {},
            "content_type": "",
            "language": "en",
            "readability_score": 0,
            # New enhanced features
            "advanced_entities": {},
            "categories": [],
            "trend_indicators": {},
            "insights": []
        }
        
        try:
            # Enhanced entity extraction
            analysis["advanced_entities"] = await self._extract_advanced_entities(preprocessed)
            
            # Content categorization
            analysis["categories"] = self._categorize_content(preprocessed, title, url)
            
            # Trend analysis indicators
            analysis["trend_indicators"] = self._analyze_trend_indicators(preprocessed, url)
            
            # Generate AI insights
            analysis["insights"] = await self._generate_ai_insights(preprocessed, title, url)
            
            # Original analysis features
            if self.client:
                openai_analysis = await self._openai_analysis(preprocessed, title, url, custom_prompt)
                analysis.update(openai_analysis)
            else:
                # Fallback analysis
                analysis["summary"] = self._generate_user_friendly_summary(preprocessed, title)[0]
                analysis["sentiment"] = self._analyze_sentiment(preprocessed)
                analysis["keywords"] = self._extract_keywords(preprocessed)
                analysis["entities"] = self._extract_entities(preprocessed)
                analysis["smart_extraction"] = self._smart_extract_fallback(preprocessed, url)
                analysis["content_type"] = self._classify_content_type(preprocessed, title, url)
                analysis["language"] = self._detect_language(preprocessed)
                analysis["readability_score"] = self._calculate_readability(preprocessed)
                
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            # Fallback to basic analysis
            analysis["summary"] = "Analysis temporarily unavailable"
            analysis["sentiment"] = {"polarity": 0, "subjectivity": 0, "label": "neutral"}
                
        return analysis

    async def comprehensive_content_analysis(
        self,
        content: str,
        title: str = "",
        url: str = "",
        features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive content analysis with customizable features
        Features: summary, keywords, topics, sentiment, entities, insights
        """
        if features is None:
            features = ["summary", "keywords", "topics", "sentiment", "entities", "insights"]
        
        # Preprocess content with format detection
        preprocessed = self._preprocess_content(content, url)
        
        # Detect and preserve content format
        content_format = self._detect_content_format(content)
        
        analysis_result = {
            "content_length": len(content),
            "processed_length": len(preprocessed),
            "language": "en",
            "timestamp": datetime.utcnow().isoformat(),
            "format_info": {
                "detected_formats": content_format,
                "format_preserved": True,
                "original_structure": self._analyze_content_structure(content)
            }
        }

        try:
            # Prefer LLM-backed comprehensive generation when available
            llm_data: Dict[str, Any] = {}
            if self._llm_available():
                llm_data = await self._llm_comprehensive_analysis(preprocessed, title, url, features)

            # Summary generation (LLM or heuristic)
            if "summary" in features:
                if llm_data.get("summary"):
                    s = llm_data["summary"]
                    analysis_result["summary"] = {
                        "text": s.get("text", ""),
                        "key_points": s.get("key_points", []) or [],
                        "word_count": s.get("word_count", len((s.get("text", "") or "").split()))
                    }
                else:
                    summary_data = self._generate_user_friendly_summary(preprocessed, title)
                    analysis_result["summary"] = {
                        "text": summary_data[0],
                        "key_points": summary_data[1] if len(summary_data) > 1 else [],
                        "word_count": len(summary_data[0].split()) if summary_data[0] else 0
                    }
                # Summary views (LLM or heuristic)
                if llm_data.get("summary_views"):
                    analysis_result["summary_views"] = llm_data.get("summary_views", {})
                else:
                    # Extract summary text and key points properly
                    summary_text = analysis_result["summary"]["text"] if isinstance(analysis_result["summary"], dict) else str(analysis_result["summary"])
                    key_points = analysis_result["summary"].get("key_points", []) if isinstance(analysis_result["summary"], dict) else []
                    analysis_result["summary_views"] = self._generate_summary_views(preprocessed, title, [summary_text, key_points])
            
            # Keywords extraction (Advanced Analytics or LLM or heuristic)
            if "keywords" in features:
                if llm_data.get("keywords"):
                    kw = llm_data["keywords"]
                    primary = [k for k in (kw.get("primary") or []) if isinstance(k, str)]
                    secondary = [k for k in (kw.get("secondary") or []) if isinstance(k, str)]
                    # Deduplicate and filter overly generic tokens
                    stopish = {"the","and","or","in","of","a","an","is","are","to","for","with","on","by","from"}
                    def clean_list(lst):
                        out = []
                        for token in lst:
                            t = token.strip()
                            if not t:
                                continue
                            low = t.lower()
                            if low in stopish or len(low) < 2:
                                continue
                            if t not in out:
                                out.append(t)
                        return out
                    primary = clean_list(primary)[:12]
                    secondary = clean_list(secondary)[:12]
                    analysis_result["keywords"] = {
                        "primary": primary,
                        "secondary": secondary,
                        "total_count": len(primary) + len(secondary)
                    }
                else:
                    # Use advanced analytics for better keyword extraction
                    advanced_keywords = self.advanced_analytics.advanced_keyword_extraction(preprocessed, 20)
                    analysis_result["keywords"] = {
                        "primary": advanced_keywords.get("primary", []),
                        "secondary": advanced_keywords.get("secondary", []),
                        "total_count": len(advanced_keywords.get("primary", [])) + len(advanced_keywords.get("secondary", []))
                    }
            
            # Topics identification (Advanced Analytics or LLM or heuristic)
            if "topics" in features:
                if llm_data.get("topics"):
                    tp = llm_data["topics"]
                    main_topics = [t for t in (tp.get("main_topics") or []) if isinstance(t, str)]
                    subtopics = [t for t in (tp.get("subtopics") or []) if isinstance(t, str)]
                    dist = tp.get("topic_distribution") or {}
                    # Normalize distribution values to 0..1
                    try:
                        total = sum(float(v) for v in dist.values())
                        if total > 0:
                            dist = {k: float(v)/total for k, v in dist.items()}
                    except Exception:
                        pass
                    analysis_result["topics"] = {
                        "main_topics": main_topics[:5],
                        "subtopics": subtopics[:5],
                        "topic_distribution": dist
                    }
                else:
                    # Use advanced topic modeling with LDA
                    advanced_topics = self.advanced_analytics.advanced_topic_modeling(preprocessed)
                    analysis_result["topics"] = {
                        "main_topics": advanced_topics.get("main_topics", []),
                        "subtopics": advanced_topics.get("subtopics", []),
                        "topic_distribution": advanced_topics.get("topic_distribution", {})
                    }
            
            # Sentiment analysis (Advanced Analytics or LLM or heuristic)
            if "sentiment" in features:
                if llm_data.get("sentiment"):
                    se = llm_data["sentiment"]
                    overall = se.get("overall") or {}
                    # Ensure required fields
                    overall.setdefault("label", "neutral")
                    overall.setdefault("polarity", 0.0)
                    overall.setdefault("subjectivity", 0.5)
                    overall.setdefault("confidence", 0.6)
                    analysis_result["sentiment"] = {
                        "overall": overall,
                        "confidence": overall.get("confidence", 0.6),
                        "emotional_tone": se.get("emotional_tone", "balanced")
                    }
                else:
                    # Use advanced sentiment analysis with emotion detection
                    advanced_sentiment = self.advanced_analytics.advanced_sentiment_analysis(preprocessed)
                    analysis_result["sentiment"] = {
                        "overall": advanced_sentiment.get("overall", {}),
                        "confidence": advanced_sentiment.get("overall", {}).get("confidence", 0.5),
                        "emotional_tone": advanced_sentiment.get("emotional_tone", "neutral")
                    }
            
            # Entity extraction (heuristic NLP, then highlights)
            if "entities" in features:
                entities = await self._extract_advanced_entities(preprocessed)
                analysis_result["entities"] = {
                    "people": entities.get("PERSON", []),
                    "organizations": entities.get("ORG", []),
                    "locations": entities.get("GPE", []),
                    "dates": entities.get("DATE", []),
                    "money": entities.get("MONEY", []),
                    "total_entities": sum(len(v) for v in entities.values())
                }
                # Source-grounded highlights for entities/keywords
                analysis_result["highlights"] = self._extract_evidence_spans(
                    preprocessed,
                    analysis_result.get("keywords", {}).get("primary", []),
                    analysis_result["entities"]
                )
            
            # AI-powered insights (LLM or heuristic)
            if "insights" in features:
                if llm_data.get("insights"):
                    ins = llm_data["insights"]
                    # Normalize expected keys
                    key_insights = ins.get("key_insights", []) or []
                    recommendations = ins.get("recommendations", []) or []
                    content_quality = ins.get("content_quality", {}) or {}
                    readability = ins.get("readability", 0)
                    details = ins.get("details", []) or []
                    # Ensure details are well-formed
                    normalized_details = []
                    for d in details:
                        if not isinstance(d, dict):
                            continue
                        normalized_details.append({
                            "text": d.get("text", ""),
                            "confidence": round(float(d.get("confidence", 0.6)), 2),
                            "rationale": d.get("rationale", ""),
                            "evidence": d.get("evidence", None)
                        })
                    analysis_result["insights"] = {
                        "key_insights": key_insights[:6],
                        "recommendations": recommendations[:6],
                        "content_quality": content_quality,
                        "readability": readability,
                        "details": normalized_details
                    }
                else:
                    # Use advanced insights generation with evidence and reasoning
                    advanced_insights = await self.advanced_analytics.generate_advanced_insights(
                        preprocessed,
                        analysis_result.get("keywords", {}).get("primary", []),
                        analysis_result.get("sentiment", {}),
                        analysis_result.get("topics", {}).get("main_topics", [])
                    )
                    
                    # Fallback to basic insights if advanced fails
                    basic_insights = await self._generate_comprehensive_insights(preprocessed, title, url)
                    
                    analysis_result["insights"] = {
                        "key_insights": basic_insights[:5],
                        "recommendations": self._generate_recommendations(preprocessed),
                        "content_quality": self._assess_content_quality(preprocessed),
                        "readability": self._calculate_readability_score(preprocessed),
                        "details": advanced_insights  # Use advanced insights as details
                    }

        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            analysis_result["error"] = f"Analysis partially failed: {str(e)}"

        return analysis_result

    async def _extract_advanced_entities(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Advanced entity extraction using spaCy and custom patterns"""
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "products": [],
            "money": [],
            "dates": [],
            "emails": [],
            "phones": [],
            "urls": [],
            "technologies": [],
            "events": []
        }
        
        try:
            # Use spaCy if available
            if nlp:
                doc = nlp(content[:1000000])  # Limit content size for performance
                
                for ent in doc.ents:
                    entity_data = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.8  # spaCy doesn't provide confidence scores by default
                    }
                    
                    if ent.label_ in ["PERSON"]:
                        entities["persons"].append(entity_data)
                    elif ent.label_ in ["ORG"]:
                        entities["organizations"].append(entity_data)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities["locations"].append(entity_data)
                    elif ent.label_ in ["PRODUCT"]:
                        entities["products"].append(entity_data)
                    elif ent.label_ in ["MONEY"]:
                        entities["money"].append(entity_data)
                    elif ent.label_ in ["DATE", "TIME"]:
                        entities["dates"].append(entity_data)
                    elif ent.label_ in ["EVENT"]:
                        entities["events"].append(entity_data)
            elif nltk:  # Use NLTK if spaCy not available
                nltk_entities = self._extract_nltk_entities(content[:100000])  # Limit content size
                for entity in nltk_entities:
                    if entity["label"] in ["PERSON"]:
                        entities["persons"].append(entity)
                    elif entity["label"] in ["ORGANIZATION"]:
                        entities["organizations"].append(entity)
                    elif entity["label"] in ["GPE"]:
                        entities["locations"].append(entity)
            
            # Custom pattern extraction
            custom_entities = self._extract_custom_entities(content)
            for key, values in custom_entities.items():
                if key in entities:
                    entities[key].extend(values)
            
            # Remove duplicates and sort by confidence
            for key in entities:
                entities[key] = self._deduplicate_entities(entities[key])
                
        except Exception as e:
            print(f"Error in advanced entity extraction: {e}")
            # Fallback to basic extraction
            basic_entities = self._extract_entities(content)
            for key, values in basic_entities.items():
                if key in entities:
                    entities[key] = [{"text": v, "confidence": 0.5} for v in values]
        
        return entities

    def _extract_nltk_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using NLTK"""
        entities = []
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            tree = ne_chunk(pos_tags)
            
            current_entity = []
            current_label = None
            start_pos = 0
            
            for i, item in enumerate(tree):
                if hasattr(item, 'label'):  # It's a named entity
                    if current_label != item.label():
                        # Save previous entity if exists
                        if current_entity:
                            entity_text = ' '.join([token for token, pos in current_entity])
                            entities.append({
                                "text": entity_text,
                                "label": current_label,
                                "start": start_pos,
                                "end": start_pos + len(entity_text),
                                "confidence": 0.7
                            })
                        
                        # Start new entity
                        current_entity = list(item)
                        current_label = item.label()
                        start_pos = text.find(current_entity[0][0], start_pos)
                    else:
                        # Continue current entity
                        current_entity.extend(list(item))
                else:
                    # Save current entity if exists
                    if current_entity:
                        entity_text = ' '.join([token for token, pos in current_entity])
                        entities.append({
                            "text": entity_text,
                            "label": current_label,
                            "start": start_pos,
                            "end": start_pos + len(entity_text),
                            "confidence": 0.7
                        })
                        current_entity = []
                        current_label = None
            
            # Don't forget the last entity
            if current_entity:
                entity_text = ' '.join([token for token, pos in current_entity])
                entities.append({
                    "text": entity_text,
                    "label": current_label,
                    "start": start_pos,
                    "end": start_pos + len(entity_text),
                    "confidence": 0.7
                })
                
        except Exception as e:
            print(f"NLTK entity extraction error: {e}")
        
        return entities

    def _extract_custom_entities(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using custom patterns"""
        entities = {
            "emails": [],
            "phones": [],
            "urls": [],
            "technologies": [],
            "money": []
        }
        
        patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phones": r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            "urls": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "money": r'\$[\d,]+\.?\d*|€[\d,]+\.?\d*|£[\d,]+\.?\d*|\d+\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)'
        }
        
        # Technology keywords
        tech_keywords = ["Python", "JavaScript", "React", "Node.js", "AI", "ML", "API", "AWS", "Docker", "Kubernetes", "blockchain", "cryptocurrency"]
        
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities[pattern_name].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        
        # Extract technology mentions
        for tech in tech_keywords:
            pattern = r'\b' + re.escape(tech) + r'\b'
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities["technologies"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                })
        
        return entities

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities and sort by confidence"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            text_lower = entity["text"].lower().strip()
            if text_lower not in seen and len(text_lower) > 1:
                seen.add(text_lower)
                unique_entities.append(entity)
        
        # Sort by confidence if available
        return sorted(unique_entities, key=lambda x: x.get("confidence", 0), reverse=True)

    def _categorize_content(self, content: str, title: str = "", url: str = "") -> List[Dict[str, Any]]:
        """Categorize content using keyword matching and ML techniques"""
        categories = []
        
        try:
            # Combine title and content for analysis
            full_text = f"{title} {content}".lower()
            
            # Calculate category scores
            category_scores = {}
            for category, keywords in self.category_keywords.items():
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    count = full_text.count(keyword.lower())
                    if count > 0:
                        score += count
                        matched_keywords.append(keyword)
                
                if score > 0:
                    # Normalize score by content length
                    normalized_score = min(score / (len(full_text.split()) / 100), 1.0)
                    category_scores[category] = {
                        "score": normalized_score,
                        "matched_keywords": matched_keywords
                    }
            
            # Sort categories by score and return top matches
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            
            for category, data in sorted_categories[:3]:  # Top 3 categories
                if data["score"] > 0.1:  # Minimum threshold
                    categories.append({
                        "category": category,
                        "confidence": data["score"],
                        "keywords": data["matched_keywords"][:5]  # Top 5 matched keywords
                    })
            
            # If no categories found, try to infer from URL
            if not categories and url:
                url_category = self._infer_category_from_url(url)
                if url_category:
                    categories.append(url_category)
            
        except Exception as e:
            print(f"Error in content categorization: {e}")
        
        return categories

    def _infer_category_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Infer category from URL patterns"""
        url_lower = url.lower()
        
        url_patterns = {
            "news": ["news", "cnn", "bbc", "reuters", "ap", "npr"],
            "technology": ["tech", "github", "stackoverflow", "medium", "dev"],
            "business": ["business", "forbes", "bloomberg", "wsj", "finance"],
            "sports": ["sports", "espn", "nfl", "nba", "fifa"],
            "entertainment": ["entertainment", "imdb", "netflix", "spotify"]
        }
        
        for category, patterns in url_patterns.items():
            for pattern in patterns:
                if pattern in url_lower:
                    return {
                        "category": category,
                        "confidence": 0.7,
                        "keywords": [pattern]
                    }
        
        return None

    def _analyze_trend_indicators(self, content: str, url: str = "") -> Dict[str, Any]:
        """Analyze content for trend indicators"""
        indicators = {
            "sentiment_trend": None,
            "price_mentions": [],
            "temporal_references": [],
            "growth_indicators": [],
            "decline_indicators": [],
            "comparison_metrics": []
        }
        
        try:
            # Sentiment analysis for trend
            sentiment = self._analyze_sentiment(content)
            indicators["sentiment_trend"] = {
                "current_sentiment": sentiment["label"],
                "polarity": sentiment["polarity"],
                "strength": abs(sentiment["polarity"])
            }
            
            # Extract price mentions
            price_pattern = r'\$[\d,]+\.?\d*|€[\d,]+\.?\d*|£[\d,]+\.?\d*|\d+\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)'
            price_matches = re.findall(price_pattern, content, re.IGNORECASE)
            indicators["price_mentions"] = list(set(price_matches))
            
            # Temporal references
            temporal_patterns = [
                r'\b(?:yesterday|today|tomorrow|last week|next week|this month|last month)\b',
                r'\b(?:increasing|decreasing|rising|falling|growing|declining)\b',
                r'\b(?:up|down|higher|lower)\s+(?:by|from)\s+\d+%?\b'
            ]
            
            for pattern in temporal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                indicators["temporal_references"].extend(matches)
            
            # Growth indicators
            growth_keywords = ["increase", "growth", "rise", "surge", "boom", "expansion", "uptick", "gain"]
            decline_keywords = ["decrease", "decline", "fall", "drop", "crash", "recession", "downturn", "loss"]
            
            for keyword in growth_keywords:
                if keyword.lower() in content.lower():
                    indicators["growth_indicators"].append(keyword)
            
            for keyword in decline_keywords:
                if keyword.lower() in content.lower():
                    indicators["decline_indicators"].append(keyword)
            
            # Extract comparison metrics
            comparison_pattern = r'\b\d+%?\s*(?:more|less|higher|lower|faster|slower)\b'
            comparisons = re.findall(comparison_pattern, content, re.IGNORECASE)
            indicators["comparison_metrics"] = list(set(comparisons))
            
        except Exception as e:
            print(f"Error in trend analysis: {e}")
        
        return indicators

    async def _generate_ai_insights(self, content: str, title: str = "", url: str = "") -> List[Dict[str, Any]]:
        """Generate AI-powered insights about the content"""
        insights = []
        
        try:
            # Generate different types of insights
            insights.extend(self._generate_content_insights(content, title))
            insights.extend(self._generate_business_insights(content))
            insights.extend(self._generate_technical_insights(content))
            
            # If OpenAI is available, generate advanced insights
            if self.client:
                ai_insights = await self._generate_openai_insights(content, title)
                insights.extend(ai_insights)
            
        except Exception as e:
            print(f"Error generating AI insights: {e}")
        
        return insights

    async def store_analysis_trends(self, scraped_data: ScrapedData, analysis: Dict[str, Any]) -> bool:
        """Store trend data from analysis results"""
        if not self.trend_service or not scraped_data.job:
            return False
        
        try:
            # Extract domain from URL
            from urllib.parse import urlparse
            domain = urlparse(scraped_data.url).netloc
            user_id = scraped_data.job.user_id
            
            # Store sentiment trend
            if "sentiment" in analysis and "polarity" in analysis["sentiment"]:
                await self.trend_service.store_trend_data(
                    user_id=user_id,
                    domain=domain,
                    trend_type="sentiment",
                    metric_name="polarity",
                    metric_value=analysis["sentiment"]["polarity"],
                    metadata={"subjectivity": analysis["sentiment"].get("subjectivity", 0)}
                )
            
            # Store content length trend
            if "content_length" in analysis:
                await self.trend_service.store_trend_data(
                    user_id=user_id,
                    domain=domain,
                    trend_type="content",
                    metric_name="length",
                    metric_value=analysis["content_length"],
                    metadata={"content_type": analysis.get("content_type", "unknown")}
                )
            
            # Store readability trend
            if "readability_score" in analysis:
                await self.trend_service.store_trend_data(
                    user_id=user_id,
                    domain=domain,
                    trend_type="readability",
                    metric_name="score",
                    metric_value=analysis["readability_score"],
                    metadata={}
                )
            
            # Store keyword count trend
            if "keywords" in analysis:
                keyword_count = len(analysis["keywords"]) if isinstance(analysis["keywords"], list) else 0
                await self.trend_service.store_trend_data(
                    user_id=user_id,
                    domain=domain,
                    trend_type="keywords",
                    metric_name="count",
                    metric_value=keyword_count,
                    metadata={}
                )
            
            return True
            
        except Exception as e:
            print(f"Error storing analysis trends: {e}")
            return False

    async def get_user_trends(self, user_id: int, domain: str = None, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive trend analysis for a user"""
        if not self.trend_service:
            return {"error": "Trend service not available"}
        
        return await self.trend_service.analyze_trends(
            user_id=user_id,
            domain=domain,
            time_range=days
        )

    def _generate_content_insights(self, content: str, title: str = "") -> List[Dict[str, Any]]:
        """Generate content-specific insights"""
        insights = []
        
        # Content length insight
        word_count = len(content.split())
        if word_count > 2000:
            insights.append({
                "type": "content_analysis",
                "insight": f"This is a comprehensive piece with {word_count} words, indicating detailed coverage of the topic.",
                "confidence": 0.9
            })
        elif word_count < 200:
            insights.append({
                "type": "content_analysis",
                "insight": f"This is a brief summary with {word_count} words, suitable for quick consumption.",
                "confidence": 0.8
            })
        
        # Readability insight
        readability = self._calculate_readability(content)
        if readability > 15:
            insights.append({
                "type": "readability",
                "insight": "This content requires advanced reading skills and may be technical in nature.",
                "confidence": 0.8
            })
        elif readability < 8:
            insights.append({
                "type": "readability",
                "insight": "This content is easily readable and accessible to a general audience.",
                "confidence": 0.8
            })
        
        return insights

    def _generate_business_insights(self, content: str) -> List[Dict[str, Any]]:
        """Generate business-related insights"""
        insights = []
        
        # Financial mentions
        financial_keywords = ["revenue", "profit", "loss", "investment", "funding", "valuation", "IPO", "acquisition"]
        found_financial = [kw for kw in financial_keywords if kw.lower() in content.lower()]
        
        if found_financial:
            insights.append({
                "type": "business_analysis",
                "insight": f"Content discusses financial aspects: {', '.join(found_financial)}",
                "confidence": 0.7
            })
        
        # Market indicators
        market_keywords = ["market", "competition", "competitor", "industry", "sector"]
        found_market = [kw for kw in market_keywords if kw.lower() in content.lower()]
        
        if found_market:
            insights.append({
                "type": "market_analysis",
                "insight": f"Content includes market analysis elements: {', '.join(found_market)}",
                "confidence": 0.7
            })
        
        return insights

    def _generate_technical_insights(self, content: str) -> List[Dict[str, Any]]:
        """Generate technical insights"""
        insights = []
        
        # Technology stack mentions
        tech_stack = ["Python", "JavaScript", "React", "Node.js", "Docker", "AWS", "API", "database"]
        found_tech = [tech for tech in tech_stack if tech.lower() in content.lower()]
        
        if found_tech:
            insights.append({
                "type": "technical_analysis",
                "insight": f"Technical content mentions: {', '.join(found_tech)}",
                "confidence": 0.8
            })
        
        # Code or technical patterns
        if re.search(r'```|`[^`]+`|function\s+\w+|class\s+\w+|import\s+\w+', content):
            insights.append({
                "type": "technical_analysis",
                "insight": "Content contains code snippets or technical documentation.",
                "confidence": 0.9
            })
        
        return insights

    async def _generate_openai_insights(self, content: str, title: str = "") -> List[Dict[str, Any]]:
        """Generate insights using OpenAI"""
        insights = []
        
        try:
            prompt = f"""
            Analyze the following content and provide 3-5 key insights:
            
            Title: {title}
            Content: {content[:2000]}...
            
            Please provide insights in the following format:
            1. [Insight type]: [Insight description]
            2. [Insight type]: [Insight description]
            
            Focus on business implications, trends, and actionable information.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse the response into structured insights
            lines = ai_response.split('\n')
            for line in lines:
                if line.strip() and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        insight_type = parts[0].strip().replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '').strip()
                        insight_text = parts[1].strip()
                        
                        insights.append({
                            "type": "ai_generated",
                            "category": insight_type,
                            "insight": insight_text,
                            "confidence": 0.8
                        })
            
        except Exception as e:
            print(f"Error generating OpenAI insights: {e}")
        
        return insights

    # Include all the original methods from the base AI service
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(content)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
                
            return {
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "label": label
            }
        except:
            return {
                "polarity": 0,
                "subjectivity": 0,
                "label": "neutral"
            }

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            # Simple keyword extraction using word frequency
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            word_freq = Counter(words)
            
            # Filter out common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
            
            filtered_words = {word: freq for word, freq in word_freq.items() if word not in stop_words and len(word) > 3}
            
            return list(dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:max_keywords]).keys())
        except:
            return []

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Basic entity extraction using regex patterns"""
        entities = {
            "emails": [],
            "phones": [],
            "urls": [],
            "dates": [],
            "prices": []
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
            
            # Date pattern
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            entities["dates"] = list(set(re.findall(date_pattern, content)))
            
        except:
            pass
        
        return entities

    def _detect_language(self, content: str) -> str:
        """Detect content language"""
        try:
            blob = TextBlob(content[:1000])
            return blob.detect_language()
        except:
            return "en"

    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)"""
        try:
            sentences = content.split('.')
            words = content.split()
            syllables = sum([self._count_syllables(word) for word in words])
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            # Simplified Flesch Reading Ease formula
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score))
        except:
            return 50  # Default middle score

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)

    def _identify_topics(self, content: str) -> List[str]:
        """Identify main topics in the content"""
        try:
            # Extract keywords and group them into topics
            keywords = self._extract_keywords(content)
            
            # Simple topic identification based on keyword clustering
            topics = []
            
            # Technology topics
            tech_keywords = [kw for kw in keywords if any(term in kw.lower() for term in 
                           ['technology', 'software', 'digital', 'ai', 'machine', 'computer', 'data', 'internet'])]
            if tech_keywords:
                topics.append("Technology")
            
            # Business topics
            business_keywords = [kw for kw in keywords if any(term in kw.lower() for term in 
                               ['business', 'market', 'company', 'financial', 'economy', 'revenue', 'profit'])]
            if business_keywords:
                topics.append("Business")
            
            # Health topics
            health_keywords = [kw for kw in keywords if any(term in kw.lower() for term in 
                             ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine', 'disease'])]
            if health_keywords:
                topics.append("Health")
            
            # Politics topics
            politics_keywords = [kw for kw in keywords if any(term in kw.lower() for term in 
                               ['politics', 'government', 'policy', 'election', 'president', 'congress', 'law'])]
            if politics_keywords:
                topics.append("Politics")
            
            # Science topics
            science_keywords = [kw for kw in keywords if any(term in kw.lower() for term in 
                              ['science', 'research', 'study', 'experiment', 'discovery', 'scientific'])]
            if science_keywords:
                topics.append("Science")
            
            # Add general topics based on content
            if not topics:
                topics = ["General", "Information"]
            
            return topics[:10]  # Return top 10 topics
            
        except Exception as e:
            print(f"Error identifying topics: {str(e)}")
            return ["General"]

    def _calculate_topic_distribution(self, content: str, topics: List[str]) -> Dict[str, float]:
        """Calculate the distribution of topics in the content"""
        try:
            content_lower = content.lower()
            total_words = len(content.split())
            
            distribution = {}
            for topic in topics:
                # Count words related to each topic
                topic_words = 0
                if topic.lower() == "technology":
                    topic_words = sum(1 for word in content.split() if any(term in word.lower() for term in 
                                    ['tech', 'digital', 'software', 'computer', 'data', 'internet']))
                elif topic.lower() == "business":
                    topic_words = sum(1 for word in content.split() if any(term in word.lower() for term in 
                                    ['business', 'market', 'company', 'financial', 'revenue']))
                elif topic.lower() == "health":
                    topic_words = sum(1 for word in content.split() if any(term in word.lower() for term in 
                                    ['health', 'medical', 'doctor', 'treatment', 'medicine']))
                else:
                    topic_words = 1  # Default weight
                
                distribution[topic] = round((topic_words / max(total_words, 1)) * 100, 2)
            
            return distribution
            
        except Exception as e:
            print(f"Error calculating topic distribution: {str(e)}")
            return {topic: 10.0 for topic in topics}

    def _analyze_emotional_tone(self, content: str) -> Dict[str, Any]:
        """Analyze the emotional tone of the content"""
        try:
            content_lower = content.lower()
            
            # Define emotion keywords
            emotions = {
                "joy": ["happy", "joy", "excited", "pleased", "delighted", "cheerful"],
                "anger": ["angry", "furious", "mad", "irritated", "annoyed", "outraged"],
                "sadness": ["sad", "depressed", "disappointed", "grief", "sorrow", "melancholy"],
                "fear": ["afraid", "scared", "worried", "anxious", "nervous", "terrified"],
                "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
                "trust": ["trust", "confident", "reliable", "secure", "certain", "assured"]
            }
            
            emotion_scores = {}
            total_words = len(content.split())
            
            for emotion, keywords in emotions.items():
                count = sum(1 for word in content.split() if word.lower() in keywords)
                emotion_scores[emotion] = round((count / max(total_words, 1)) * 100, 2)
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
            
            return {
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
                "emotional_intensity": max(emotion_scores.values()) if emotion_scores else 0
            }
            
        except Exception as e:
            print(f"Error analyzing emotional tone: {str(e)}")
            return {"dominant_emotion": "neutral", "emotion_scores": {}, "emotional_intensity": 0}

    async def _generate_comprehensive_insights(self, content: str, title: str, url: str) -> List[str]:
        """Generate comprehensive insights about the content"""
        try:
            insights = []
            
            # Content length insights
            word_count = len(content.split())
            if word_count > 1000:
                insights.append("This is a comprehensive, long-form content piece suitable for in-depth analysis.")
            elif word_count < 100:
                insights.append("This is a brief content piece that may benefit from expansion.")
            else:
                insights.append("This content has a moderate length suitable for quick consumption.")
            
            # Sentiment insights
            sentiment = self._analyze_sentiment(content)
            if sentiment.get("polarity", 0) > 0.3:
                insights.append("The content has a positive tone and optimistic outlook.")
            elif sentiment.get("polarity", 0) < -0.3:
                insights.append("The content expresses negative sentiment or concerns.")
            else:
                insights.append("The content maintains a neutral and balanced perspective.")
            
            # Complexity insights
            readability = self._calculate_readability_score(content)
            if readability > 70:
                insights.append("The content is easily readable and accessible to a general audience.")
            elif readability < 30:
                insights.append("The content is complex and may require specialized knowledge.")
            else:
                insights.append("The content has moderate complexity suitable for educated readers.")
            
            # Structure insights
            sentences = content.split('.')
            avg_sentence_length = len(content.split()) / max(len(sentences), 1)
            if avg_sentence_length > 20:
                insights.append("The content uses long, complex sentences that may impact readability.")
            elif avg_sentence_length < 10:
                insights.append("The content uses short, concise sentences for easy comprehension.")
            
            return insights[:5]  # Return top 5 insights

        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return ["Content analysis completed successfully."]

    def _generate_summary_views(self, content: str, title: str, summary_data: List[str]) -> Dict[str, Any]:
        """Create professional-grade summary perspectives with enhanced alignment and focus."""
        base_summary = summary_data[0] if summary_data and len(summary_data) > 0 else ""
        bullets = summary_data[1] if summary_data and len(summary_data) > 1 else []
        
        # Extract content insights for better view generation
        content_insights = self._analyze_content_for_views(content, title)
        
        # TL;DR: Conversational yet comprehensive (120-180 words)
        tldr_text = self._generate_tldr_view(base_summary, bullets, content_insights)
        
        # Executive: Strategic business perspective (200-300 words)
        executive_text = self._generate_executive_view(base_summary, bullets, content_insights)
        
        # Technical: Detailed technical analysis (250-400 words)
        technical_data = self._generate_technical_view(base_summary, bullets, content_insights, content)
        
        # Marketing: Benefit-driven narrative (180-250 words)
        marketing_data = self._generate_marketing_view(base_summary, bullets, content_insights)

        return {
            "tldr": {
                "text": tldr_text,
                "style": "conversational",
                "word_count": len(tldr_text.split()),
                "focus": "essence_capture"
            },
            "executive": {
                "text": executive_text,
                "focus": "strategic_impact",
                "word_count": len(executive_text.split()),
                "perspective": "business_leadership"
            },
            "technical": {
                "text": technical_data["text"],
                "mentions": technical_data["mentions"],
                "complexity": technical_data["complexity"],
                "word_count": len(technical_data["text"].split()),
                "focus": "technical_depth"
            },
            "marketing": {
                "text": marketing_data["text"],
                "hooks": marketing_data["hooks"],
                "appeal": marketing_data["appeal"],
                "word_count": len(marketing_data["text"].split()),
                "focus": "value_proposition"
            }
        }
    
    def _analyze_content_for_views(self, content: str, title: str) -> Dict[str, Any]:
        """Analyze content to extract insights for enhanced view generation."""
        insights = {
            "domain": self._detect_content_domain(content, title),
            "complexity_level": self._assess_complexity_level(content),
            "key_themes": self._extract_key_themes(content),
            "business_value": self._identify_business_value(content),
            "technical_elements": self._identify_technical_elements(content),
            "audience_level": self._determine_audience_level(content),
            "action_items": self._extract_action_items(content),
            "impact_areas": self._identify_impact_areas(content)
        }
        return insights
    
    def _generate_tldr_view(self, base_summary: str, bullets: List[str], insights: Dict[str, Any]) -> str:
        """Generate a conversational yet comprehensive TLDR view (120-180 words)."""
        # Start with the most essential point
        essential_points = bullets[:2] if bullets else [base_summary[:100]]
        
        # Build conversational summary
        tldr_parts = []
        
        # Opening hook
        if insights.get("domain"):
            tldr_parts.append(f"This {insights['domain']} content covers")
        else:
            tldr_parts.append("In essence, this covers")
        
        # Core content
        if essential_points:
            core_content = " and ".join(essential_points[:2])
            tldr_parts.append(core_content.lower())
        
        # Key insight
        if insights.get("key_themes"):
            main_theme = insights["key_themes"][0] if insights["key_themes"] else ""
            if main_theme:
                tldr_parts.append(f"The main focus is on {main_theme.lower()}")
        
        # Impact or outcome
        if insights.get("impact_areas"):
            impact = insights["impact_areas"][0] if insights["impact_areas"] else ""
            if impact:
                tldr_parts.append(f"with implications for {impact.lower()}")
        
        tldr_text = ". ".join(tldr_parts) + "."
        
        # Ensure word count is within range (120-180 words)
        words = tldr_text.split()
        if len(words) < 120:
            # Add more detail from base summary
            additional_content = base_summary.split(". ")[1:3]
            tldr_text += " " + ". ".join(additional_content)
        elif len(words) > 180:
            # Trim to fit
            tldr_text = " ".join(words[:180]) + "..."
        
        return tldr_text
    
    def _generate_executive_view(self, base_summary: str, bullets: List[str], insights: Dict[str, Any]) -> str:
        """Generate strategic business perspective (200-300 words)."""
        exec_parts = []
        
        # Strategic overview
        if insights.get("business_value"):
            exec_parts.append(f"Strategic Overview: {insights['business_value'][0]}")
        else:
            exec_parts.append(f"Strategic Overview: {base_summary.split('.')[0]}.")
        
        # Key business impacts
        if bullets:
            business_bullets = [b for b in bullets if any(term in b.lower() for term in 
                              ['revenue', 'cost', 'efficiency', 'growth', 'market', 'competitive', 'roi', 'profit'])]
            if business_bullets:
                exec_parts.append(f"Business Impact: {'. '.join(business_bullets[:3])}.")
        
        # Decision-making insights
        if insights.get("action_items"):
            exec_parts.append(f"Key Decisions Required: {'. '.join(insights['action_items'][:2])}.")
        
        # Risk and opportunity assessment
        risk_opportunity = self._assess_risks_opportunities(base_summary, bullets)
        if risk_opportunity:
            exec_parts.append(f"Risk & Opportunity Assessment: {risk_opportunity}")
        
        # Implementation considerations
        if insights.get("complexity_level"):
            complexity = insights["complexity_level"]
            exec_parts.append(f"Implementation Complexity: {complexity} - requires appropriate resource allocation and timeline planning.")
        
        executive_text = " ".join(exec_parts)
        
        # Ensure word count is within range (200-300 words)
        words = executive_text.split()
        if len(words) < 200:
            executive_text += f" Additional considerations include market positioning, stakeholder alignment, and long-term strategic value creation."
        elif len(words) > 300:
            executive_text = " ".join(words[:300]) + "..."
        
        return executive_text
    
    def _generate_technical_view(self, base_summary: str, bullets: List[str], insights: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Generate detailed technical analysis (250-400 words)."""
        tech_elements = insights.get("technical_elements", [])
        
        # Technical overview
        tech_parts = []
        tech_parts.append(f"Technical Analysis: {base_summary}")
        
        # Technical specifications
        if tech_elements:
            tech_parts.append(f"Key Technical Components: {', '.join(tech_elements[:5])}.")
        
        # Implementation details
        tech_bullets = [b for b in bullets if any(term in b.lower() for term in 
                       ['implement', 'develop', 'configure', 'deploy', 'integrate', 'optimize'])]
        if tech_bullets:
            tech_parts.append(f"Implementation Details: {'. '.join(tech_bullets)}.")
        
        # Technical complexity assessment
        complexity_assessment = self._assess_technical_complexity(content, tech_elements)
        tech_parts.append(f"Complexity Assessment: {complexity_assessment}")
        
        # Performance and scalability considerations
        perf_considerations = self._identify_performance_considerations(content)
        if perf_considerations:
            tech_parts.append(f"Performance Considerations: {perf_considerations}")
        
        technical_text = " ".join(tech_parts)
        
        # Ensure word count is within range (250-400 words)
        words = technical_text.split()
        if len(words) < 250:
            technical_text += " Additional technical considerations include security protocols, data integrity measures, system integration requirements, and maintenance procedures."
        elif len(words) > 400:
            technical_text = " ".join(words[:400]) + "..."
        
        return {
            "text": technical_text,
            "mentions": tech_elements[:8],
            "complexity": insights.get("complexity_level", "moderate")
        }
    
    def _generate_marketing_view(self, base_summary: str, bullets: List[str], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benefit-driven narrative (180-250 words)."""
        marketing_parts = []
        
        # Value proposition
        value_prop = self._extract_value_proposition(base_summary, bullets)
        marketing_parts.append(f"Value Proposition: {value_prop}")
        
        # Key benefits
        benefits = self._identify_key_benefits(bullets, insights)
        if benefits:
            marketing_parts.append(f"Key Benefits: {'. '.join(benefits)}.")
        
        # Target audience appeal
        audience_appeal = self._generate_audience_appeal(insights)
        marketing_parts.append(f"Target Appeal: {audience_appeal}")
        
        # Competitive advantages
        competitive_advantages = self._identify_competitive_advantages(base_summary, bullets)
        if competitive_advantages:
            marketing_parts.append(f"Competitive Edge: {competitive_advantages}")
        
        # Call to action
        cta = self._generate_call_to_action(insights)
        marketing_parts.append(f"Next Steps: {cta}")
        
        marketing_text = " ".join(marketing_parts)
        
        # Generate compelling hooks
        hooks = self._generate_marketing_hooks(base_summary, bullets, insights)
        
        # Determine appeal type
        appeal_type = self._determine_appeal_type(insights)
        
        # Ensure word count is within range (180-250 words)
        words = marketing_text.split()
        if len(words) < 180:
            marketing_text += " This solution delivers measurable results and sustainable value for organizations seeking competitive advantage."
        elif len(words) > 250:
            marketing_text = " ".join(words[:250]) + "..."
        
        return {
            "text": marketing_text,
            "hooks": hooks,
            "appeal": appeal_type
        }

    def _extract_evidence_spans(self, content: str, keywords: List[str], entities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract source-grounded quotes for keywords and entities."""
        def find_quotes_for_term(term: str) -> List[Dict[str, Any]]:
            matches = []
            for m in re.finditer(re.escape(term), content, re.IGNORECASE):
                # Extract sentence containing match
                start = m.start()
                end = m.end()
                # Find sentence boundaries
                left = content.rfind('.', 0, start)
                right = content.find('.', end)
                left = 0 if left == -1 else left + 1
                right = len(content) if right == -1 else right + 1
                quote = content[left:right].strip()
                matches.append({"term": term, "quote": quote, "start": start, "end": end})
            return matches[:3]

        by_keyword = {}
        for kw in (keywords or [])[:10]:
            by_keyword[kw] = find_quotes_for_term(kw)

        # Flatten basic entities to terms
        entity_terms = []
        for category in ["people", "organizations", "locations"]:
            for ent in (entities.get(category, []) or [])[:10]:
                entity_terms.append(ent)
        by_entity = {}
        for ent in entity_terms:
            by_entity[ent] = find_quotes_for_term(ent)

        return {"by_keyword": by_keyword, "by_entity": by_entity}

    def _build_insight_details(
        self,
        content: str,
        key_insights: List[str],
        primary_keywords: List[str],
        sentiment_overall: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Attach confidence and rationale to insights using simple heuristics."""
        details = []
        polarity = 0.0
        subjectivity = 0.0
        label = "neutral"
        if isinstance(sentiment_overall, dict):
            polarity = float(sentiment_overall.get("polarity", 0.0))
            subjectivity = float(sentiment_overall.get("subjectivity", 0.0))
            label = sentiment_overall.get("label", "neutral")

        for ins in key_insights:
            # Confidence: boosted if insight contains primary keywords or strong sentiment
            keyword_boost = any(kw.lower() in ins.lower() for kw in (primary_keywords or [])[:5])
            sentiment_boost = abs(polarity) > 0.3
            base_conf = 0.6
            conf = base_conf + (0.15 if keyword_boost else 0) + (0.15 if sentiment_boost else 0)
            conf = max(0.0, min(0.95, conf))

            # Rationale: explain heuristic
            rationale_parts = []
            if keyword_boost:
                rationale_parts.append("Matches primary keywords from the content.")
            if sentiment_boost:
                rationale_parts.append(f"Aligned with detected {label} sentiment (polarity {polarity:.2f}).")
            if not rationale_parts:
                rationale_parts.append("Derived from content length, readability, and structure analysis.")

            # Evidence: pull a quote if possible
            quote = None
            for kw in (primary_keywords or [])[:5]:
                m = re.search(re.escape(kw), content, re.IGNORECASE)
                if m:
                    # Sentence containing the keyword
                    s = content.rfind('.', 0, m.start())
                    e = content.find('.', m.end())
                    s = 0 if s == -1 else s + 1
                    e = len(content) if e == -1 else e + 1
                    quote = content[s:e].strip()
                    break

            details.append({
                "text": ins,
                "confidence": round(conf, 2),
                "rationale": " ".join(rationale_parts),
                "evidence": quote
            })

        return details

    def _generate_recommendations(self, content: str) -> List[str]:
        """Generate recommendations for content improvement"""
        try:
            recommendations = []
            
            # Length recommendations
            word_count = len(content.split())
            if word_count < 300:
                recommendations.append("Consider expanding the content with more details and examples.")
            elif word_count > 2000:
                recommendations.append("Consider breaking this into smaller, more digestible sections.")
            
            # Readability recommendations
            readability = self._calculate_readability_score(content)
            if readability < 50:
                recommendations.append("Simplify sentence structure and use more common words to improve readability.")
            
            # Structure recommendations
            sentences = content.split('.')
            if len(sentences) < 3:
                recommendations.append("Add more detailed explanations and supporting information.")
            
            # Engagement recommendations
            if "?" not in content:
                recommendations.append("Consider adding questions to engage readers and encourage interaction.")
            
            if not recommendations:
                recommendations.append("The content is well-structured and engaging.")
            
            return recommendations[:5]
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Continue creating quality content."]

    def _assess_content_quality(self, content: str) -> Dict[str, Any]:
        """Assess the overall quality of the content"""
        try:
            word_count = len(content.split())
            sentence_count = len(content.split('.'))
            readability = self._calculate_readability_score(content)
            
            # Calculate quality score
            quality_score = 0
            
            # Length factor (optimal range: 300-1500 words)
            if 300 <= word_count <= 1500:
                quality_score += 25
            elif word_count > 100:
                quality_score += 15
            
            # Readability factor
            if readability > 60:
                quality_score += 25
            elif readability > 30:
                quality_score += 15
            
            # Structure factor
            if sentence_count > 5:
                quality_score += 25
            elif sentence_count > 2:
                quality_score += 15
            
            # Content diversity factor
            unique_words = len(set(content.lower().split()))
            diversity_ratio = unique_words / max(word_count, 1)
            if diversity_ratio > 0.5:
                quality_score += 25
            elif diversity_ratio > 0.3:
                quality_score += 15
            
            # Determine quality level
            if quality_score >= 80:
                quality_level = "Excellent"
            elif quality_score >= 60:
                quality_level = "Good"
            elif quality_score >= 40:
                quality_level = "Fair"
            else:
                quality_level = "Needs Improvement"
            
            return {
                "score": quality_score,
                "level": quality_level,
                "word_count": word_count,
                "readability": readability,
                "diversity_ratio": round(diversity_ratio * 100, 2)
            }
            
        except Exception as e:
            print(f"Error assessing content quality: {str(e)}")
            return {"score": 50, "level": "Fair", "word_count": 0, "readability": 50, "diversity_ratio": 50}

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score using Flesch Reading Ease formula"""
        try:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            words = content.split()
            syllables = sum([self._count_syllables(word) for word in words])
            
            if len(sentences) == 0 or len(words) == 0:
                return 50  # Default score
            
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            # Flesch Reading Ease formula
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score))
            
        except Exception as e:
            print(f"Error calculating readability: {str(e)}")
            return 50

    def _classify_content_type(self, content: str, title: str, url: str) -> str:
        """Classify the type of content"""
        content_lower = f"{title} {content}".lower()
        
        if any(word in content_lower for word in ["news", "breaking", "report", "journalist"]):
            return "news"
        elif any(word in content_lower for word in ["product", "buy", "price", "shop", "store"]):
            return "ecommerce"
        elif any(word in content_lower for word in ["blog", "opinion", "thoughts", "personal"]):
            return "blog"
        elif any(word in content_lower for word in ["company", "about us", "team", "contact"]):
            return "corporate"
        elif any(word in content_lower for word in ["tutorial", "how to", "guide", "step"]):
            return "educational"
        else:
            return "general"

    def _generate_user_friendly_summary(self, content: str, title: str = "", max_sentences: int = 8) -> Tuple[str, List[str]]:
        """Generate a comprehensive user-friendly summary with advanced abstractive and extractive techniques"""
        try:
            # Determine appropriate summary length based on content size
            content_length = len(content)
            if content_length > 8000:
                target_word_length = 500  # Comprehensive summary for extensive content
                max_sentences = min(15, len(sent_tokenize(content)) // 3)
            elif content_length > 4000:
                target_word_length = 350  # Detailed summary for substantial content
                max_sentences = min(12, len(sent_tokenize(content)) // 3)
            elif content_length > 2000:
                target_word_length = 250  # Standard summary for moderate content
                max_sentences = min(8, len(sent_tokenize(content)) // 4)
            else:
                target_word_length = 180  # Concise summary for short content
                max_sentences = min(6, len(sent_tokenize(content)) // 2)
            
            # Try comprehensive abstractive summarization first
            try:
                summary_result = self.advanced_analytics.comprehensive_abstractive_summary(
                    content, title, target_word_length
                )
                
                if summary_result and summary_result.get("text") and len(summary_result["text"].strip()) > 50:
                    abstractive_summary = summary_result["text"]
                    key_points = summary_result.get("key_points", [])
                    
                    # Ensure we have adequate key points
                    if len(key_points) < 4:
                        # Generate additional key points using extractive method
                        _, additional_points = self._enhanced_extractive_summary_with_points(content, title, max_sentences)
                        key_points.extend(additional_points[:8-len(key_points)])
                    
                    return abstractive_summary, key_points[:10]
            except Exception as e:
                logger.warning(f"Abstractive summarization failed: {e}")
            
            # Fallback to enhanced extractive summarization
            return self._enhanced_extractive_summary_with_points(content, title, max_sentences)
            
        except Exception as e:
            print(f"Summary generation error: {e}")
            # Enhanced fallback
            sentences = content.split('.')[:6]
            summary = '. '.join([s.strip() for s in sentences if len(s.strip()) > 10]) + '.'
            return summary, []
    
    def _enhanced_extractive_summary_with_points(self, content: str, title: str = "", max_sentences: int = 8) -> Tuple[str, List[str]]:
        """Enhanced extractive summarization with comprehensive key points generation"""
        try:
            # Use NLTK for better sentence tokenization
            sentences = sent_tokenize(content)
            if len(sentences) <= 3:
                return content, []
            
            # Enhanced sentence scoring with multiple factors
            sentence_scores = []
            words = content.lower().split()
            word_freq = Counter(words)
            
            # Remove stop words for better scoring
            filtered_words = [w for w in words if w not in self.stop_words and len(w) > 2]
            filtered_word_freq = Counter(filtered_words)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                    
                sentence_words = sentence.lower().split()
                filtered_sentence_words = [w for w in sentence_words if w not in self.stop_words and len(w) > 2]
                
                # Base score from word frequency
                base_score = sum(filtered_word_freq.get(word, 0) for word in filtered_sentence_words)
                
                # Position scoring (beginning and end are important)
                position_score = 0
                if i < len(sentences) * 0.15:  # First 15%
                    position_score = 3
                elif i > len(sentences) * 0.85:  # Last 15%
                    position_score = 2
                elif i < len(sentences) * 0.3:  # First 30%
                    position_score = 1
                
                # Length scoring (prefer medium-length sentences)
                length_score = 0
                word_count = len(sentence_words)
                if 15 <= word_count <= 35:
                    length_score = 2
                elif 8 <= word_count <= 50:
                    length_score = 1
                
                # Keyword scoring (sentences with important terms)
                keyword_score = 0
                important_terms = [
                    'conclusion', 'result', 'finding', 'important', 'significant', 'key', 'main', 'primary',
                    'therefore', 'however', 'moreover', 'furthermore', 'additionally', 'consequently',
                    'research', 'study', 'analysis', 'data', 'evidence', 'shows', 'indicates', 'suggests',
                    'impact', 'effect', 'benefit', 'advantage', 'challenge', 'problem', 'solution'
                ]
                for term in important_terms:
                    if term in sentence.lower():
                        keyword_score += 1
                
                # Title relevance scoring
                title_score = 0
                if title:
                    title_words = set(title.lower().split())
                    sentence_word_set = set(sentence_words)
                    common_words = title_words.intersection(sentence_word_set)
                    title_score = len(common_words) * 0.5
                
                # Numerical data scoring (sentences with numbers often contain facts)
                numerical_score = 0
                if re.search(r'\d+', sentence):
                    numerical_score = 1
                
                total_score = base_score + position_score + length_score + keyword_score + title_score + numerical_score
                sentence_scores.append((sentence.strip(), total_score, i))
            
            # Sort by score and select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            selected_sentences = sentence_scores[:max_sentences]
            
            # Sort selected sentences by original order to maintain flow
            selected_sentences.sort(key=lambda x: x[2])
            
            # Create comprehensive summary
            summary_sentences = [sent[0] for sent in selected_sentences]
            summary = ' '.join(summary_sentences)
            
            # Generate key points from selected sentences
            key_points = []
            for sentence, score, _ in sentence_scores[:min(10, len(sentence_scores))]:
                if len(sentence) > 20 and score > 2:  # Only high-scoring, substantial sentences
                    # Clean and format as bullet point
                    clean_sentence = sentence.strip()
                    if not clean_sentence.endswith('.'):
                        clean_sentence += '.'
                    key_points.append(clean_sentence)
            
            return summary, key_points[:8]  # Limit to 8 key points
            
        except Exception as e:
            print(f"Summary generation error: {e}")
            # Enhanced fallback
            sentences = content.split('.')[:6]
            summary = '. '.join([s.strip() for s in sentences if len(s.strip()) > 10]) + '.'
            return summary, []

    def _smart_extract_fallback(self, content: str, url: str) -> Dict[str, Any]:
        """Fallback smart extraction"""
        return {
            "headline": self._extract_headline(content),
            "key_facts": self._extract_key_facts(content),
            "contact_info": self._extract_contact_info(content)
        }

    def _extract_headline(self, content: str) -> str:
        """Extract main headline"""
        lines = content.split('\n')
        for line in lines:
            if len(line.strip()) > 10 and len(line.strip()) < 200:
                return line.strip()
        return ""

    def _extract_key_facts(self, content: str) -> List[str]:
        """Extract key facts"""
        facts = []
        sentences = content.split('.')
        for sentence in sentences[:5]:
            if len(sentence.strip()) > 20:
                facts.append(sentence.strip())
        return facts

    def _extract_contact_info(self, content: str) -> Dict[str, Any]:
        """Extract contact information"""
        return {
            "emails": self._extract_entities(content)["emails"],
            "phones": self._extract_entities(content)["phones"]
        }

    def _preprocess_content(self, content: str, url: str = "") -> str:
        """Enhanced preprocessing for diverse content formats"""
        if not content:
            return ""
        
        # Detect and preserve content format
        content_format = self._detect_content_format(content)
        
        # Preserve mathematical expressions
        content = self._preserve_mathematical_expressions(content)
        
        # Preserve bullet points and lists
        content = self._preserve_list_structures(content)
        
        # Preserve code blocks and technical content
        content = self._preserve_technical_content(content)
        
        # Clean up whitespace while preserving structure
        content = self._clean_whitespace_preserving_structure(content)
        
        # Store format information for later use
        self._content_format = content_format
        
        return content.strip()
    
    def _detect_content_format(self, content: str) -> Dict[str, Any]:
        """Detect the format and structure of input content"""
        format_info = {
            "mathematical_expressions": False,
            "bullet_points": False,
            "numbered_lists": False,
            "code_blocks": False,
            "tables": False,
            "headers": False,
            "statistical_notation": False,
            "structure_type": "paragraph"
        }
        
        # Detect mathematical expressions
        math_patterns = [
            r'\b\d+\s*[\+\-\*\/\^\=]\s*\d+',  # Basic math operations
            r'\b[a-zA-Z]\s*[\+\-\*\/\^\=]\s*[a-zA-Z0-9]',  # Algebraic expressions
            r'\b(?:sin|cos|tan|log|ln|sqrt|exp)\s*\(',  # Mathematical functions
            r'\b\d+\s*\^\s*\d+',  # Exponents
            r'\b\d+\s*\/\s*\d+',  # Fractions
            r'∑|∫|∂|√|π|α|β|γ|δ|θ|λ|μ|σ|φ|ψ|ω',  # Mathematical symbols
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                format_info["mathematical_expressions"] = True
                break
        
        # Detect statistical notation
        stats_patterns = [
            r'\bp\s*[<>=]\s*0\.\d+',  # P-values
            r'\br\s*=\s*0\.\d+',  # Correlation coefficients
            r'\b[±]\s*\d+',  # Plus/minus notation
            r'\b\d+\.\d+\s*\[.*\]',  # Confidence intervals
            r'\b(?:mean|median|std|variance|correlation)\s*[:=]',  # Statistical terms
        ]
        
        for pattern in stats_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                format_info["statistical_notation"] = True
                break
        
        # Detect bullet points and lists
        if re.search(r'^\s*[•·▪▫‣⁃]\s+', content, re.MULTILINE):
            format_info["bullet_points"] = True
            format_info["structure_type"] = "bullet_list"
        
        if re.search(r'^\s*\d+[\.\)]\s+', content, re.MULTILINE):
            format_info["numbered_lists"] = True
            format_info["structure_type"] = "numbered_list"
        
        # Detect code blocks
        if re.search(r'```|`[^`]+`|def\s+\w+|class\s+\w+|function\s+\w+', content):
            format_info["code_blocks"] = True
        
        # Detect tables
        if re.search(r'\|.*\|.*\|', content) or re.search(r'\t.*\t.*\t', content):
            format_info["tables"] = True
        
        # Detect headers
        header_patterns = [
            r'^\s*#+\s+.+$',  # Markdown headers with optional leading whitespace
            r'^\s*[A-Z][A-Z\s]+\s*$',  # ALL CAPS headers
            r'^\s*[A-Z][^a-z\n]*\s*$',  # Title case headers
            r'^\s*[A-Z][a-zA-Z\s]+:?\s*$',  # Section headers with optional colon
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, content, re.MULTILINE):
                format_info["headers"] = True
                break
        
        return format_info
    
    def _preserve_mathematical_expressions(self, content: str) -> str:
        """Preserve mathematical expressions and formulas"""
        # Protect mathematical operators and symbols
        math_replacements = {
            ' + ': ' PLUS ',
            ' - ': ' MINUS ',
            ' * ': ' MULTIPLY ',
            ' / ': ' DIVIDE ',
            ' ^ ': ' POWER ',
            ' = ': ' EQUALS ',
            '²': ' SQUARED ',
            '³': ' CUBED ',
            '√': ' SQRT ',
            'π': ' PI ',
            '∑': ' SUM ',
            '∫': ' INTEGRAL ',
            '∂': ' PARTIAL ',
            '≤': ' LESS_EQUAL ',
            '≥': ' GREATER_EQUAL ',
            '≠': ' NOT_EQUAL ',
            '±': ' PLUS_MINUS '
        }
        
        # Store original expressions for restoration
        self._math_expressions = {}
        
        # Replace mathematical expressions with placeholders
        for original, placeholder in math_replacements.items():
            if original in content:
                content = content.replace(original, placeholder)
        
        # Preserve complex mathematical expressions
        math_pattern = r'(\b\w+\s*[=\+\-\*\/\^]\s*[\w\d\(\)\+\-\*\/\^\.]+)'
        matches = re.findall(math_pattern, content)
        for i, match in enumerate(matches):
            placeholder = f"MATH_EXPR_{i}"
            self._math_expressions[placeholder] = match
            content = content.replace(match, placeholder, 1)
        
        return content
    
    def _preserve_list_structures(self, content: str) -> str:
        """Preserve bullet points and numbered lists"""
        # Preserve bullet points
        content = re.sub(r'^(\s*)[•·▪▫‣⁃]\s+', r'\1BULLET_POINT ', content, flags=re.MULTILINE)
        
        # Preserve numbered lists
        content = re.sub(r'^(\s*)(\d+)[\.\)]\s+', r'\1NUMBER_\2_POINT ', content, flags=re.MULTILINE)
        
        # Preserve lettered lists
        content = re.sub(r'^(\s*)([a-zA-Z])[\.\)]\s+', r'\1LETTER_\2_POINT ', content, flags=re.MULTILINE)
        
        return content
    
    def _preserve_technical_content(self, content: str) -> str:
        """Preserve code blocks and technical formatting"""
        # Preserve code blocks
        content = re.sub(r'```([^`]+)```', r'CODE_BLOCK_\1_END_CODE', content, flags=re.DOTALL)
        
        # Preserve inline code
        content = re.sub(r'`([^`]+)`', r'INLINE_CODE_\1_END_INLINE', content)
        
        # Preserve technical terms with special characters
        content = re.sub(r'([A-Z][a-z]*[A-Z][a-zA-Z]*)', r'TECH_TERM_\1', content)  # CamelCase
        content = re.sub(r'([a-z]+_[a-z_]+)', r'SNAKE_CASE_\1', content)  # snake_case
        
        return content
    
    def _clean_whitespace_preserving_structure(self, content: str) -> str:
        """Clean whitespace while preserving important structure"""
        # Preserve line breaks in lists and structured content
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Keep structure indicators
            if any(indicator in line for indicator in ['BULLET_POINT', 'NUMBER_', 'LETTER_', 'CODE_BLOCK']):
                cleaned_lines.append(line.strip())
            else:
                # Regular line cleaning
                cleaned_line = re.sub(r'\s+', ' ', line).strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _restore_preserved_content(self, text: str) -> str:
        """Restore preserved mathematical and structural elements"""
        if not hasattr(self, '_content_format'):
            return text
        
        # Restore mathematical expressions
        if hasattr(self, '_math_expressions'):
            for placeholder, original in self._math_expressions.items():
                text = text.replace(placeholder, original)
        
        # Restore mathematical operators
        math_restorations = {
            ' PLUS ': ' + ',
            ' MINUS ': ' - ',
            ' MULTIPLY ': ' * ',
            ' DIVIDE ': ' / ',
            ' POWER ': ' ^ ',
            ' EQUALS ': ' = ',
            ' SQUARED ': '²',
            ' CUBED ': '³',
            ' SQRT ': '√',
            ' PI ': 'π',
            ' SUM ': '∑',
            ' INTEGRAL ': '∫',
            ' PARTIAL ': '∂',
            ' LESS_EQUAL ': '≤',
            ' GREATER_EQUAL ': '≥',
            ' NOT_EQUAL ': '≠',
            ' PLUS_MINUS ': '±'
        }
        
        for placeholder, original in math_restorations.items():
            text = text.replace(placeholder, original)
        
        # Restore list structures
        text = re.sub(r'BULLET_POINT ', '• ', text)
        text = re.sub(r'NUMBER_(\d+)_POINT ', r'\1. ', text)
        text = re.sub(r'LETTER_([a-zA-Z])_POINT ', r'\1. ', text)
        
        # Restore technical content
        text = re.sub(r'CODE_BLOCK_([^_]+)_END_CODE', r'```\1```', text)
        text = re.sub(r'INLINE_CODE_([^_]+)_END_INLINE', r'`\1`', text)
        text = re.sub(r'TECH_TERM_([A-Za-z]+)', r'\1', text)
        text = re.sub(r'SNAKE_CASE_([a-z_]+)', r'\1', text)
        
        return text

    async def _openai_analysis(self, content: str, title: str, url: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """OpenAI analysis (placeholder for compatibility)"""
        return {
            "summary": self._generate_user_friendly_summary(content, title)[0],
            "sentiment": self._analyze_sentiment(content),
            "keywords": self._extract_keywords(content),
            "entities": self._extract_entities(content),
            "smart_extraction": self._smart_extract_fallback(content, url),
            "content_type": self._classify_content_type(content, title, url),
            "language": self._detect_language(content),
            "readability_score": self._calculate_readability(content)
        }
    
    def _generate_format_instructions(self, content_format: dict) -> str:
        """Generate format-specific instructions for the LLM based on detected content format."""
        instructions = []
        
        if content_format.get('mathematical_expressions', False):
            instructions.append(
                "MATHEMATICAL CONTENT DETECTED:\n"
                "- Preserve ALL mathematical expressions, formulas, equations, and symbols exactly as written\n"
                "- Maintain proper mathematical notation (superscripts, subscripts, fractions, etc.)\n"
                "- Include mathematical context and explanations in the summary\n"
                "- Use LaTeX notation where appropriate for complex formulas"
            )
        
        if content_format.get('statistical_notation', False):
            instructions.append(
                "STATISTICAL CONTENT DETECTED:\n"
                "- Preserve statistical notation, p-values, confidence intervals, and correlation coefficients\n"
                "- Maintain precision in statistical reporting and significance levels\n"
                "- Include statistical context and methodology in the summary\n"
                "- Ensure accuracy in statistical interpretations and conclusions"
            )
        
        if content_format.get('bullet_points', False) or content_format.get('numbered_lists', False):
            instructions.append(
                "LIST/BULLET POINT CONTENT DETECTED:\n"
                "- Preserve the hierarchical structure of lists and bullet points\n"
                "- Maintain numbering sequences and bullet point styles\n"
                "- Keep the logical organization and grouping of list items\n"
                "- Summarize list content while preserving the structured format"
            )
        
        if content_format.get('code_blocks', False):
            instructions.append(
                "CODE CONTENT DETECTED:\n"
                "- Preserve code blocks, syntax, and programming language specifics\n"
                "- Maintain proper indentation and code structure\n"
                "- Include technical explanations and code functionality in summary\n"
                "- Preserve variable names, function names, and technical terminology"
            )
        
        if content_format.get('headers', False):
            instructions.append(
                "STRUCTURED CONTENT WITH HEADERS DETECTED:\n"
                "- Preserve document structure and section hierarchy\n"
                "- Maintain the logical flow and organization of sections\n"
                "- Include section relationships and document outline in summary\n"
                "- Ensure proper heading levels and structural coherence"
            )
        
        if content_format.get('tables', False):
            instructions.append(
                "TABLE CONTENT DETECTED:\n"
                "- Preserve tabular data structure and relationships\n"
                "- Maintain data accuracy and numerical precision\n"
                "- Include table headers and column relationships in summary\n"
                "- Present data insights while preserving original format"
            )
        
        if not instructions:
            instructions.append(
                "STANDARD TEXT CONTENT:\n"
                "- Focus on clarity, coherence, and comprehensive coverage\n"
                "- Maintain the original tone and style where appropriate\n"
                "- Ensure grammatical excellence and professional presentation"
            )
        
        return "\n\n".join(instructions)
    
    # Helper methods for enhanced summary views
    def _detect_content_domain(self, content: str, title: str) -> str:
        """Detect the domain/field of the content."""
        domains = {
            "technology": ["software", "programming", "AI", "machine learning", "database", "API", "cloud", "cybersecurity"],
            "business": ["strategy", "management", "revenue", "profit", "market", "customer", "sales", "ROI"],
            "healthcare": ["medical", "patient", "treatment", "diagnosis", "clinical", "pharmaceutical", "therapy"],
            "finance": ["investment", "financial", "banking", "trading", "portfolio", "risk", "capital"],
            "education": ["learning", "student", "curriculum", "academic", "research", "university", "training"],
            "science": ["research", "experiment", "hypothesis", "data", "analysis", "methodology", "findings"]
        }
        
        text_to_analyze = (title + " " + content).lower()
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in text_to_analyze)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else "general"
    
    def _assess_complexity_level(self, content: str) -> str:
        """Assess the complexity level of the content."""
        # Simple heuristics for complexity assessment
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', content))  # Acronyms
        complex_words = len([w for w in content.split() if len(w) > 8])
        
        complexity_score = (avg_sentence_length / 20) + (technical_terms / 10) + (complex_words / word_count * 100)
        
        if complexity_score > 3:
            return "high"
        elif complexity_score > 1.5:
            return "moderate"
        else:
            return "low"
    
    def _extract_key_themes(self, content: str) -> List[str]:
        """Extract key themes from content."""
        # Simple keyword extraction for themes
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            if word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'were', 'said']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5] if freq > 1]
    
    def _identify_business_value(self, content: str) -> List[str]:
        """Identify business value propositions in content."""
        value_indicators = [
            "increase revenue", "reduce costs", "improve efficiency", "competitive advantage",
            "market opportunity", "customer satisfaction", "operational excellence", "innovation",
            "scalability", "productivity", "profitability", "growth potential"
        ]
        
        found_values = []
        content_lower = content.lower()
        for indicator in value_indicators:
            if indicator in content_lower:
                found_values.append(indicator.title())
        
        return found_values[:3]
    
    def _identify_technical_elements(self, content: str) -> List[str]:
        """Identify technical elements in content."""
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\.\w+\b',   # Dotted notation (APIs, domains)
            r'\b\d+\.\d+\b',   # Version numbers
            r'\b[a-zA-Z]+://\S+\b',  # URLs
            r'\b[a-zA-Z_]+\(\)',  # Function calls
        ]
        
        technical_elements = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, content)
            technical_elements.extend(matches)
        
        # Add common technical terms
        tech_keywords = ["API", "database", "algorithm", "framework", "protocol", "interface", 
                        "architecture", "deployment", "integration", "optimization"]
        
        for keyword in tech_keywords:
            if keyword.lower() in content.lower():
                technical_elements.append(keyword)
        
        return list(set(technical_elements))[:10]
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the original content structure for format preservation."""
        structure = {
            "total_lines": len(content.split('\n')),
            "paragraphs": len([p for p in content.split('\n\n') if p.strip()]),
            "sentences": len([s for s in content.split('.') if s.strip()]),
            "has_headers": bool(re.search(r'^#+\s', content, re.MULTILINE)),
            "has_bullet_points": bool(re.search(r'^\s*[-*•]\s', content, re.MULTILINE)),
            "has_numbered_lists": bool(re.search(r'^\s*\d+\.\s', content, re.MULTILINE)),
            "has_code_blocks": bool(re.search(r'```|`[^`]+`', content)),
            "has_math_expressions": bool(re.search(r'\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]', content)),
            "has_tables": bool(re.search(r'\|.*\|', content)),
            "has_urls": bool(re.search(r'https?://\S+', content)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)),
            "indentation_levels": len(set(re.findall(r'^(\s*)', content, re.MULTILINE))),
            "special_formatting": {
                "bold": bool(re.search(r'\*\*[^*]+\*\*|__[^_]+__', content)),
                "italic": bool(re.search(r'\*[^*]+\*|_[^_]+_', content)),
                "quotes": bool(re.search(r'^>\s', content, re.MULTILINE)),
                "code_inline": bool(re.search(r'`[^`]+`', content))
            }
        }
        return structure
    
    def _determine_audience_level(self, content: str) -> str:
        """Determine the target audience level."""
        technical_density = len(re.findall(r'\b[A-Z]{2,}\b', content)) / len(content.split())
        jargon_words = ["implement", "configure", "optimize", "integrate", "deploy", "architecture"]
        jargon_count = sum(1 for word in jargon_words if word in content.lower())
        
        if technical_density > 0.05 or jargon_count > 3:
            return "technical"
        elif any(word in content.lower() for word in ["strategy", "business", "market", "revenue"]):
            return "business"
        else:
            return "general"
    
    def _extract_action_items(self, content: str) -> List[str]:
        """Extract actionable items from content."""
        action_patterns = [
            r'should \w+',
            r'need to \w+',
            r'must \w+',
            r'recommend \w+',
            r'suggest \w+',
            r'implement \w+',
            r'develop \w+',
            r'create \w+',
            r'establish \w+',
            r'consider \w+'
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            actions.extend(matches)
        
        return list(set(actions))[:5]
    
    def _identify_impact_areas(self, content: str) -> List[str]:
        """Identify areas of impact mentioned in content."""
        impact_areas = [
            "operations", "technology", "business", "customers", "market", "revenue",
            "efficiency", "productivity", "quality", "security", "compliance", "innovation"
        ]
        
        found_impacts = []
        content_lower = content.lower()
        for area in impact_areas:
            if area in content_lower:
                found_impacts.append(area)
        
        return found_impacts[:4]
    
    def _assess_risks_opportunities(self, base_summary: str, bullets: List[str]) -> str:
        """Assess risks and opportunities from content."""
        risk_keywords = ["risk", "challenge", "threat", "concern", "issue", "problem"]
        opportunity_keywords = ["opportunity", "potential", "benefit", "advantage", "growth", "improvement"]
        
        text_to_analyze = (base_summary + " " + " ".join(bullets)).lower()
        
        risks = sum(1 for keyword in risk_keywords if keyword in text_to_analyze)
        opportunities = sum(1 for keyword in opportunity_keywords if keyword in text_to_analyze)
        
        if opportunities > risks:
            return "High opportunity potential with manageable risk profile"
        elif risks > opportunities:
            return "Significant risks identified requiring mitigation strategies"
        else:
            return "Balanced risk-opportunity profile requiring careful evaluation"
    
    def _assess_technical_complexity(self, content: str, tech_elements: List[str]) -> str:
        """Assess technical complexity of implementation."""
        complexity_indicators = len(tech_elements)
        integration_mentions = sum(1 for word in ["integrate", "deploy", "configure", "implement"] 
                                 if word in content.lower())
        
        total_complexity = complexity_indicators + integration_mentions
        
        if total_complexity > 10:
            return "High complexity requiring specialized expertise and extended timeline"
        elif total_complexity > 5:
            return "Moderate complexity with standard implementation requirements"
        else:
            return "Low complexity suitable for rapid deployment"
    
    def _identify_performance_considerations(self, content: str) -> str:
        """Identify performance and scalability considerations."""
        perf_keywords = ["performance", "scalability", "speed", "efficiency", "optimization", 
                        "latency", "throughput", "capacity", "load", "response time"]
        
        found_keywords = [kw for kw in perf_keywords if kw in content.lower()]
        
        if found_keywords:
            return f"Key performance factors include {', '.join(found_keywords[:3])}"
        else:
            return "Standard performance requirements apply"
    
    def _extract_value_proposition(self, base_summary: str, bullets: List[str]) -> str:
        """Extract the core value proposition."""
        # Look for value-indicating phrases
        value_phrases = ["provides", "enables", "delivers", "offers", "improves", "increases", "reduces"]
        
        for bullet in bullets:
            for phrase in value_phrases:
                if phrase in bullet.lower():
                    return bullet
        
        # Fallback to first sentence of summary
        return base_summary.split('.')[0] if base_summary else "Delivers comprehensive solution"
    
    def _identify_key_benefits(self, bullets: List[str], insights: Dict[str, Any]) -> List[str]:
        """Identify key benefits from content."""
        benefit_keywords = ["benefit", "advantage", "improvement", "enhancement", "optimization", 
                           "efficiency", "productivity", "savings", "growth", "value"]
        
        benefits = []
        for bullet in bullets:
            if any(keyword in bullet.lower() for keyword in benefit_keywords):
                benefits.append(bullet)
        
        # Add domain-specific benefits
        if insights.get("business_value"):
            benefits.extend(insights["business_value"][:2])
        
        return benefits[:4]
    
    def _generate_audience_appeal(self, insights: Dict[str, Any]) -> str:
        """Generate audience-specific appeal."""
        audience = insights.get("audience_level", "general")
        domain = insights.get("domain", "general")
        
        appeals = {
            "technical": f"Ideal for technical professionals seeking {domain} solutions",
            "business": f"Perfect for business leaders driving {domain} transformation",
            "general": f"Accessible to all stakeholders interested in {domain} advancement"
        }
        
        return appeals.get(audience, appeals["general"])
    
    def _identify_competitive_advantages(self, base_summary: str, bullets: List[str]) -> str:
        """Identify competitive advantages."""
        advantage_keywords = ["unique", "innovative", "advanced", "superior", "leading", 
                             "cutting-edge", "state-of-the-art", "breakthrough", "revolutionary"]
        
        text_to_analyze = (base_summary + " " + " ".join(bullets)).lower()
        
        found_advantages = [kw for kw in advantage_keywords if kw in text_to_analyze]
        
        if found_advantages:
            return f"Features {', '.join(found_advantages[:2])} capabilities that differentiate from competitors"
        else:
            return "Provides competitive positioning through comprehensive feature set"
    
    def _generate_call_to_action(self, insights: Dict[str, Any]) -> str:
        """Generate appropriate call to action."""
        complexity = insights.get("complexity_level", "moderate")
        domain = insights.get("domain", "general")
        
        ctas = {
            "high": f"Engage specialists for detailed {domain} implementation planning",
            "moderate": f"Schedule consultation to explore {domain} opportunities",
            "low": f"Begin immediate evaluation for {domain} adoption"
        }
        
        return ctas.get(complexity, ctas["moderate"])
    
    def _generate_marketing_hooks(self, base_summary: str, bullets: List[str], insights: Dict[str, Any]) -> List[str]:
        """Generate compelling marketing hooks."""
        hooks = []
        
        # Value-based hook
        if insights.get("business_value"):
            hooks.append(f"Unlock {insights['business_value'][0].lower()} potential")
        
        # Problem-solution hook
        if bullets:
            first_bullet = bullets[0]
            hooks.append(f"Solve {first_bullet.lower()[:50]}...")
        
        # Innovation hook
        domain = insights.get("domain", "business")
        hooks.append(f"Transform your {domain} operations")
        
        return hooks[:3]
    
    def _determine_appeal_type(self, insights: Dict[str, Any]) -> str:
        """Determine the type of marketing appeal."""
        if insights.get("business_value"):
            return "value_driven"
        elif insights.get("technical_elements"):
            return "innovation_focused"
        elif insights.get("complexity_level") == "low":
            return "accessibility_focused"
        else:
            return "solution_oriented"

    # ==================== PROFESSIONAL-GRADE GAP DETECTION SYSTEM ====================
    
    def detect_content_gaps(self, content: str, title: str = "", domain: str = "") -> Dict[str, Any]:
        """
        Advanced gap detection system that identifies missing information, 
        logical inconsistencies, and content completeness issues.
        """
        try:
            gaps_analysis = {
                "structural_gaps": self._detect_structural_gaps(content),
                "logical_gaps": self._detect_logical_inconsistencies(content),
                "information_gaps": self._detect_information_gaps(content, title, domain),
                "evidence_gaps": self._detect_evidence_gaps(content),
                "completeness_score": self._calculate_completeness_score(content),
                "gap_severity": "low",
                "recommendations": []
            }
            
            # Calculate overall gap severity
            total_gaps = (len(gaps_analysis["structural_gaps"]) + 
                         len(gaps_analysis["logical_gaps"]) + 
                         len(gaps_analysis["information_gaps"]) + 
                         len(gaps_analysis["evidence_gaps"]))
            
            if total_gaps >= 8:
                gaps_analysis["gap_severity"] = "critical"
            elif total_gaps >= 5:
                gaps_analysis["gap_severity"] = "high"
            elif total_gaps >= 3:
                gaps_analysis["gap_severity"] = "medium"
            else:
                gaps_analysis["gap_severity"] = "low"
            
            # Generate specific recommendations
            gaps_analysis["recommendations"] = self._generate_gap_recommendations(gaps_analysis)
            
            return gaps_analysis
            
        except Exception as e:
            print(f"Error in gap detection: {str(e)}")
            return {
                "structural_gaps": [],
                "logical_gaps": [],
                "information_gaps": [],
                "evidence_gaps": [],
                "completeness_score": 50,
                "gap_severity": "unknown",
                "recommendations": ["Unable to perform gap analysis due to processing error"]
            }
    
    def _detect_structural_gaps(self, content: str) -> List[Dict[str, Any]]:
        """Detect structural gaps in content organization."""
        gaps = []
        
        # Check for introduction/conclusion
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) > 5:
            intro_indicators = ['introduction', 'overview', 'begin', 'start', 'first']
            conclusion_indicators = ['conclusion', 'summary', 'finally', 'end', 'last']
            
            has_intro = any(indicator in content.lower()[:200] for indicator in intro_indicators)
            has_conclusion = any(indicator in content.lower()[-200:] for indicator in conclusion_indicators)
            
            if not has_intro:
                gaps.append({
                    "type": "missing_introduction",
                    "severity": "medium",
                    "description": "Content lacks a clear introduction or overview section",
                    "suggestion": "Add an introductory paragraph that outlines the main topics"
                })
            
            if not has_conclusion:
                gaps.append({
                    "type": "missing_conclusion",
                    "severity": "medium", 
                    "description": "Content lacks a proper conclusion or summary",
                    "suggestion": "Add a concluding section that summarizes key points"
                })
        
        # Check for logical flow and transitions
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'additionally']
        transition_count = sum(1 for word in transition_words if word in content.lower())
        
        if len(sentences) > 10 and transition_count < 2:
            gaps.append({
                "type": "poor_transitions",
                "severity": "low",
                "description": "Content lacks transitional phrases for smooth flow",
                "suggestion": "Add transitional words and phrases to improve readability"
            })
        
        # Check for section headers/organization
        header_patterns = [r'^#+\s+.+$', r'^[A-Z][A-Za-z\s]+:$', r'^\d+\.\s+[A-Z]']
        has_headers = any(re.search(pattern, content, re.MULTILINE) for pattern in header_patterns)
        
        if len(content.split()) > 500 and not has_headers:
            gaps.append({
                "type": "missing_structure",
                "severity": "medium",
                "description": "Long content lacks clear section headers or organization",
                "suggestion": "Add section headers to break up content into logical segments"
            })
        
        return gaps
    
    def _detect_logical_inconsistencies(self, content: str) -> List[Dict[str, Any]]:
        """Detect logical inconsistencies and contradictions."""
        gaps = []
        
        # Check for contradictory statements
        contradiction_patterns = [
            (r'\b(always|never|all|none|every)\b', r'\b(sometimes|some|few|rarely)\b'),
            (r'\b(increase|rise|grow|improve)\b', r'\b(decrease|fall|decline|worsen)\b'),
            (r'\b(positive|beneficial|good)\b', r'\b(negative|harmful|bad)\b')
        ]
        
        for positive_pattern, negative_pattern in contradiction_patterns:
            positive_matches = re.findall(positive_pattern, content.lower())
            negative_matches = re.findall(negative_pattern, content.lower())
            
            if positive_matches and negative_matches and len(positive_matches) > 2 and len(negative_matches) > 2:
                gaps.append({
                    "type": "potential_contradiction",
                    "severity": "medium",
                    "description": f"Content contains potentially contradictory statements about the same topic",
                    "suggestion": "Review content for logical consistency and clarify any contradictions"
                })
                break
        
        # Check for unsupported claims
        claim_indicators = ['studies show', 'research indicates', 'experts say', 'data reveals']
        citation_indicators = ['according to', 'source:', 'reference:', '(', '[']
        
        claims = sum(1 for indicator in claim_indicators if indicator in content.lower())
        citations = sum(1 for indicator in citation_indicators if indicator in content.lower())
        
        if claims > citations + 1:
            gaps.append({
                "type": "unsupported_claims",
                "severity": "high",
                "description": "Content makes claims without adequate supporting evidence or citations",
                "suggestion": "Add sources, references, or evidence to support factual claims"
            })
        
        return gaps
    
    def _detect_information_gaps(self, content: str, title: str = "", domain: str = "") -> List[Dict[str, Any]]:
        """Detect missing information based on content type and domain."""
        gaps = []
        
        # Domain-specific gap detection
        domain_requirements = {
            "technology": ["implementation", "requirements", "benefits", "limitations"],
            "business": ["objectives", "strategy", "metrics", "timeline"],
            "science": ["methodology", "results", "discussion", "conclusion"],
            "healthcare": ["symptoms", "treatment", "risks", "outcomes"],
            "education": ["objectives", "methods", "assessment", "resources"]
        }
        
        if domain and domain in domain_requirements:
            required_elements = domain_requirements[domain]
            missing_elements = []
            
            for element in required_elements:
                if element not in content.lower():
                    missing_elements.append(element)
            
            if missing_elements:
                gaps.append({
                    "type": "domain_specific_gaps",
                    "severity": "medium",
                    "description": f"Missing key {domain} elements: {', '.join(missing_elements)}",
                    "suggestion": f"Consider adding information about: {', '.join(missing_elements)}"
                })
        
        # Check for missing context
        if len(content.split()) > 200:
            context_indicators = ['background', 'context', 'history', 'overview']
            has_context = any(indicator in content.lower() for indicator in context_indicators)
            
            if not has_context:
                gaps.append({
                    "type": "missing_context",
                    "severity": "medium",
                    "description": "Content lacks sufficient background or contextual information",
                    "suggestion": "Add background information to help readers understand the context"
                })
        
        # Check for missing examples or illustrations
        example_indicators = ['example', 'instance', 'case study', 'illustration', 'for example']
        has_examples = any(indicator in content.lower() for indicator in example_indicators)
        
        if len(content.split()) > 300 and not has_examples:
            gaps.append({
                "type": "missing_examples",
                "severity": "low",
                "description": "Content could benefit from concrete examples or illustrations",
                "suggestion": "Add specific examples to illustrate key points"
            })
        
        return gaps
    
    def _detect_evidence_gaps(self, content: str) -> List[Dict[str, Any]]:
        """Detect gaps in evidence and supporting information."""
        gaps = []
        
        # Check for statistical claims without data
        stat_patterns = [r'\d+%', r'\d+\s*(percent|percentage)', r'majority', r'most', r'significant']
        has_stats = any(re.search(pattern, content.lower()) for pattern in stat_patterns)
        
        source_patterns = [r'study', r'research', r'survey', r'report', r'data', r'statistics']
        has_sources = any(re.search(pattern, content.lower()) for pattern in source_patterns)
        
        if has_stats and not has_sources:
            gaps.append({
                "type": "unsubstantiated_statistics",
                "severity": "high",
                "description": "Content includes statistical claims without citing sources",
                "suggestion": "Provide sources for statistical information and data claims"
            })
        
        # Check for expert opinions without attribution
        opinion_indicators = ['experts believe', 'specialists say', 'professionals recommend']
        attribution_indicators = ['dr.', 'professor', 'according to', 'says']
        
        has_opinions = any(indicator in content.lower() for indicator in opinion_indicators)
        has_attribution = any(indicator in content.lower() for indicator in attribution_indicators)
        
        if has_opinions and not has_attribution:
            gaps.append({
                "type": "unattributed_opinions",
                "severity": "medium",
                "description": "Content references expert opinions without proper attribution",
                "suggestion": "Identify and attribute expert opinions to specific individuals or organizations"
            })
        
        return gaps
    
    def _calculate_completeness_score(self, content: str) -> int:
        """Calculate overall content completeness score (0-100)."""
        score = 0
        
        # Length completeness (20 points)
        word_count = len(content.split())
        if word_count >= 500:
            score += 20
        elif word_count >= 300:
            score += 15
        elif word_count >= 150:
            score += 10
        elif word_count >= 50:
            score += 5
        
        # Structure completeness (20 points)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) >= 10:
            score += 20
        elif len(sentences) >= 5:
            score += 15
        elif len(sentences) >= 3:
            score += 10
        
        # Content depth (20 points)
        depth_indicators = ['because', 'therefore', 'however', 'furthermore', 'analysis', 'detailed']
        depth_count = sum(1 for indicator in depth_indicators if indicator in content.lower())
        if depth_count >= 5:
            score += 20
        elif depth_count >= 3:
            score += 15
        elif depth_count >= 1:
            score += 10
        
        # Evidence presence (20 points)
        evidence_indicators = ['study', 'research', 'data', 'source', 'reference', 'according to']
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content.lower())
        if evidence_count >= 3:
            score += 20
        elif evidence_count >= 2:
            score += 15
        elif evidence_count >= 1:
            score += 10
        
        # Clarity and organization (20 points)
        clarity_indicators = ['first', 'second', 'finally', 'in conclusion', 'overview', 'summary']
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in content.lower())
        if clarity_count >= 3:
            score += 20
        elif clarity_count >= 2:
            score += 15
        elif clarity_count >= 1:
            score += 10
        
        return min(score, 100)
    
    def _generate_gap_recommendations(self, gaps_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on detected gaps."""
        recommendations = []
        
        # Priority recommendations based on severity
        high_severity_gaps = []
        medium_severity_gaps = []
        low_severity_gaps = []
        
        for gap_category in ["structural_gaps", "logical_gaps", "information_gaps", "evidence_gaps"]:
            for gap in gaps_analysis.get(gap_category, []):
                if gap.get("severity") == "high":
                    high_severity_gaps.append(gap)
                elif gap.get("severity") == "medium":
                    medium_severity_gaps.append(gap)
                else:
                    low_severity_gaps.append(gap)
        
        # Add high priority recommendations first
        for gap in high_severity_gaps[:3]:
            recommendations.append(f"🔴 CRITICAL: {gap.get('suggestion', 'Address this gap')}")
        
        for gap in medium_severity_gaps[:3]:
            recommendations.append(f"🟡 IMPORTANT: {gap.get('suggestion', 'Consider addressing this gap')}")
        
        for gap in low_severity_gaps[:2]:
            recommendations.append(f"🟢 ENHANCEMENT: {gap.get('suggestion', 'Optional improvement')}")
        
        # Add general recommendations based on completeness score
        completeness = gaps_analysis.get("completeness_score", 50)
        if completeness < 60:
            recommendations.append("📈 OVERALL: Expand content depth and add more supporting details")
        if completeness < 40:
            recommendations.append("📝 STRUCTURE: Improve content organization and logical flow")
        
        return recommendations[:8]  # Limit to top 8 recommendations

    def enterprise_quality_assessment(self, content: str, title: str = "", url: str = "") -> Dict[str, Any]:
        """
        Comprehensive enterprise-grade quality assessment with 15+ metrics
        """
        try:
            assessment = {
                "overall_score": 0,
                "grade": "F",
                "metrics": {},
                "professional_indicators": {},
                "improvement_areas": [],
                "strengths": [],
                "industry_benchmarks": {},
                "detailed_analysis": {}
            }
            
            # Core content metrics
            content_metrics = self._assess_content_metrics(content, title)
            assessment["metrics"].update(content_metrics)
            
            # Professional presentation metrics
            presentation_metrics = self._assess_presentation_quality(content)
            assessment["metrics"].update(presentation_metrics)
            
            # Authority and credibility metrics
            authority_metrics = self._assess_authority_indicators(content, url)
            assessment["metrics"].update(authority_metrics)
            
            # Evidence and support metrics
            evidence_metrics = self._assess_evidence_quality(content)
            assessment["metrics"].update(evidence_metrics)
            
            # Calculate overall score and grade
            overall_score = self._calculate_enterprise_score(assessment["metrics"])
            assessment["overall_score"] = overall_score
            assessment["grade"] = self._assign_professional_grade(overall_score)
            
            # Generate professional indicators
            assessment["professional_indicators"] = self._generate_professional_indicators(assessment["metrics"])
            
            # Identify strengths and improvement areas
            assessment["strengths"] = self._identify_content_strengths(assessment["metrics"])
            assessment["improvement_areas"] = self._identify_improvement_areas(assessment["metrics"])
            
            # Industry benchmarks
            assessment["industry_benchmarks"] = self._generate_industry_benchmarks(assessment["metrics"])
            
            # Detailed analysis
            assessment["detailed_analysis"] = self._generate_detailed_quality_analysis(assessment["metrics"], content)
            
            return assessment
            
        except Exception as e:
            print(f"Error in enterprise quality assessment: {str(e)}")
            return self._get_default_assessment()

    def _assess_content_metrics(self, content: str, title: str) -> Dict[str, float]:
        """Assess core content quality metrics"""
        metrics = {}
        
        # 1. Content Length and Depth
        word_count = len(content.split())
        metrics["content_depth"] = min(100, (word_count / 500) * 100)  # Optimal: 500+ words
        
        # 2. Vocabulary Sophistication
        unique_words = len(set(content.lower().split()))
        vocabulary_diversity = (unique_words / max(word_count, 1)) * 100
        metrics["vocabulary_sophistication"] = min(100, vocabulary_diversity * 2)
        
        # 3. Sentence Structure Complexity
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        metrics["sentence_complexity"] = min(100, (avg_sentence_length / 20) * 100)  # Optimal: 15-25 words
        
        # 4. Readability Balance
        readability = self._calculate_readability_score(content)
        # Professional content should be readable but not too simple
        if 40 <= readability <= 70:
            metrics["readability_balance"] = 100
        elif 30 <= readability <= 80:
            metrics["readability_balance"] = 80
        else:
            metrics["readability_balance"] = 50
        
        # 5. Topic Coherence
        metrics["topic_coherence"] = self._assess_topic_coherence(content, title)
        
        return metrics

    def _assess_presentation_quality(self, content: str) -> Dict[str, float]:
        """Assess professional presentation quality"""
        metrics = {}
        
        # 6. Formatting and Structure
        metrics["formatting_quality"] = self._assess_formatting_quality(content)
        
        # 7. Paragraph Organization
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        if 50 <= avg_paragraph_length <= 150:  # Optimal paragraph length
            metrics["paragraph_organization"] = 100
        elif 30 <= avg_paragraph_length <= 200:
            metrics["paragraph_organization"] = 80
        else:
            metrics["paragraph_organization"] = 60
        
        # 8. Professional Tone
        metrics["professional_tone"] = self._assess_professional_tone(content)
        
        # 9. Clarity and Precision
        metrics["clarity_precision"] = self._assess_clarity_precision(content)
        
        return metrics

    def _assess_authority_indicators(self, content: str, url: str) -> Dict[str, float]:
        """Assess authority and credibility indicators"""
        metrics = {}
        
        # 10. Citation and References
        metrics["citation_quality"] = self._assess_citations(content)
        
        # 11. Expertise Indicators
        metrics["expertise_indicators"] = self._assess_expertise_indicators(content)
        
        # 12. Factual Accuracy Indicators
        metrics["factual_accuracy"] = self._assess_factual_indicators(content)
        
        return metrics

    def _assess_evidence_quality(self, content: str) -> Dict[str, float]:
        """Assess evidence and support quality"""
        metrics = {}
        
        # 13. Supporting Evidence
        metrics["supporting_evidence"] = self._assess_supporting_evidence(content)
        
        # 14. Data and Statistics Usage
        metrics["data_usage"] = self._assess_data_usage(content)
        
        # 15. Logical Flow
        metrics["logical_flow"] = self._assess_logical_flow(content)
        
        # 16. Conclusion Strength
        metrics["conclusion_strength"] = self._assess_conclusion_strength(content)
        
        return metrics

    def _assess_topic_coherence(self, content: str, title: str) -> float:
        """Assess how well content stays on topic"""
        try:
            if not title:
                return 75  # Default if no title
            
            title_words = set(title.lower().split())
            content_words = content.lower().split()
            
            # Calculate topic relevance
            title_mentions = sum(1 for word in content_words if word in title_words)
            coherence_score = min(100, (title_mentions / max(len(content_words), 1)) * 1000)
            
            return max(50, coherence_score)  # Minimum 50
        except:
            return 75

    def _assess_formatting_quality(self, content: str) -> float:
        """Assess formatting and structural quality"""
        score = 50  # Base score
        
        # Check for headers
        if any(line.startswith('#') or line.isupper() for line in content.split('\n')):
            score += 15
        
        # Check for lists
        if any(line.strip().startswith(('•', '-', '*', '1.', '2.')) for line in content.split('\n')):
            score += 15
        
        # Check for proper paragraph breaks
        if '\n\n' in content:
            score += 10
        
        # Check for emphasis (bold, italic indicators)
        if any(marker in content for marker in ['**', '*', '_', 'IMPORTANT', 'NOTE:']):
            score += 10
        
        return min(100, score)

    def _assess_professional_tone(self, content: str) -> float:
        """Assess professional tone and language"""
        professional_indicators = [
            'furthermore', 'however', 'therefore', 'consequently', 'moreover',
            'analysis', 'research', 'study', 'findings', 'conclusion',
            'evidence', 'data', 'results', 'methodology', 'approach'
        ]
        
        unprofessional_indicators = [
            'awesome', 'cool', 'weird', 'crazy', 'totally', 'super',
            'lol', 'omg', 'btw', 'fyi', 'asap'
        ]
        
        content_lower = content.lower()
        professional_count = sum(1 for word in professional_indicators if word in content_lower)
        unprofessional_count = sum(1 for word in unprofessional_indicators if word in content_lower)
        
        # Calculate score
        score = 70 + (professional_count * 5) - (unprofessional_count * 10)
        return max(0, min(100, score))

    def _assess_clarity_precision(self, content: str) -> float:
        """Assess clarity and precision of language"""
        # Check for vague language
        vague_terms = ['thing', 'stuff', 'very', 'really', 'quite', 'somewhat', 'maybe', 'perhaps']
        precise_terms = ['specifically', 'precisely', 'exactly', 'clearly', 'definitively']
        
        content_lower = content.lower()
        vague_count = sum(1 for term in vague_terms if term in content_lower)
        precise_count = sum(1 for term in precise_terms if term in content_lower)
        
        word_count = len(content.split())
        vague_ratio = vague_count / max(word_count, 1)
        precise_ratio = precise_count / max(word_count, 1)
        
        score = 70 - (vague_ratio * 1000) + (precise_ratio * 500)
        return max(0, min(100, score))

    def _assess_citations(self, content: str) -> float:
        """Assess citation and reference quality"""
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (2024), etc.
            r'according to',  # "according to"
            r'source:',  # "source:"
            r'reference:',  # "reference:"
            r'study by',  # "study by"
            r'research shows',  # "research shows"
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Score based on citation density
        word_count = len(content.split())
        citation_density = citation_count / max(word_count, 1) * 1000
        
        return min(100, citation_density * 20)

    def _assess_expertise_indicators(self, content: str) -> float:
        """Assess indicators of subject matter expertise"""
        expertise_indicators = [
            'research', 'study', 'analysis', 'methodology', 'findings',
            'peer-reviewed', 'published', 'journal', 'conference',
            'expert', 'specialist', 'professional', 'certified',
            'experience', 'background', 'qualification'
        ]
        
        content_lower = content.lower()
        expertise_count = sum(1 for indicator in expertise_indicators if indicator in content_lower)
        
        return min(100, expertise_count * 8)

    def _assess_factual_indicators(self, content: str) -> float:
        """Assess indicators of factual accuracy"""
        factual_indicators = [
            'data shows', 'statistics indicate', 'research confirms',
            'studies demonstrate', 'evidence suggests', 'proven',
            'verified', 'documented', 'established', 'confirmed'
        ]
        
        speculation_indicators = [
            'i think', 'i believe', 'maybe', 'possibly', 'might be',
            'could be', 'seems like', 'appears to', 'probably'
        ]
        
        content_lower = content.lower()
        factual_count = sum(1 for indicator in factual_indicators if indicator in content_lower)
        speculation_count = sum(1 for indicator in speculation_indicators if indicator in content_lower)
        
        score = 70 + (factual_count * 10) - (speculation_count * 5)
        return max(0, min(100, score))

    def _assess_supporting_evidence(self, content: str) -> float:
        """Assess quality and quantity of supporting evidence"""
        evidence_indicators = [
            'example', 'for instance', 'case study', 'data',
            'statistics', 'survey', 'poll', 'research',
            'experiment', 'test', 'trial', 'observation'
        ]
        
        content_lower = content.lower()
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content_lower)
        
        return min(100, evidence_count * 12)

    def _assess_data_usage(self, content: str) -> float:
        """Assess usage of data and statistics"""
        # Look for numbers, percentages, and data-related terms
        number_pattern = r'\b\d+(?:\.\d+)?%?\b'
        numbers = re.findall(number_pattern, content)
        
        data_terms = ['percent', 'percentage', 'ratio', 'rate', 'average', 'median', 'mean']
        data_count = sum(1 for term in data_terms if term in content.lower())
        
        score = len(numbers) * 5 + data_count * 10
        return min(100, score)

    def _assess_logical_flow(self, content: str) -> float:
        """Assess logical flow and organization"""
        transition_words = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'however', 'therefore', 'consequently', 'furthermore',
            'moreover', 'additionally', 'in contrast', 'similarly'
        ]
        
        content_lower = content.lower()
        transition_count = sum(1 for word in transition_words if word in content_lower)
        
        # Check for logical structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        structure_score = min(50, len(paragraphs) * 10)  # More paragraphs = better structure
        
        transition_score = min(50, transition_count * 8)
        
        return structure_score + transition_score

    def _assess_conclusion_strength(self, content: str) -> float:
        """Assess strength of conclusions and recommendations"""
        conclusion_indicators = [
            'conclusion', 'summary', 'in summary', 'to conclude',
            'therefore', 'thus', 'hence', 'as a result',
            'recommendation', 'suggest', 'propose', 'recommend'
        ]
        
        # Look for conclusion indicators in the last 20% of content
        content_length = len(content)
        last_section = content[int(content_length * 0.8):].lower()
        
        conclusion_count = sum(1 for indicator in conclusion_indicators if indicator in last_section)
        
        return min(100, conclusion_count * 25)

    def _calculate_enterprise_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall enterprise quality score"""
        weights = {
            'content_depth': 0.08,
            'vocabulary_sophistication': 0.06,
            'sentence_complexity': 0.05,
            'readability_balance': 0.07,
            'topic_coherence': 0.08,
            'formatting_quality': 0.06,
            'paragraph_organization': 0.05,
            'professional_tone': 0.08,
            'clarity_precision': 0.07,
            'citation_quality': 0.08,
            'expertise_indicators': 0.06,
            'factual_accuracy': 0.08,
            'supporting_evidence': 0.07,
            'data_usage': 0.05,
            'logical_flow': 0.06,
            'conclusion_strength': 0.05
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, score in metrics.items():
            if metric in weights:
                total_score += score * weights[metric]
                total_weight += weights[metric]
        
        return round(total_score / max(total_weight, 1), 1)

    def _assign_professional_grade(self, score: float) -> str:
        """Assign professional grade based on score"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 45:
            return "D+"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def _generate_professional_indicators(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate professional quality indicators"""
        indicators = {}
        
        # Content Quality
        content_avg = (metrics.get('content_depth', 0) + metrics.get('vocabulary_sophistication', 0) + 
                      metrics.get('topic_coherence', 0)) / 3
        if content_avg >= 80:
            indicators['content_quality'] = "Excellent - Comprehensive and sophisticated"
        elif content_avg >= 60:
            indicators['content_quality'] = "Good - Well-developed content"
        else:
            indicators['content_quality'] = "Needs Improvement - Expand depth and sophistication"
        
        # Professional Presentation
        presentation_avg = (metrics.get('formatting_quality', 0) + metrics.get('professional_tone', 0) + 
                           metrics.get('clarity_precision', 0)) / 3
        if presentation_avg >= 80:
            indicators['presentation'] = "Excellent - Highly professional presentation"
        elif presentation_avg >= 60:
            indicators['presentation'] = "Good - Professional standards met"
        else:
            indicators['presentation'] = "Needs Improvement - Enhance professional presentation"
        
        # Authority & Credibility
        authority_avg = (metrics.get('citation_quality', 0) + metrics.get('expertise_indicators', 0) + 
                        metrics.get('factual_accuracy', 0)) / 3
        if authority_avg >= 80:
            indicators['authority'] = "Excellent - High credibility and authority"
        elif authority_avg >= 60:
            indicators['authority'] = "Good - Adequate credibility established"
        else:
            indicators['authority'] = "Needs Improvement - Strengthen credibility indicators"
        
        return indicators

    def _identify_content_strengths(self, metrics: Dict[str, float]) -> List[str]:
        """Identify content strengths based on high-scoring metrics"""
        strengths = []
        
        for metric, score in metrics.items():
            if score >= 85:
                if metric == 'content_depth':
                    strengths.append("Comprehensive content depth and coverage")
                elif metric == 'vocabulary_sophistication':
                    strengths.append("Sophisticated vocabulary and language use")
                elif metric == 'professional_tone':
                    strengths.append("Excellent professional tone and style")
                elif metric == 'citation_quality':
                    strengths.append("Strong citation and reference practices")
                elif metric == 'logical_flow':
                    strengths.append("Excellent logical organization and flow")
                elif metric == 'supporting_evidence':
                    strengths.append("Strong supporting evidence and examples")
        
        return strengths[:5]  # Top 5 strengths

    def _identify_improvement_areas(self, metrics: Dict[str, float]) -> List[str]:
        """Identify areas needing improvement based on low-scoring metrics"""
        improvements = []
        
        for metric, score in metrics.items():
            if score < 60:
                if metric == 'content_depth':
                    improvements.append("Expand content depth and detail")
                elif metric == 'vocabulary_sophistication':
                    improvements.append("Enhance vocabulary sophistication")
                elif metric == 'professional_tone':
                    improvements.append("Improve professional tone and language")
                elif metric == 'citation_quality':
                    improvements.append("Add more citations and references")
                elif metric == 'logical_flow':
                    improvements.append("Improve logical organization and transitions")
                elif metric == 'supporting_evidence':
                    improvements.append("Add more supporting evidence and examples")
                elif metric == 'clarity_precision':
                    improvements.append("Enhance clarity and precision of language")
        
        return improvements[:5]  # Top 5 improvement areas

    def _generate_industry_benchmarks(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate industry benchmark comparisons"""
        benchmarks = {}
        
        overall_avg = sum(metrics.values()) / len(metrics)
        
        if overall_avg >= 85:
            benchmarks['industry_comparison'] = "Exceeds industry standards"
            benchmarks['competitive_position'] = "Top-tier professional content"
        elif overall_avg >= 70:
            benchmarks['industry_comparison'] = "Meets industry standards"
            benchmarks['competitive_position'] = "Competitive professional content"
        elif overall_avg >= 55:
            benchmarks['industry_comparison'] = "Below industry standards"
            benchmarks['competitive_position'] = "Requires improvement for competitiveness"
        else:
            benchmarks['industry_comparison'] = "Significantly below standards"
            benchmarks['competitive_position'] = "Major improvements needed"
        
        return benchmarks

    def _generate_detailed_quality_analysis(self, metrics: Dict[str, float], content: str) -> Dict[str, Any]:
        """Generate detailed quality analysis"""
        analysis = {
            'content_statistics': {
                'word_count': len(content.split()),
                'sentence_count': len([s for s in content.split('.') if s.strip()]),
                'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                'readability_score': self._calculate_readability_score(content)
            },
            'quality_distribution': self._analyze_quality_distribution(metrics),
            'professional_readiness': self._assess_professional_readiness(metrics),
            'recommendations_priority': self._generate_priority_recommendations(metrics)
        }
        
        return analysis

    def _analyze_quality_distribution(self, metrics: Dict[str, float]) -> Dict[str, int]:
        """Analyze distribution of quality scores"""
        excellent = sum(1 for score in metrics.values() if score >= 85)
        good = sum(1 for score in metrics.values() if 70 <= score < 85)
        fair = sum(1 for score in metrics.values() if 55 <= score < 70)
        poor = sum(1 for score in metrics.values() if score < 55)
        
        return {
            'excellent_metrics': excellent,
            'good_metrics': good,
            'fair_metrics': fair,
            'poor_metrics': poor
        }

    def _assess_professional_readiness(self, metrics: Dict[str, float]) -> str:
        """Assess overall professional readiness"""
        critical_metrics = ['professional_tone', 'citation_quality', 'factual_accuracy', 'logical_flow']
        critical_scores = [metrics.get(metric, 0) for metric in critical_metrics]
        critical_avg = sum(critical_scores) / len(critical_scores)
        
        if critical_avg >= 80:
            return "Ready for professional publication"
        elif critical_avg >= 65:
            return "Minor revisions needed for professional standards"
        elif critical_avg >= 50:
            return "Moderate revisions required"
        else:
            return "Major revisions needed before professional use"

    def _generate_priority_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate priority-ordered recommendations"""
        recommendations = []
        
        # Sort metrics by score (lowest first for improvement)
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1])
        
        for metric, score in sorted_metrics[:3]:  # Top 3 lowest scores
            if score < 70:
                if metric == 'citation_quality':
                    recommendations.append("HIGH: Add credible sources and citations")
                elif metric == 'professional_tone':
                    recommendations.append("HIGH: Improve professional language and tone")
                elif metric == 'content_depth':
                    recommendations.append("MEDIUM: Expand content depth and detail")
                elif metric == 'logical_flow':
                    recommendations.append("MEDIUM: Improve organization and transitions")
        
        return recommendations

    def _get_default_assessment(self) -> Dict[str, Any]:
        """Return default assessment in case of errors"""
        return {
            "overall_score": 50.0,
            "grade": "C",
            "metrics": {},
            "professional_indicators": {"error": "Assessment failed"},
            "improvement_areas": ["Unable to assess - please try again"],
            "strengths": [],
            "industry_benchmarks": {"status": "Assessment unavailable"},
            "detailed_analysis": {"error": "Analysis failed"}
         }

    def generate_ai_enhancement_suggestions(self, content: str, title: str = "", url: str = "") -> Dict[str, Any]:
        """
        Generate AI-powered content enhancement suggestions with specific recommendations
        """
        try:
            # Get comprehensive analysis
            gaps_analysis = self.detect_content_gaps(content, title)
            quality_assessment = self.enterprise_quality_assessment(content, title, url)
            
            enhancement_suggestions = {
                "priority_improvements": [],
                "content_enhancements": [],
                "structural_improvements": [],
                "professional_upgrades": [],
                "gap_filling_recommendations": [],
                "ai_generated_content": {},
                "implementation_guide": {},
                "estimated_impact": {}
            }
            
            # Generate priority improvements based on quality metrics
            enhancement_suggestions["priority_improvements"] = self._generate_priority_improvements(
                quality_assessment["metrics"], gaps_analysis
            )
            
            # Generate content enhancement suggestions
            enhancement_suggestions["content_enhancements"] = self._generate_content_enhancements(
                content, quality_assessment["metrics"]
            )
            
            # Generate structural improvements
            enhancement_suggestions["structural_improvements"] = self._generate_structural_improvements(
                content, gaps_analysis
            )
            
            # Generate professional upgrades
            enhancement_suggestions["professional_upgrades"] = self._generate_professional_upgrades(
                content, quality_assessment["metrics"]
            )
            
            # Generate gap-filling recommendations
            enhancement_suggestions["gap_filling_recommendations"] = self._generate_gap_filling_recommendations(
                gaps_analysis, content
            )
            
            # Generate AI-powered content suggestions
            enhancement_suggestions["ai_generated_content"] = self._generate_ai_content_suggestions(
                content, title, gaps_analysis
            )
            
            # Create implementation guide
            enhancement_suggestions["implementation_guide"] = self._create_implementation_guide(
                enhancement_suggestions
            )
            
            # Estimate impact of improvements
            enhancement_suggestions["estimated_impact"] = self._estimate_improvement_impact(
                quality_assessment, gaps_analysis
            )
            
            return enhancement_suggestions
            
        except Exception as e:
            print(f"Error generating AI enhancement suggestions: {str(e)}")
            return self._get_default_enhancement_suggestions()

    def _generate_priority_improvements(self, quality_metrics: Dict[str, float], gaps_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate priority improvements based on quality metrics and gaps"""
        improvements = []
        
        # Identify critical quality issues
        critical_metrics = {k: v for k, v in quality_metrics.items() if v < 50}
        
        for metric, score in sorted(critical_metrics.items(), key=lambda x: x[1]):
            improvement = {
                "type": "critical_quality",
                "metric": metric,
                "current_score": score,
                "target_score": 75,
                "priority": "HIGH",
                "description": self._get_metric_improvement_description(metric),
                "specific_actions": self._get_metric_specific_actions(metric),
                "estimated_effort": self._estimate_improvement_effort(metric, score)
            }
            improvements.append(improvement)
        
        # Add critical gaps
        completeness_score = gaps_analysis.get("completeness_score", 100)
        if completeness_score < 60:
            improvements.append({
                "type": "content_completeness",
                "current_score": completeness_score,
                "target_score": 80,
                "priority": "HIGH",
                "description": "Address critical content gaps and missing information",
                "specific_actions": [
                    "Add missing sections identified in gap analysis",
                    "Expand on incomplete topics",
                    "Provide supporting evidence for claims"
                ],
                "estimated_effort": "Medium"
            })
        
        return improvements[:5]  # Top 5 priority improvements

    def _generate_content_enhancements(self, content: str, quality_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate specific content enhancement suggestions"""
        enhancements = []
        
        # Vocabulary enhancement
        vocab_score = quality_metrics.get("vocabulary_sophistication", 0)
        if vocab_score < 70:
            enhancements.append({
                "category": "vocabulary",
                "title": "Enhance Vocabulary Sophistication",
                "description": "Improve word choice and language sophistication",
                "suggestions": [
                    "Replace common words with more precise alternatives",
                    "Use domain-specific terminology appropriately",
                    "Vary sentence structure and word choice",
                    "Eliminate redundant phrases"
                ],
                "examples": self._generate_vocabulary_examples(content)
            })
        
        # Content depth enhancement
        depth_score = quality_metrics.get("content_depth", 0)
        if depth_score < 70:
            enhancements.append({
                "category": "depth",
                "title": "Increase Content Depth",
                "description": "Expand content with more detailed information",
                "suggestions": [
                    "Add more detailed explanations",
                    "Include additional examples and case studies",
                    "Provide background context",
                    "Expand on key concepts"
                ],
                "target_word_count": max(500, len(content.split()) * 1.5)
            })
        
        # Evidence enhancement
        evidence_score = quality_metrics.get("supporting_evidence", 0)
        if evidence_score < 70:
            enhancements.append({
                "category": "evidence",
                "title": "Strengthen Supporting Evidence",
                "description": "Add credible sources and supporting data",
                "suggestions": [
                    "Include relevant statistics and data",
                    "Add expert quotes and opinions",
                    "Reference credible sources",
                    "Provide real-world examples"
                ],
                "evidence_types": ["statistics", "expert_opinions", "case_studies", "research_findings"]
            })
        
        return enhancements

    def _generate_structural_improvements(self, content: str, gaps_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate structural improvement suggestions"""
        improvements = []
        
        # Check for structural gaps
        structural_gaps = gaps_analysis.get("structural_gaps", [])
        
        if structural_gaps:
            improvements.append({
                "category": "organization",
                "title": "Improve Content Organization",
                "description": "Enhance logical flow and structure",
                "issues": [gap.get("description", "") for gap in structural_gaps],
                "solutions": [
                    "Add clear section headers",
                    "Use transitional phrases between sections",
                    "Create logical progression of ideas",
                    "Add introduction and conclusion sections"
                ]
            })
        
        # Paragraph organization
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            improvements.append({
                "category": "paragraphs",
                "title": "Improve Paragraph Structure",
                "description": "Break content into well-organized paragraphs",
                "suggestions": [
                    "Split long paragraphs into focused sections",
                    "Ensure each paragraph has a clear main idea",
                    "Use topic sentences to introduce paragraphs",
                    "Maintain consistent paragraph length"
                ]
            })
        
        # Formatting improvements
        if not any(line.startswith('#') for line in content.split('\n')):
            improvements.append({
                "category": "formatting",
                "title": "Add Structural Formatting",
                "description": "Improve readability with proper formatting",
                "suggestions": [
                    "Add section headers and subheaders",
                    "Use bullet points for lists",
                    "Emphasize key terms and concepts",
                    "Add visual breaks between sections"
                ]
            })
        
        return improvements

    def _generate_professional_upgrades(self, content: str, quality_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate professional upgrade suggestions"""
        upgrades = []
        
        # Professional tone
        tone_score = quality_metrics.get("professional_tone", 0)
        if tone_score < 75:
            upgrades.append({
                "category": "tone",
                "title": "Enhance Professional Tone",
                "description": "Improve language formality and professionalism",
                "improvements": [
                    "Use formal language and terminology",
                    "Eliminate casual expressions",
                    "Adopt objective, third-person perspective",
                    "Use professional vocabulary"
                ],
                "tone_indicators": self._analyze_tone_indicators(content)
            })
        
        # Citation quality
        citation_score = quality_metrics.get("citation_quality", 0)
        if citation_score < 70:
            upgrades.append({
                "category": "citations",
                "title": "Improve Citation Practices",
                "description": "Add proper citations and references",
                "requirements": [
                    "Cite all factual claims",
                    "Use consistent citation format",
                    "Include publication dates",
                    "Reference authoritative sources"
                ],
                "citation_style": "APA"
            })
        
        # Factual accuracy
        accuracy_score = quality_metrics.get("factual_accuracy", 0)
        if accuracy_score < 75:
            upgrades.append({
                "category": "accuracy",
                "title": "Strengthen Factual Accuracy",
                "description": "Improve credibility and fact-based content",
                "actions": [
                    "Replace opinions with facts",
                    "Add verification for claims",
                    "Use precise language",
                    "Avoid speculation"
                ]
            })
        
        return upgrades

    def _generate_gap_filling_recommendations(self, gaps_analysis: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Generate specific recommendations for filling content gaps"""
        recommendations = []
        
        # Information gaps
        info_gaps = gaps_analysis.get("information_gaps", [])
        for gap in info_gaps[:3]:  # Top 3 information gaps
            recommendations.append({
                "gap_type": "information",
                "description": gap.get("description", ""),
                "severity": gap.get("severity", "medium"),
                "suggested_content": self._suggest_gap_content(gap, content),
                "research_needed": gap.get("research_required", False),
                "priority": self._determine_gap_priority(gap)
            })
        
        # Evidence gaps
        evidence_gaps = gaps_analysis.get("evidence_gaps", [])
        for gap in evidence_gaps[:2]:  # Top 2 evidence gaps
            recommendations.append({
                "gap_type": "evidence",
                "description": gap.get("description", ""),
                "severity": gap.get("severity", "medium"),
                "evidence_types": ["statistics", "expert_quotes", "case_studies"],
                "suggested_sources": self._suggest_evidence_sources(gap),
                "priority": self._determine_gap_priority(gap)
            })
        
        return recommendations

    def _generate_ai_content_suggestions(self, content: str, title: str, gaps_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered content suggestions"""
        suggestions = {
            "introduction_enhancement": "",
            "conclusion_strengthening": "",
            "transition_improvements": [],
            "topic_expansions": [],
            "supporting_examples": []
        }
        
        # Introduction enhancement
        if not content.strip().startswith(("Introduction", "Overview", "Abstract")):
            suggestions["introduction_enhancement"] = self._generate_introduction_suggestion(title, content)
        
        # Conclusion strengthening
        if not any(word in content.lower()[-200:] for word in ["conclusion", "summary", "in summary"]):
            suggestions["conclusion_strengthening"] = self._generate_conclusion_suggestion(content)
        
        # Transition improvements
        suggestions["transition_improvements"] = self._suggest_transitions(content)
        
        # Topic expansions
        suggestions["topic_expansions"] = self._suggest_topic_expansions(content, gaps_analysis)
        
        # Supporting examples
        suggestions["supporting_examples"] = self._suggest_supporting_examples(content)
        
        return suggestions

    def _create_implementation_guide(self, enhancement_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Create step-by-step implementation guide"""
        guide = {
            "phase_1_critical": [],
            "phase_2_important": [],
            "phase_3_enhancement": [],
            "estimated_timeline": "",
            "resource_requirements": []
        }
        
        # Organize by priority
        for improvement in enhancement_suggestions.get("priority_improvements", []):
            if improvement.get("priority") == "HIGH":
                guide["phase_1_critical"].append({
                    "task": improvement.get("description", ""),
                    "actions": improvement.get("specific_actions", []),
                    "effort": improvement.get("estimated_effort", "Medium")
                })
        
        # Add content enhancements to phase 2
        for enhancement in enhancement_suggestions.get("content_enhancements", []):
            guide["phase_2_important"].append({
                "task": enhancement.get("title", ""),
                "actions": enhancement.get("suggestions", [])
            })
        
        # Add professional upgrades to phase 3
        for upgrade in enhancement_suggestions.get("professional_upgrades", []):
            guide["phase_3_enhancement"].append({
                "task": upgrade.get("title", ""),
                "actions": upgrade.get("improvements", upgrade.get("requirements", []))
            })
        
        # Estimate timeline
        total_tasks = len(guide["phase_1_critical"]) + len(guide["phase_2_important"]) + len(guide["phase_3_enhancement"])
        if total_tasks <= 3:
            guide["estimated_timeline"] = "1-2 hours"
        elif total_tasks <= 6:
            guide["estimated_timeline"] = "2-4 hours"
        else:
            guide["estimated_timeline"] = "4-8 hours"
        
        guide["resource_requirements"] = [
            "Access to credible sources for citations",
            "Time for research and fact-checking",
            "Style guide for professional formatting"
        ]
        
        return guide

    def _estimate_improvement_impact(self, quality_assessment: Dict[str, Any], gaps_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the impact of implementing improvements"""
        current_score = quality_assessment.get("overall_score", 50)
        current_grade = quality_assessment.get("grade", "C")
        
        # Estimate potential improvement
        potential_increase = 0
        
        # Critical improvements can add 15-25 points
        critical_count = len([m for m in quality_assessment.get("metrics", {}).values() if m < 50])
        potential_increase += min(25, critical_count * 8)
        
        # Gap filling can add 10-15 points
        completeness = gaps_analysis.get("completeness_score", 100)
        if completeness < 60:
            potential_increase += 15
        elif completeness < 80:
            potential_increase += 10
        
        projected_score = min(95, current_score + potential_increase)
        projected_grade = self._assign_professional_grade(projected_score)
        
        return {
            "current_score": current_score,
            "current_grade": current_grade,
            "projected_score": projected_score,
            "projected_grade": projected_grade,
            "potential_improvement": potential_increase,
            "confidence_level": "High" if potential_increase >= 15 else "Medium",
            "key_impact_areas": [
                "Professional credibility",
                "Content authority",
                "Reader engagement",
                "Information completeness"
            ]
        }

    # Helper methods for content generation
    def _get_metric_improvement_description(self, metric: str) -> str:
        """Get description for metric improvement"""
        descriptions = {
            "content_depth": "Expand content with more detailed information and comprehensive coverage",
            "vocabulary_sophistication": "Enhance vocabulary with more precise and sophisticated language",
            "professional_tone": "Improve language formality and professional presentation",
            "citation_quality": "Add proper citations and credible source references",
            "logical_flow": "Improve organization and logical progression of ideas",
            "supporting_evidence": "Strengthen content with more supporting evidence and examples"
        }
        return descriptions.get(metric, f"Improve {metric.replace('_', ' ')}")

    def _get_metric_specific_actions(self, metric: str) -> List[str]:
        """Get specific actions for metric improvement"""
        actions = {
            "content_depth": [
                "Add more detailed explanations",
                "Include additional examples",
                "Expand on key concepts",
                "Provide background context"
            ],
            "vocabulary_sophistication": [
                "Replace common words with precise alternatives",
                "Use domain-specific terminology",
                "Vary sentence structure",
                "Eliminate redundant phrases"
            ],
            "professional_tone": [
                "Use formal language",
                "Adopt objective perspective",
                "Eliminate casual expressions",
                "Use professional vocabulary"
            ],
            "citation_quality": [
                "Add source citations",
                "Include publication dates",
                "Reference authoritative sources",
                "Use consistent citation format"
            ]
        }
        return actions.get(metric, ["Improve this aspect of content quality"])

    def _estimate_improvement_effort(self, metric: str, score: float) -> str:
        """Estimate effort required for improvement"""
        if score < 30:
            return "High"
        elif score < 50:
            return "Medium"
        else:
            return "Low"

    def _generate_vocabulary_examples(self, content: str) -> List[Dict[str, str]]:
        """Generate vocabulary improvement examples"""
        common_replacements = [
            {"original": "very important", "improved": "crucial"},
            {"original": "a lot of", "improved": "numerous"},
            {"original": "good", "improved": "effective"},
            {"original": "bad", "improved": "detrimental"},
            {"original": "shows", "improved": "demonstrates"}
        ]
        
        examples = []
        content_lower = content.lower()
        
        for replacement in common_replacements:
            if replacement["original"] in content_lower:
                examples.append(replacement)
        
        return examples[:3]

    def _analyze_tone_indicators(self, content: str) -> Dict[str, List[str]]:
        """Analyze tone indicators in content"""
        casual_indicators = []
        professional_indicators = []
        
        casual_terms = ["awesome", "cool", "totally", "super", "really"]
        professional_terms = ["furthermore", "however", "consequently", "analysis", "methodology"]
        
        content_lower = content.lower()
        
        for term in casual_terms:
            if term in content_lower:
                casual_indicators.append(term)
        
        for term in professional_terms:
            if term in content_lower:
                professional_indicators.append(term)
        
        return {
            "casual_indicators": casual_indicators,
            "professional_indicators": professional_indicators
        }

    def _suggest_gap_content(self, gap: Dict[str, Any], content: str) -> str:
        """Suggest content to fill identified gaps"""
        gap_type = gap.get("type", "")
        
        if "methodology" in gap_type.lower():
            return "Add a detailed methodology section explaining the approach, data sources, and analytical methods used."
        elif "conclusion" in gap_type.lower():
            return "Include a comprehensive conclusion that summarizes key findings and their implications."
        elif "background" in gap_type.lower():
            return "Provide background context and relevant historical information to frame the topic."
        else:
            return f"Address the identified gap: {gap.get('description', 'Missing information')}"

    def _suggest_evidence_sources(self, gap: Dict[str, Any]) -> List[str]:
        """Suggest evidence sources for gaps"""
        return [
            "Peer-reviewed academic journals",
            "Government statistical databases",
            "Industry reports and white papers",
            "Expert interviews and quotes",
            "Case studies and real-world examples"
        ]

    def _determine_gap_priority(self, gap: Dict[str, Any]) -> str:
        """Determine priority level for gap"""
        severity = gap.get("severity", "medium")
        if severity == "high":
            return "HIGH"
        elif severity == "medium":
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_introduction_suggestion(self, title: str, content: str) -> str:
        """Generate introduction suggestion"""
        return f"Consider adding an introduction that provides an overview of {title}, outlines the main topics to be covered, and establishes the purpose and scope of the content."

    def _generate_conclusion_suggestion(self, content: str) -> str:
        """Generate conclusion suggestion"""
        return "Add a conclusion section that summarizes the key points, highlights the main findings or insights, and provides final thoughts or recommendations."

    def _suggest_transitions(self, content: str) -> List[str]:
        """Suggest transition improvements"""
        return [
            "Add transitional phrases between paragraphs",
            "Use connecting words like 'furthermore', 'however', 'consequently'",
            "Create logical bridges between different topics",
            "Ensure smooth flow from one idea to the next"
        ]

    def _suggest_topic_expansions(self, content: str, gaps_analysis: Dict[str, Any]) -> List[str]:
        """Suggest topic expansions based on gaps"""
        expansions = []
        
        info_gaps = gaps_analysis.get("information_gaps", [])
        for gap in info_gaps[:2]:
            expansions.append(f"Expand on: {gap.get('description', 'Identified topic')}")
        
        return expansions

    def _suggest_supporting_examples(self, content: str) -> List[str]:
        """Suggest supporting examples"""
        return [
            "Add real-world case studies",
            "Include statistical data and figures",
            "Provide concrete examples",
            "Add expert quotes and opinions"
        ]

    def _get_default_enhancement_suggestions(self) -> Dict[str, Any]:
        """Return default enhancement suggestions in case of errors"""
        return {
            "priority_improvements": [{"error": "Unable to generate suggestions"}],
            "content_enhancements": [],
            "structural_improvements": [],
            "professional_upgrades": [],
            "gap_filling_recommendations": [],
            "ai_generated_content": {"error": "Content generation failed"},
            "implementation_guide": {"error": "Guide generation failed"},
             "estimated_impact": {"error": "Impact estimation failed"}
         }

    def multi_dimensional_professional_scoring(self, content: str, title: str = "", url: str = "", industry: str = "general") -> Dict[str, Any]:
        """
        Comprehensive multi-dimensional professional scoring system with industry benchmarks
        """
        try:
            scoring_result = {
                "overall_professional_score": 0,
                "dimensional_scores": {},
                "industry_benchmarks": {},
                "competitive_analysis": {},
                "professional_indicators": {},
                "improvement_roadmap": {},
                "certification_readiness": {},
                "market_positioning": {}
            }
            
            # Calculate dimensional scores
            scoring_result["dimensional_scores"] = self._calculate_dimensional_scores(content, title, url)
            
            # Get industry benchmarks
            scoring_result["industry_benchmarks"] = self._get_industry_benchmarks(industry)
            
            # Perform competitive analysis
            scoring_result["competitive_analysis"] = self._perform_competitive_analysis(
                scoring_result["dimensional_scores"], industry
            )
            
            # Calculate overall professional score
            scoring_result["overall_professional_score"] = self._calculate_overall_professional_score(
                scoring_result["dimensional_scores"]
            )
            
            # Generate professional indicators
            scoring_result["professional_indicators"] = self._generate_dimensional_professional_indicators(
                scoring_result["dimensional_scores"], scoring_result["overall_professional_score"]
            )
            
            # Create improvement roadmap
            scoring_result["improvement_roadmap"] = self._create_improvement_roadmap(
                scoring_result["dimensional_scores"], scoring_result["industry_benchmarks"]
            )
            
            # Assess certification readiness
            scoring_result["certification_readiness"] = self._assess_certification_readiness(
                scoring_result["dimensional_scores"], industry
            )
            
            # Determine market positioning
            scoring_result["market_positioning"] = self._determine_market_positioning(
                scoring_result["overall_professional_score"], scoring_result["competitive_analysis"]
            )
            
            return scoring_result
            
        except Exception as e:
            print(f"Error in multi-dimensional professional scoring: {str(e)}")
            return self._get_default_professional_scoring()

    def _calculate_dimensional_scores(self, content: str, title: str, url: str) -> Dict[str, float]:
        """Calculate scores across multiple professional dimensions"""
        dimensions = {}
        
        # Content Quality Dimension (25% weight)
        dimensions["content_quality"] = self._score_content_quality_dimension(content)
        
        # Professional Presentation Dimension (20% weight)
        dimensions["professional_presentation"] = self._score_presentation_dimension(content, title)
        
        # Authority & Credibility Dimension (20% weight)
        dimensions["authority_credibility"] = self._score_authority_dimension(content, url)
        
        # Technical Excellence Dimension (15% weight)
        dimensions["technical_excellence"] = self._score_technical_dimension(content)
        
        # Communication Effectiveness Dimension (10% weight)
        dimensions["communication_effectiveness"] = self._score_communication_dimension(content)
        
        # Innovation & Insight Dimension (10% weight)
        dimensions["innovation_insight"] = self._score_innovation_dimension(content, title)
        
        return dimensions

    def _score_content_quality_dimension(self, content: str) -> float:
        """Score content quality dimension"""
        score = 0
        max_score = 100
        
        # Depth and comprehensiveness (30 points)
        word_count = len(content.split())
        if word_count >= 1000:
            score += 30
        elif word_count >= 500:
            score += 20
        elif word_count >= 250:
            score += 10
        
        # Information accuracy and factual content (25 points)
        factual_indicators = len([word for word in content.lower().split() 
                                if word in ["research", "study", "data", "analysis", "evidence", "findings"]])
        score += min(25, factual_indicators * 3)
        
        # Content structure and organization (25 points)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) >= 5:
            score += 25
        elif len(paragraphs) >= 3:
            score += 15
        elif len(paragraphs) >= 2:
            score += 10
        
        # Relevance and focus (20 points)
        if len(content.strip()) > 100:  # Basic content check
            score += 20
        
        return min(max_score, score)

    def _score_presentation_dimension(self, content: str, title: str) -> float:
        """Score professional presentation dimension"""
        score = 0
        max_score = 100
        
        # Professional formatting (30 points)
        has_headers = any(line.strip().startswith('#') for line in content.split('\n'))
        has_lists = any(line.strip().startswith(('-', '*', '1.', '2.')) for line in content.split('\n'))
        
        if has_headers:
            score += 15
        if has_lists:
            score += 15
        
        # Language sophistication (25 points)
        sophisticated_words = ["furthermore", "consequently", "nevertheless", "comprehensive", 
                             "methodology", "implementation", "optimization", "strategic"]
        sophisticated_count = sum(1 for word in sophisticated_words if word in content.lower())
        score += min(25, sophisticated_count * 5)
        
        # Professional tone (25 points)
        casual_words = ["awesome", "cool", "super", "totally", "really good"]
        casual_count = sum(1 for word in casual_words if word in content.lower())
        if casual_count == 0:
            score += 25
        elif casual_count <= 2:
            score += 15
        else:
            score += 5
        
        # Title quality (20 points)
        if title and len(title.strip()) > 5:
            score += 20
        
        return min(max_score, score)

    def _score_authority_dimension(self, content: str, url: str) -> float:
        """Score authority and credibility dimension"""
        score = 0
        max_score = 100
        
        # Citations and references (40 points)
        citation_patterns = ["http", "www", "doi:", "et al", "journal", "published"]
        citation_count = sum(1 for pattern in citation_patterns if pattern in content.lower())
        score += min(40, citation_count * 8)
        
        # Expert language and terminology (30 points)
        expert_terms = ["methodology", "analysis", "framework", "implementation", "optimization",
                       "evaluation", "assessment", "validation", "verification", "systematic"]
        expert_count = sum(1 for term in expert_terms if term in content.lower())
        score += min(30, expert_count * 3)
        
        # Source credibility (20 points)
        if url:
            credible_domains = [".edu", ".gov", ".org", "research", "academic"]
            if any(domain in url.lower() for domain in credible_domains):
                score += 20
            else:
                score += 10
        
        # Objective presentation (10 points)
        subjective_words = ["i think", "i believe", "in my opinion", "personally"]
        subjective_count = sum(1 for phrase in subjective_words if phrase in content.lower())
        if subjective_count == 0:
            score += 10
        elif subjective_count <= 2:
            score += 5
        
        return min(max_score, score)

    def _score_technical_dimension(self, content: str) -> float:
        """Score technical excellence dimension"""
        score = 0
        max_score = 100
        
        # Technical accuracy (40 points)
        technical_terms = ["algorithm", "system", "process", "method", "technique", "approach",
                          "solution", "implementation", "architecture", "design"]
        technical_count = sum(1 for term in technical_terms if term in content.lower())
        score += min(40, technical_count * 4)
        
        # Detailed explanations (30 points)
        explanation_words = ["because", "therefore", "thus", "consequently", "as a result",
                           "due to", "leads to", "causes", "results in"]
        explanation_count = sum(1 for word in explanation_words if word in content.lower())
        score += min(30, explanation_count * 5)
        
        # Precision and specificity (30 points)
        specific_indicators = ["specifically", "precisely", "exactly", "particular", "detailed"]
        specific_count = sum(1 for word in specific_indicators if word in content.lower())
        score += min(30, specific_count * 6)
        
        return min(max_score, score)

    def _score_communication_dimension(self, content: str) -> float:
        """Score communication effectiveness dimension"""
        score = 0
        max_score = 100
        
        # Clarity and readability (50 points)
        sentences = [s.strip() for s in content.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 10 <= avg_sentence_length <= 20:
            score += 50
        elif 8 <= avg_sentence_length <= 25:
            score += 35
        else:
            score += 20
        
        # Logical flow (30 points)
        transition_words = ["however", "furthermore", "additionally", "moreover", "consequently",
                          "therefore", "thus", "meanwhile", "subsequently"]
        transition_count = sum(1 for word in transition_words if word in content.lower())
        score += min(30, transition_count * 6)
        
        # Engagement (20 points)
        engaging_elements = ["example", "case study", "illustration", "demonstration"]
        engaging_count = sum(1 for element in engaging_elements if element in content.lower())
        score += min(20, engaging_count * 5)
        
        return min(max_score, score)

    def _score_innovation_dimension(self, content: str, title: str) -> float:
        """Score innovation and insight dimension"""
        score = 0
        max_score = 100
        
        # Novel insights (40 points)
        insight_words = ["innovative", "novel", "unique", "breakthrough", "revolutionary",
                        "cutting-edge", "advanced", "pioneering", "groundbreaking"]
        insight_count = sum(1 for word in insight_words if word in content.lower())
        score += min(40, insight_count * 8)
        
        # Creative problem-solving (30 points)
        problem_solving_words = ["solution", "approach", "strategy", "method", "technique",
                               "framework", "model", "system"]
        problem_count = sum(1 for word in problem_solving_words if word in content.lower())
        score += min(30, problem_count * 4)
        
        # Forward-thinking (30 points)
        future_words = ["future", "trend", "emerging", "potential", "opportunity",
                       "development", "evolution", "advancement"]
        future_count = sum(1 for word in future_words if word in content.lower())
        score += min(30, future_count * 5)
        
        return min(max_score, score)

    def _get_industry_benchmarks(self, industry: str) -> Dict[str, Any]:
        """Get industry-specific benchmarks"""
        benchmarks = {
            "technology": {
                "content_quality": 85,
                "professional_presentation": 80,
                "authority_credibility": 75,
                "technical_excellence": 90,
                "communication_effectiveness": 75,
                "innovation_insight": 85,
                "overall_threshold": 82
            },
            "healthcare": {
                "content_quality": 90,
                "professional_presentation": 85,
                "authority_credibility": 95,
                "technical_excellence": 85,
                "communication_effectiveness": 80,
                "innovation_insight": 70,
                "overall_threshold": 85
            },
            "finance": {
                "content_quality": 85,
                "professional_presentation": 90,
                "authority_credibility": 90,
                "technical_excellence": 80,
                "communication_effectiveness": 85,
                "innovation_insight": 75,
                "overall_threshold": 84
            },
            "education": {
                "content_quality": 80,
                "professional_presentation": 75,
                "authority_credibility": 85,
                "technical_excellence": 70,
                "communication_effectiveness": 90,
                "innovation_insight": 75,
                "overall_threshold": 79
            },
            "general": {
                "content_quality": 75,
                "professional_presentation": 70,
                "authority_credibility": 70,
                "technical_excellence": 65,
                "communication_effectiveness": 75,
                "innovation_insight": 65,
                "overall_threshold": 70
            }
        }
        
        return benchmarks.get(industry.lower(), benchmarks["general"])

    def _perform_competitive_analysis(self, dimensional_scores: Dict[str, float], industry: str) -> Dict[str, Any]:
        """Perform competitive analysis against industry standards"""
        benchmarks = self._get_industry_benchmarks(industry)
        
        analysis = {
            "industry": industry,
            "competitive_position": {},
            "strengths": [],
            "weaknesses": [],
            "market_percentile": 0,
            "competitive_advantages": [],
            "improvement_priorities": []
        }
        
        # Calculate competitive position for each dimension
        for dimension, score in dimensional_scores.items():
            benchmark = benchmarks.get(dimension, 70)
            if score >= benchmark + 10:
                analysis["competitive_position"][dimension] = "Above Market"
                analysis["strengths"].append(dimension)
            elif score >= benchmark:
                analysis["competitive_position"][dimension] = "Market Standard"
            elif score >= benchmark - 10:
                analysis["competitive_position"][dimension] = "Below Market"
                analysis["weaknesses"].append(dimension)
            else:
                analysis["competitive_position"][dimension] = "Significantly Below Market"
                analysis["weaknesses"].append(dimension)
                analysis["improvement_priorities"].append(dimension)
        
        # Calculate market percentile
        overall_score = self._calculate_overall_professional_score(dimensional_scores)
        overall_benchmark = benchmarks.get("overall_threshold", 70)
        
        if overall_score >= overall_benchmark + 15:
            analysis["market_percentile"] = 90
        elif overall_score >= overall_benchmark + 10:
            analysis["market_percentile"] = 80
        elif overall_score >= overall_benchmark + 5:
            analysis["market_percentile"] = 70
        elif overall_score >= overall_benchmark:
            analysis["market_percentile"] = 60
        elif overall_score >= overall_benchmark - 5:
            analysis["market_percentile"] = 50
        else:
            analysis["market_percentile"] = 30
        
        # Identify competitive advantages
        for strength in analysis["strengths"]:
            if dimensional_scores[strength] >= benchmarks.get(strength, 70) + 15:
                analysis["competitive_advantages"].append(f"Exceptional {strength.replace('_', ' ')}")
        
        return analysis

    def _calculate_overall_professional_score(self, dimensional_scores: Dict[str, float]) -> float:
        """Calculate weighted overall professional score"""
        weights = {
            "content_quality": 0.25,
            "professional_presentation": 0.20,
            "authority_credibility": 0.20,
            "technical_excellence": 0.15,
            "communication_effectiveness": 0.10,
            "innovation_insight": 0.10
        }
        
        weighted_score = 0
        total_weight = 0
        
        for dimension, score in dimensional_scores.items():
            weight = weights.get(dimension, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0

    def _generate_dimensional_professional_indicators(self, dimensional_scores: Dict[str, float], overall_score: float) -> Dict[str, Any]:
        """Generate professional indicators and certifications"""
        indicators = {
            "professional_level": "",
            "certifications_eligible": [],
            "industry_recognition": "",
            "quality_badges": [],
            "professional_status": "",
            "credibility_score": 0
        }
        
        # Determine professional level
        if overall_score >= 90:
            indicators["professional_level"] = "Expert"
            indicators["professional_status"] = "Industry Leader"
        elif overall_score >= 80:
            indicators["professional_level"] = "Advanced Professional"
            indicators["professional_status"] = "Senior Professional"
        elif overall_score >= 70:
            indicators["professional_level"] = "Professional"
            indicators["professional_status"] = "Qualified Professional"
        elif overall_score >= 60:
            indicators["professional_level"] = "Developing Professional"
            indicators["professional_status"] = "Emerging Professional"
        else:
            indicators["professional_level"] = "Entry Level"
            indicators["professional_status"] = "Developing"
        
        # Determine eligible certifications
        if dimensional_scores.get("authority_credibility", 0) >= 85:
            indicators["certifications_eligible"].append("Content Authority Certification")
        if dimensional_scores.get("technical_excellence", 0) >= 80:
            indicators["certifications_eligible"].append("Technical Excellence Badge")
        if dimensional_scores.get("professional_presentation", 0) >= 85:
            indicators["certifications_eligible"].append("Professional Communication Certificate")
        
        # Industry recognition level
        if overall_score >= 85:
            indicators["industry_recognition"] = "High Recognition"
        elif overall_score >= 75:
            indicators["industry_recognition"] = "Moderate Recognition"
        else:
            indicators["industry_recognition"] = "Building Recognition"
        
        # Quality badges
        for dimension, score in dimensional_scores.items():
            if score >= 85:
                indicators["quality_badges"].append(f"{dimension.replace('_', ' ').title()} Excellence")
        
        # Credibility score (0-100)
        indicators["credibility_score"] = min(100, int(overall_score))
        
        return indicators

    def _create_improvement_roadmap(self, dimensional_scores: Dict[str, float], benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """Create improvement roadmap based on gaps"""
        roadmap = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_objectives": [],
            "priority_matrix": {},
            "estimated_timeline": "",
            "resource_allocation": {}
        }
        
        # Identify gaps and create actions
        gaps = []
        for dimension, score in dimensional_scores.items():
            benchmark = benchmarks.get(dimension, 70)
            if score < benchmark:
                gap_size = benchmark - score
                gaps.append({
                    "dimension": dimension,
                    "gap": gap_size,
                    "priority": "High" if gap_size > 15 else "Medium" if gap_size > 10 else "Low"
                })
        
        # Sort by gap size
        gaps.sort(key=lambda x: x["gap"], reverse=True)
        
        # Create immediate actions (top 2 gaps)
        for gap in gaps[:2]:
            roadmap["immediate_actions"].append({
                "dimension": gap["dimension"],
                "action": f"Improve {gap['dimension'].replace('_', ' ')}",
                "target_increase": min(20, gap["gap"] + 5),
                "timeline": "1-2 weeks"
            })
        
        # Create short-term goals (next 2-3 gaps)
        for gap in gaps[2:5]:
            roadmap["short_term_goals"].append({
                "dimension": gap["dimension"],
                "goal": f"Achieve benchmark in {gap['dimension'].replace('_', ' ')}",
                "target_score": benchmarks.get(gap["dimension"], 70),
                "timeline": "1-2 months"
            })
        
        # Long-term objectives
        roadmap["long_term_objectives"] = [
            "Achieve industry leadership position",
            "Maintain consistent high-quality standards",
            "Develop competitive advantages in key areas"
        ]
        
        # Priority matrix
        for gap in gaps:
            impact = "High" if gap["gap"] > 15 else "Medium"
            effort = "Low" if gap["gap"] < 10 else "Medium" if gap["gap"] < 20 else "High"
            roadmap["priority_matrix"][gap["dimension"]] = {
                "impact": impact,
                "effort": effort,
                "priority": gap["priority"]
            }
        
        # Timeline estimation
        total_gaps = len(gaps)
        if total_gaps <= 2:
            roadmap["estimated_timeline"] = "2-4 weeks"
        elif total_gaps <= 4:
            roadmap["estimated_timeline"] = "1-3 months"
        else:
            roadmap["estimated_timeline"] = "3-6 months"
        
        return roadmap

    def _assess_certification_readiness(self, dimensional_scores: Dict[str, float], industry: str) -> Dict[str, Any]:
        """Assess readiness for professional certifications"""
        readiness = {
            "overall_readiness": "",
            "certification_scores": {},
            "requirements_met": [],
            "requirements_pending": [],
            "recommended_certifications": [],
            "preparation_needed": {}
        }
        
        # Define certification requirements
        certifications = {
            "Professional Content Creator": {
                "content_quality": 75,
                "professional_presentation": 70,
                "communication_effectiveness": 75
            },
            "Industry Expert": {
                "authority_credibility": 85,
                "technical_excellence": 80,
                "innovation_insight": 75
            },
            "Quality Assurance Professional": {
                "content_quality": 85,
                "authority_credibility": 80,
                "technical_excellence": 75
            }
        }
        
        # Assess each certification
        for cert_name, requirements in certifications.items():
            meets_requirements = True
            cert_score = 0
            pending = []
            
            for dimension, required_score in requirements.items():
                current_score = dimensional_scores.get(dimension, 0)
                cert_score += current_score
                
                if current_score >= required_score:
                    readiness["requirements_met"].append(f"{cert_name}: {dimension}")
                else:
                    meets_requirements = False
                    gap = required_score - current_score
                    pending.append(f"{dimension} (need +{gap:.1f})")
                    readiness["requirements_pending"].append(f"{cert_name}: {dimension}")
            
            avg_score = cert_score / len(requirements)
            readiness["certification_scores"][cert_name] = avg_score
            
            if meets_requirements:
                readiness["recommended_certifications"].append(cert_name)
            else:
                readiness["preparation_needed"][cert_name] = pending
        
        # Overall readiness assessment
        ready_count = len(readiness["recommended_certifications"])
        total_certs = len(certifications)
        
        if ready_count == total_certs:
            readiness["overall_readiness"] = "Fully Ready"
        elif ready_count >= total_certs * 0.7:
            readiness["overall_readiness"] = "Mostly Ready"
        elif ready_count >= total_certs * 0.3:
            readiness["overall_readiness"] = "Partially Ready"
        else:
            readiness["overall_readiness"] = "Preparation Needed"
        
        return readiness

    def _determine_market_positioning(self, overall_score: float, competitive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine market positioning and strategic recommendations"""
        positioning = {
            "market_tier": "",
            "competitive_strategy": "",
            "positioning_statement": "",
            "strategic_recommendations": [],
            "market_opportunities": [],
            "differentiation_factors": []
        }
        
        # Determine market tier
        percentile = competitive_analysis.get("market_percentile", 50)
        
        if percentile >= 90:
            positioning["market_tier"] = "Premium/Leadership"
            positioning["competitive_strategy"] = "Market Leadership"
            positioning["positioning_statement"] = "Industry-leading content with exceptional professional standards"
        elif percentile >= 80:
            positioning["market_tier"] = "High-Quality Professional"
            positioning["competitive_strategy"] = "Quality Differentiation"
            positioning["positioning_statement"] = "High-quality professional content exceeding industry standards"
        elif percentile >= 60:
            positioning["market_tier"] = "Market Standard"
            positioning["competitive_strategy"] = "Competitive Parity"
            positioning["positioning_statement"] = "Professional content meeting industry expectations"
        else:
            positioning["market_tier"] = "Developing/Entry"
            positioning["competitive_strategy"] = "Improvement Focus"
            positioning["positioning_statement"] = "Developing professional content with growth potential"
        
        # Strategic recommendations based on positioning
        if percentile >= 80:
            positioning["strategic_recommendations"] = [
                "Maintain leadership position through continuous innovation",
                "Leverage strengths for thought leadership opportunities",
                "Mentor others and establish industry presence"
            ]
        elif percentile >= 60:
            positioning["strategic_recommendations"] = [
                "Focus on differentiating strengths",
                "Address key weaknesses to move to premium tier",
                "Build authority through consistent quality"
            ]
        else:
            positioning["strategic_recommendations"] = [
                "Prioritize fundamental quality improvements",
                "Focus on meeting industry benchmarks",
                "Develop core professional competencies"
            ]
        
        # Market opportunities
        strengths = competitive_analysis.get("strengths", [])
        for strength in strengths:
            positioning["market_opportunities"].append(f"Leverage {strength.replace('_', ' ')} advantage")
        
        # Differentiation factors
        competitive_advantages = competitive_analysis.get("competitive_advantages", [])
        positioning["differentiation_factors"] = competitive_advantages
        
        return positioning

    def _get_default_professional_scoring(self) -> Dict[str, Any]:
        """Return default professional scoring in case of errors"""
        return {
            "overall_professional_score": 0,
            "dimensional_scores": {"error": "Scoring failed"},
            "industry_benchmarks": {"error": "Benchmarks unavailable"},
            "competitive_analysis": {"error": "Analysis failed"},
            "professional_indicators": {"error": "Indicators unavailable"},
            "improvement_roadmap": {"error": "Roadmap generation failed"},
            "certification_readiness": {"error": "Assessment failed"},
            "market_positioning": {"error": "Positioning analysis failed"}
        }