"""
Enhanced Summarization Service with Customizable Options and Text Highlighting
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Optional dependencies: import gracefully and fall back when unavailable
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

import nltk
from textblob import TextBlob

try:
    import spacy
except Exception:
    spacy = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModel = None

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class SummaryType(Enum):
    BRIEF = "brief"
    BALANCED = "balanced"
    DETAILED = "detailed"
    EXECUTIVE = "executive"

class DetailLevel(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    COMPREHENSIVE = "comprehensive"

class OutputFormat(Enum):
    PARAGRAPH = "paragraph"
    BULLET_POINTS = "bullet_points"
    MIXED = "mixed"

class FocusArea(Enum):
    MAIN_CONTENT = "main_content"
    KEY_FACTS = "key_facts"
    ACTIONABLE_ITEMS = "actionable_items"
    TECHNICAL_DETAILS = "technical_details"
    BUSINESS_INSIGHTS = "business_insights"
    TRENDS = "trends"

@dataclass
class SummaryCustomization:
    summary_type: SummaryType = SummaryType.BALANCED
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.MIXED
    focus_areas: List[FocusArea] = None
    highlight_relevant_text: bool = True
    include_keywords: bool = True
    max_length: Optional[int] = None
    user_query: Optional[str] = None

@dataclass
class HighlightedText:
    text: str
    relevance: str  # 'high', 'medium', 'low'
    context: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None

@dataclass
class EnhancedSummary:
    text: str
    key_points: List[str]
    highlights: List[HighlightedText]
    keywords: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence_score: float

class EnhancedSummarizationService:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.sentence_transformer = None
        self.nlp = None
        
        # Initialize OpenAI
        try:
            if OpenAI:
                self.openai_client = OpenAI()
            else:
                self.openai_client = None
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")
        
        # Initialize Anthropic
        try:
            if anthropic:
                self.anthropic_client = anthropic.Anthropic()
            else:
                self.anthropic_client = None
        except Exception as e:
            logger.warning(f"Anthropic initialization failed: {e}")
        
        # Initialize sentence transformer for semantic similarity
        try:
            if SentenceTransformer:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                self.sentence_transformer = None
        except Exception as e:
            logger.warning(f"SentenceTransformer initialization failed: {e}")
        
        # Initialize spaCy
        try:
            if spacy:
                self.nlp = spacy.load("en_core_web_sm")
            else:
                self.nlp = None
        except Exception as e:
            logger.warning(f"spaCy initialization failed: {e}")
            try:
                if spacy:
                    self.nlp = spacy.load("en_core_web_md")
                else:
                    self.nlp = None
            except:
                logger.warning("No spaCy model available")

    def generate_enhanced_summary(
        self, 
        content: str, 
        title: str = "", 
        url: str = "",
        customization: SummaryCustomization = None
    ) -> EnhancedSummary:
        """Generate an enhanced summary with customization options and highlighting."""
        
        if customization is None:
            customization = SummaryCustomization()
        
        try:
            # Use AI models if available, otherwise fallback to local processing
            if self.openai_client:
                return self._generate_ai_summary_openai(content, title, url, customization)
            elif self.anthropic_client:
                return self._generate_ai_summary_anthropic(content, title, url, customization)
            else:
                return self._generate_local_summary(content, title, url, customization)
        
        except Exception as e:
            logger.error(f"Enhanced summary generation failed: {e}")
            return self._generate_fallback_summary(content, title, customization)

    def _generate_ai_summary_openai(
        self, 
        content: str, 
        title: str, 
        url: str, 
        customization: SummaryCustomization
    ) -> EnhancedSummary:
        """Generate summary using OpenAI GPT models with enhanced accuracy and intelligence."""
        
        prompt = self._build_enhanced_prompt(content, title, customization)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for maximum accuracy and reasoning
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an elite AI content analyst with PhD-level expertise in linguistics, cognitive science, "
                            "and information architecture. Your specializations include:\n"
                            "• Advanced semantic analysis and contextual understanding\n"
                            "• Factual accuracy verification and source credibility assessment\n"
                            "• Multi-perspective analysis and bias detection\n"
                            "• Strategic insight extraction and actionable intelligence\n"
                            "• Professional communication and executive-level reporting\n\n"
                            "Your mission: Create summaries that are not just accurate, but intellectually enriching, "
                            "strategically valuable, and immediately actionable. Focus on:\n"
                            "1. Capturing nuanced meaning and implicit insights\n"
                            "2. Identifying patterns, trends, and strategic implications\n"
                            "3. Highlighting quantitative data and key metrics\n"
                            "4. Extracting actionable recommendations and next steps\n"
                            "5. Maintaining professional tone while ensuring accessibility\n\n"
                            "Always provide structured, JSON-formatted responses with rich metadata."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.05,  # Very low temperature for maximum consistency and accuracy
                max_tokens=3500,  # Increased token limit for comprehensive analysis
                top_p=0.85,  # Focused sampling for quality
                frequency_penalty=0.15,  # Reduce repetition more aggressively
                presence_penalty=0.2,  # Encourage diverse, comprehensive content
                response_format={"type": "json_object"}  # Ensure structured JSON output
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._parse_ai_response(result, content, customization)
            
        except Exception as e:
            logger.error(f"OpenAI summary generation failed: {e}")
            return self._generate_local_summary(content, title, url, customization)

    def _generate_ai_summary_anthropic(
        self, 
        content: str, 
        title: str, 
        url: str, 
        customization: SummaryCustomization
    ) -> EnhancedSummary:
        """Generate summary using Anthropic Claude."""
        
        prompt = self._build_enhanced_prompt(content, title, customization)
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            result = json.loads(response.content[0].text)
            return self._parse_ai_response(result, content, customization)
            
        except Exception as e:
            logger.error(f"Anthropic summary generation failed: {e}")
            return self._generate_local_summary(content, title, url, customization)

    def _build_enhanced_prompt(self, content: str, title: str, customization: SummaryCustomization) -> str:
        """Build sophisticated prompt for AI models with advanced analysis requirements."""
        
        # Preprocess content for better analysis
        processed_content = self._preprocess_content_for_analysis(content)
        
        # Advanced length requirements with quality metrics
        length_guide = {
            DetailLevel.SHORT: "1-2 concise sentences for summary, 3-4 strategic key points, 75-125 words total",
            DetailLevel.MEDIUM: "2-3 comprehensive sentences for summary, 5-7 actionable key points, 150-250 words total", 
            DetailLevel.LONG: "3-4 detailed sentences for summary, 7-10 strategic key points, 300-500 words total",
            DetailLevel.COMPREHENSIVE: "4-6 executive-level sentences for summary, 10-15 comprehensive key points, 500+ words total"
        }
        
        # Advanced focus area mapping with strategic context
        focus_mapping = {
            'main_content': 'core narrative, primary themes, and central arguments',
            'key_facts': 'verifiable data, statistics, concrete evidence, and factual claims',
            'actionable_items': 'specific recommendations, next steps, implementation strategies, and decision points',
            'technical_details': 'technical specifications, methodologies, processes, and system information',
            'business_insights': 'market implications, competitive advantages, revenue opportunities, and strategic value',
            'trends': 'emerging patterns, future projections, market movements, and predictive indicators'
        }
        
        # Enhanced focus instructions with strategic context
        focus_instructions = ""
        if customization.focus_areas:
            focus_details = [focus_mapping.get(area.value, area.value.replace('_', ' ')) for area in customization.focus_areas]
            focus_instructions = f"STRATEGIC FOCUS: Prioritize analysis of {', '.join(focus_details)}. Ensure these areas receive enhanced attention and detailed coverage in your analysis. "
        
        # Advanced user query processing with intent analysis
        query_context = ""
        highlighting_instructions = ""
        if customization.user_query:
            query_context = f"USER INTENT ANALYSIS: The user is specifically seeking information about: '{customization.user_query}'. Conduct intent analysis to understand underlying needs and provide comprehensive coverage of this topic with strategic context and actionable insights. "
            highlighting_instructions = f"PRECISION HIGHLIGHTING: Identify and mark ALL text segments with direct relevance to '{customization.user_query}'. Use semantic analysis to find related concepts, implications, and contextual information. Provide relevance scoring and detailed justification for each highlight. "
        
        # Content type analysis for adaptive prompting
        content_type_hint = self._analyze_content_type(processed_content, title)
        
        # Advanced prompt with multi-layered analysis framework
        prompt = f"""
You are an elite AI content strategist and senior business analyst with advanced degrees in Information Science, 
Business Intelligence, and Cognitive Psychology. You possess expert-level capabilities in:

CORE COMPETENCIES:
• Advanced semantic analysis with contextual reasoning and inference
• Multi-dimensional content architecture and information hierarchy mapping
• Strategic business intelligence extraction and competitive analysis
• Quantitative data interpretation and statistical significance assessment
• Cross-domain knowledge synthesis and pattern recognition
• Executive-level communication and strategic recommendation formulation
• Bias detection, fact verification, and credibility assessment
• Predictive analysis and trend identification

ANALYSIS FRAMEWORK:
1. SEMANTIC LAYER: Deep linguistic analysis, entity recognition, relationship mapping
2. STRATEGIC LAYER: Business implications, competitive intelligence, market positioning
3. QUANTITATIVE LAYER: Data extraction, metric analysis, statistical validation
4. ACTIONABLE LAYER: Recommendation synthesis, implementation pathways, decision support
5. QUALITY LAYER: Accuracy verification, source credibility, bias assessment

CONTENT CLASSIFICATION: {content_type_hint}

CUSTOMIZATION PARAMETERS:
• Summary Type: {customization.summary_type.value.upper()} (optimized for this content type)
• Detail Level: {customization.detail_level.value.upper()} - {length_guide[customization.detail_level]}
• Output Format: {customization.output_format.value.upper()} with professional structure
• {focus_instructions}{query_context}{highlighting_instructions}
• Advanced Highlighting: {customization.highlight_relevant_text}
• Enhanced Keywords: {customization.include_keywords}
• Length Constraint: {customization.max_length or 'Optimized for content depth'}

PROFESSIONAL EXCELLENCE STANDARDS:
• ACCURACY: Zero tolerance for speculation; fact-based analysis only
• CLARITY: Executive-level communication with technical precision
• COMPLETENESS: Comprehensive coverage without redundancy
• ACTIONABILITY: Strategic insights with implementation guidance
• RELEVANCE: User-centric analysis with contextual intelligence
• STRUCTURE: Logical flow with hierarchical information architecture

ADVANCED OUTPUT REQUIREMENTS:
• Use sophisticated business vocabulary with domain-specific terminology
• Integrate quantitative evidence with qualitative insights
• Provide multi-perspective analysis with balanced viewpoints
• Include confidence indicators and uncertainty acknowledgment
• Structure information by strategic importance and business impact
• Ensure scalable insights applicable across organizational levels

INTELLIGENT HIGHLIGHTING PROTOCOL:
• Semantic relevance scoring with contextual weighting
• Multi-dimensional relevance assessment (direct, indirect, strategic, tactical)
• Confidence scoring for each highlighted segment
• Categorization by information type and business value
• Cross-reference validation with user intent analysis

TARGET CONTENT ANALYSIS:
Title: "{title}"
Content Type: {content_type_hint}
Content Length: {len(processed_content)} characters
Content: {processed_content[:10000]}  # Enhanced content window for comprehensive analysis

INSTRUCTIONS:
1. Generate a professional-grade summary that matches the specified type and detail level exactly
2. Extract key points that align with the focus areas and user query with strategic importance
3. Identify and highlight ALL text segments relevant to the user's query with detailed reasoning
4. Extract important keywords with relevance weights, categories, and business context
5. Provide comprehensive confidence score based on content quality, completeness, and accuracy
6. Ensure all information is accurate, well-structured, and professionally presented
7. Include quantitative data, stakeholders, and business impact assessment

RESPONSE FORMAT (JSON):
{{
    "summary": {{
        "text": "Professional-grade summary with precise, business-focused language",
        "type": "{customization.summary_type.value}",
        "word_count": 0,
        "accuracy_notes": "Detailed notes on accuracy, completeness, and professional standards",
        "executive_summary": "2-3 sentence executive-level summary for C-level decision makers"
    }},
    "key_points": [
        {{
            "point": "Specific strategic key point with concrete details and business context",
            "importance": "high|medium|low",
            "category": "strategic|operational|financial|technical|market",
            "business_relevance": "Why this point matters for business decisions"
        }}
    ],
    "highlights": [
        {{
            "text": "exact text segment from content",
            "relevance": "high|medium|low",
            "context": "why this is relevant to the user's specific query and business needs",
            "reasoning": "detailed explanation of relevance, importance, and business implications",
            "category": "factual|strategic|actionable|quantitative|stakeholder",
            "user_query_relevance": "specific connection to user's query: '{customization.user_query}'",
            "business_impact": "potential business impact or strategic importance"
        }}
    ],
    "keywords": [
        {{
            "text": "keyword or phrase",
            "weight": 0.95,
            "category": "strategic|operational|technical|market|financial",
            "context": "business context and strategic relevance",
            "user_query_relevance": "relevance to user's specific interest"
        }}
    ],
    "insights": [
        {{
            "insight": "strategic insight with business implications",
            "type": "market_trend|competitive_intelligence|strategic_implication|operational_insight",
            "confidence": 0.9,
            "business_impact": "potential impact on business strategy or operations"
        }}
    ],
    "actionable_items": [
        {{
            "action": "specific strategic or operational action",
            "priority": "high|medium|low",
            "context": "business context and strategic importance",
            "stakeholder": "who should take this action",
            "timeline": "suggested timeline or urgency"
        }}
    ],
    "quantitative_data": [
        {{
            "metric": "specific number, percentage, or data point",
            "context": "what this metric represents",
            "significance": "why this data point is important",
            "source": "where in the content this data appears"
        }}
    ],
    "stakeholders": [
        {{
            "entity": "person, organization, or group mentioned",
            "role": "their role or relevance",
            "importance": "why they matter to the content",
            "connection_to_query": "how they relate to user's query"
        }}
    ],
    "metadata": {{
        "confidence": 0.95,
        "content_type": "detected content type",
        "focus_alignment": 0.90,
        "completeness": 0.85,
        "accuracy_score": 0.92,
        "professional_grade": true,
        "business_relevance": 0.88,
        "user_query_coverage": 0.90,
        "processing_notes": "comprehensive notes about analysis quality and professional standards"
    }}
}}

Ensure the summary is professional-grade, highly accurate, comprehensive, and tailored to the user's specific requirements. 
Focus on precision, completeness, actionable insights, and intelligent highlighting of user-relevant content.
"""
        
        return prompt

    def _preprocess_content_for_analysis(self, content: str) -> str:
        """Enhanced content preprocessing with intelligent analysis and smart truncation."""
        
        # Remove HTML tags and clean up
        import re
        processed = re.sub(r'<[^>]+>', ' ', content)
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove common web noise patterns with enhanced patterns
        noise_patterns = [
            r'cookie|privacy|terms|conditions|gdpr|ccpa',
            r'follow us|subscribe|newsletter|sign up|join',
            r'advertisement|sponsored|promoted|ads by|affiliate',
            r'click here|read more|learn more|see more|view all',
            r'share|tweet|like|follow|facebook|twitter|linkedin',
            r'copyright|all rights reserved|\u00a9|\(c\)',
            r'loading|please wait|error|404|not found',
            r'menu|navigation|breadcrumb|sidebar|footer',
            r'search|filter|sort by|page \d+|next|previous',
        ]
        
        for pattern in noise_patterns:
            processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
        
        # Remove excessive punctuation and special characters
        processed = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', processed)
        processed = re.sub(r'\.{3,}', '...', processed)  # Normalize ellipsis
        processed = re.sub(r'[\!\?]{2,}', '!', processed)  # Normalize exclamation/question marks
        
        # Clean up extra whitespace and normalize
        processed = re.sub(r'\s+', ' ', processed.strip())
        
        # Smart truncation based on content structure
        processed = self._apply_smart_truncation(processed)
        
        return processed

    def _apply_smart_truncation(self, content: str, max_chars: int = 15000) -> str:
        """Apply intelligent truncation that preserves content structure and meaning."""
        
        if len(content) <= max_chars:
            return content
        
        # Split into sentences for intelligent truncation
        sentences = self._extract_sentences(content)
        
        # Calculate sentence importance scores
        sentence_scores = self._calculate_sentence_importance(sentences)
        
        # Select most important sentences that fit within limit
        selected_sentences = []
        current_length = 0
        
        # Sort sentences by importance (descending)
        sorted_sentences = sorted(
            enumerate(sentences), 
            key=lambda x: sentence_scores.get(x[0], 0), 
            reverse=True
        )
        
        # Add sentences in order of importance until we reach the limit
        for idx, sentence in sorted_sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_chars:
                selected_sentences.append((idx, sentence))
                current_length += sentence_length
            else:
                break
        
        # Sort selected sentences back to original order
        selected_sentences.sort(key=lambda x: x[0])
        
        # Join sentences and ensure coherent flow
        truncated_content = ' '.join([sentence for _, sentence in selected_sentences])
        
        # Add truncation indicator if content was shortened
        if len(truncated_content) < len(content):
            truncated_content += " [Content intelligently truncated for optimal analysis]"
        
        return truncated_content

    def _calculate_sentence_importance(self, sentences: List[str]) -> Dict[int, float]:
        """Calculate importance scores for sentences based on multiple factors."""
        
        scores = {}
        total_sentences = len(sentences)
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Position-based scoring (beginning and end are more important)
            if i < total_sentences * 0.2:  # First 20%
                score += 0.3
            elif i > total_sentences * 0.8:  # Last 20%
                score += 0.2
            
            # Length-based scoring (moderate length preferred)
            sentence_length = len(sentence.split())
            if 10 <= sentence_length <= 30:
                score += 0.2
            elif sentence_length > 30:
                score += 0.1
            
            # Content quality indicators
            if any(keyword in sentence.lower() for keyword in [
                'important', 'key', 'main', 'primary', 'essential', 'critical',
                'significant', 'major', 'fundamental', 'core', 'central'
            ]):
                score += 0.3
            
            # Technical/business content indicators
            if any(keyword in sentence.lower() for keyword in [
                'result', 'conclusion', 'finding', 'data', 'analysis',
                'research', 'study', 'report', 'strategy', 'solution'
            ]):
                score += 0.2
            
            # Numerical data presence (often important)
            if re.search(r'\d+%|\$\d+|\d+\.\d+|\d{4}', sentence):
                score += 0.2
            
            # Question sentences (often important for context)
            if sentence.strip().endswith('?'):
                score += 0.1
            
            scores[i] = score
        
        return scores

    def _extract_sentences(self, content: str) -> List[str]:
        """Extract sentences from content using intelligent sentence boundary detection."""
        
        # Use NLTK for better sentence tokenization if available
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Ensure punkt tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            sentences = sent_tokenize(content)
        except ImportError:
            # Fallback to regex-based sentence splitting
            sentences = self._regex_sentence_split(content)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences (likely fragments)
            if len(sentence) > 10 and len(sentence.split()) > 2:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _regex_sentence_split(self, content: str) -> List[str]:
        """Fallback regex-based sentence splitting."""
        
        # Split on sentence endings, but be careful with abbreviations
        import re
        
        # First, protect common abbreviations
        protected = re.sub(r'\b(Mr|Mrs|Dr|Prof|Inc|Ltd|Co|etc|vs|i\.e|e\.g)\.', r'\1<DOT>', content)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', protected)
        
        # Restore protected dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return sentences

    def _analyze_content_type(self, content: str, title: str) -> str:
        """Analyze content type for adaptive prompting and processing."""
        
        content_lower = content.lower()
        title_lower = title.lower() if title else ""
        
        # Technical content indicators
        technical_indicators = ['api', 'algorithm', 'framework', 'implementation', 'code', 'technical', 'system', 'architecture', 'database', 'software']
        
        # Business content indicators
        business_indicators = ['revenue', 'market', 'strategy', 'business', 'financial', 'investment', 'growth', 'profit', 'customer', 'sales']
        
        # News/article indicators
        news_indicators = ['reported', 'according to', 'sources', 'breaking', 'update', 'announced', 'statement', 'press release']
        
        # Academic/research indicators
        academic_indicators = ['study', 'research', 'analysis', 'findings', 'methodology', 'conclusion', 'abstract', 'hypothesis']
        
        # Count indicators
        technical_score = sum(1 for indicator in technical_indicators if indicator in content_lower or indicator in title_lower)
        business_score = sum(1 for indicator in business_indicators if indicator in content_lower or indicator in title_lower)
        news_score = sum(1 for indicator in news_indicators if indicator in content_lower)
        academic_score = sum(1 for indicator in academic_indicators if indicator in content_lower or indicator in title_lower)
        
        # Determine content type
        scores = {
            'Technical Documentation': technical_score,
            'Business Content': business_score,
            'News Article': news_score,
            'Academic/Research': academic_score
        }
        
        # Get the highest scoring type
        content_type = max(scores, key=scores.get)
        
        # If no clear winner, analyze structure
        if scores[content_type] == 0:
            if len(content.split('\n')) > 10:  # Multiple paragraphs
                content_type = 'Long-form Article'
            elif '?' in content:  # Questions present
                content_type = 'FAQ/Q&A Content'
            elif re.search(r'\d+\.|\*|\-', content):  # Lists present
                content_type = 'Structured List Content'
            else:
                content_type = 'General Content'
        
        return content_type

    def _parse_ai_response(self, result: Dict, content: str, customization: SummaryCustomization) -> EnhancedSummary:
        """Parse AI response into EnhancedSummary object."""
        
        # Extract highlights
        highlights = []
        for highlight_data in result.get('highlights', []):
            highlight = HighlightedText(
                text=highlight_data.get('text', ''),
                relevance=highlight_data.get('relevance', 'medium'),
                context=highlight_data.get('context', '')
            )
            highlights.append(highlight)
        
        # Extract keywords
        keywords = result.get('keywords', [])
        
        # Extract metadata
        metadata = result.get('metadata', {})
        metadata.update({
            'customization': {
                'summary_type': customization.summary_type.value,
                'detail_level': customization.detail_level.value,
                'output_format': customization.output_format.value,
                'focus_areas': [area.value for area in (customization.focus_areas or [])]
            }
        })
        
        return EnhancedSummary(
            text=result.get('summary', {}).get('text', ''),
            key_points=result.get('key_points', []),
            highlights=highlights,
            keywords=keywords,
            metadata=metadata,
            confidence_score=metadata.get('confidence', 0.7)
        )

    def _generate_local_summary(
        self, 
        content: str, 
        title: str, 
        url: str, 
        customization: SummaryCustomization
    ) -> EnhancedSummary:
        """Generate summary using local NLP processing."""
        
        try:
            # Extract sentences
            sentences = self._extract_sentences(content)
            
            # Extract keywords
            keywords = self._extract_enhanced_keywords(content)
            
            # Generate summary based on customization
            summary_text, key_points = self._generate_customized_local_summary(
                sentences, keywords, customization
            )
            
            # Generate highlights
            highlights = self._generate_highlights(
                content, summary_text, customization.user_query
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(content, summary_text, key_points)
            
            metadata = {
                'method': 'local_nlp',
                'content_length': len(content),
                'sentence_count': len(sentences),
                'keyword_count': len(keywords),
                'confidence': confidence
            }
            
            return EnhancedSummary(
                text=summary_text,
                key_points=key_points,
                highlights=highlights,
                keywords=keywords,
                metadata=metadata,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Local summary generation failed: {e}")
            return self._generate_fallback_summary(content, title, customization)

    def _extract_sentences(self, content: str) -> List[str]:
        """Extract and clean sentences from content."""
        # Clean content
        content = re.sub(r"\s+", " ", content).strip()

        # Prefer NLTK's sentence tokenizer for better boundaries
        try:
            sentences = nltk.sent_tokenize(content)
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", content)

        # Filter, normalize, and lightly smooth sentences
        cleaned_sentences = []
        for sentence in sentences:
            s = sentence.strip()
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"\s*,\s*", ", ", s)
            # Ensure terminal punctuation
            if s and s[-1] not in ".!?":
                s += "."
            # Keep informative, medium-length sentences
            word_count = len(s.split())
            if 8 <= word_count <= 40 and s[0].isupper():
                cleaned_sentences.append(s)

        return cleaned_sentences

    def _extract_enhanced_keywords(self, content: str) -> List[Dict[str, Any]]:
        """Extract keywords with enhanced processing."""
        keywords = []
        
        try:
            # Use spaCy if available
            if self.nlp:
                doc = self.nlp(content)
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        keywords.append({
                            'text': ent.text,
                            'weight': 0.9,
                            'category': 'entity',
                            'type': ent.label_
                        })
                
                # Extract important nouns and adjectives
                for token in doc:
                    if (token.pos_ in ['NOUN', 'ADJ'] and 
                        not token.is_stop and 
                        len(token.text) > 3 and
                        token.is_alpha):
                        keywords.append({
                            'text': token.lemma_,
                            'weight': 0.7,
                            'category': 'descriptive',
                            'type': token.pos_
                        })
            
            # Fallback to TextBlob
            else:
                blob = TextBlob(content)
                # Include noun phrases as keyphrases to capture multi-word concepts
                try:
                    for phrase in blob.noun_phrases:
                        phrase = phrase.strip()
                        if len(phrase) > 3:
                            keywords.append({
                                'text': phrase.lower(),
                                'weight': 0.8,
                                'category': 'keyphrase',
                                'type': 'NP'
                            })
                except Exception:
                    pass
                for word, pos in blob.tags:
                    if (pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] and
                        len(word) > 3 and
                        word.isalpha()):
                        keywords.append({
                            'text': word.lower(),
                            'weight': 0.6,
                            'category': 'basic',
                            'type': pos
                        })
        
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
        
        # Remove duplicates and sort by weight
        unique_keywords = {}
        for kw in keywords:
            text = kw['text'].lower()
            if text not in unique_keywords or kw['weight'] > unique_keywords[text]['weight']:
                unique_keywords[text] = kw
        
        return sorted(unique_keywords.values(), key=lambda x: x['weight'], reverse=True)[:20]

    def _generate_customized_local_summary(
        self, 
        sentences: List[str], 
        keywords: List[Dict[str, Any]], 
        customization: SummaryCustomization
    ) -> Tuple[str, List[str]]:
        """Generate customized summary using local processing."""
        
        # Score sentences based on customization
        sentence_scores = self._score_sentences(sentences, keywords, customization)
        
        # Determine number of sentences based on detail level
        num_sentences = {
            DetailLevel.SHORT: 2,
            DetailLevel.MEDIUM: 3,
            DetailLevel.LONG: 4,
            DetailLevel.COMPREHENSIVE: 5
        }[customization.detail_level]
        
        # Select top sentences
        top_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_sentences]
        
        # Sort by original order
        selected_indices = sorted([idx for idx, _ in top_sentences])
        summary_sentences = [sentences[idx] for idx in selected_indices]
        
        # Create summary text
        summary_text = ' '.join(summary_sentences)
        summary_text = self._polish_summary_text(summary_text)
        
        # Generate key points
        key_points = self._generate_key_points(sentences, keywords, customization)
        key_points = [self._polish_key_point(kp) for kp in key_points]
        
        return summary_text, key_points

    def _score_sentences(
        self, 
        sentences: List[str], 
        keywords: List[Dict[str, Any]], 
        customization: SummaryCustomization
    ) -> Dict[int, float]:
        """Score sentences based on customization preferences."""
        
        scores = {}
        keyword_texts = [kw['text'].lower() for kw in keywords[:10]]
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            sentence_lower = sentence.lower()
            
            # Keyword relevance
            keyword_score = sum(
                sentence_lower.count(kw) * (keywords[j]['weight'] if j < len(keywords) else 0.5)
                for j, kw in enumerate(keyword_texts)
            )
            score += keyword_score * 2
            
            # Position bonus (first sentences often important)
            position_bonus = max(0, 1.0 - (i * 0.1))
            score += position_bonus
            
            # Length preference (medium length preferred)
            length = len(sentence.split())
            if 15 <= length <= 30:
                score += 1.0
            elif 10 <= length <= 40:
                score += 0.5

            # Grammar/clarity heuristics: fewer commas and presence of clear verbs
            comma_count = sentence.count(',')
            if comma_count == 0:
                score += 0.3
            elif comma_count <= 2:
                score += 0.1
            if re.search(r"\b(is|are|was|were|has|have|introduces|enables|drives|improves|reduces|supports)\b", sentence_lower):
                score += 0.3
            
            # Focus area bonuses
            if customization.focus_areas:
                for focus in customization.focus_areas:
                    if focus == FocusArea.KEY_FACTS and any(
                        indicator in sentence_lower 
                        for indicator in ['important', 'key', 'main', 'primary', 'significant']
                    ):
                        score += 1.0
                    elif focus == FocusArea.ACTIONABLE_ITEMS and any(
                        indicator in sentence_lower 
                        for indicator in ['should', 'must', 'need', 'require', 'action', 'step']
                    ):
                        score += 1.0
                    elif focus == FocusArea.TECHNICAL_DETAILS and any(
                        indicator in sentence_lower 
                        for indicator in ['technical', 'system', 'method', 'process', 'algorithm']
                    ):
                        score += 1.0
            
            scores[i] = score
        
        return scores

    def _generate_key_points(
        self, 
        sentences: List[str], 
        keywords: List[Dict[str, Any]], 
        customization: SummaryCustomization
    ) -> List[str]:
        """Generate key points based on customization."""
        
        # Determine number of key points
        num_points = {
            DetailLevel.SHORT: 3,
            DetailLevel.MEDIUM: 5,
            DetailLevel.LONG: 7,
            DetailLevel.COMPREHENSIVE: 10
        }[customization.detail_level]
        
        # Score sentences for key points (different from summary scoring)
        key_point_scores = {}
        keyword_texts = [kw['text'].lower() for kw in keywords[:15]]
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            sentence_lower = sentence.lower()
            
            # Look for specific patterns that make good key points
            if any(pattern in sentence_lower for pattern in [
                'key', 'important', 'significant', 'main', 'primary', 
                'notable', 'crucial', 'essential', 'major'
            ]):
                score += 2.0
            
            # Numbers and statistics make good key points
            if re.search(r'\d+', sentence):
                score += 1.5
            
            # Keyword density
            keyword_count = sum(sentence_lower.count(kw) for kw in keyword_texts)
            score += keyword_count * 0.5
            
            # Sentence structure (avoid very long or very short)
            length = len(sentence.split())
            if 8 <= length <= 25:
                score += 1.0
            
            key_point_scores[i] = score
        
        # Select top sentences for key points
        top_key_points = sorted(
            key_point_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_points]
        
        # Format as key points
        key_points = []
        for idx, _ in top_key_points:
            sentence = sentences[idx].strip()
            # Trim punctuation and subordinate clauses to keep bullets focused
            sentence = sentence.rstrip('.!?')
            sentence = re.split(r"\b(which|that|because|however|although|whereas)\b", sentence, maxsplit=1)[0].strip()
            sentence = sentence.split(',')[0].strip()
            key_points.append(sentence)
        
        return key_points

    def _polish_summary_text(self, text: str) -> str:
        """Light grammar and readability smoothing for locally generated summaries."""
        try:
            t = re.sub(r"\s+", " ", text).strip()
            t = re.sub(r"\s*,\s*", ", ", t)
            t = re.sub(r"\s*\.\s*", ". ", t)
            t = re.sub(r"\s*;\s*", "; ", t)
            sentences = re.split(r"(?<=[.!?])\s+", t)
            cleaned = []
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if s and not s[0].isupper():
                    s = s[0].upper() + s[1:]
                s = re.sub(r"\b(and|but|or)\s*$", "", s, flags=re.IGNORECASE).strip()
                if s and s[-1] not in ".!?":
                    s += "."
                cleaned.append(s)
            return " ".join(cleaned)
        except Exception:
            return text

    def _polish_key_point(self, text: str) -> str:
        """Polish bullet text for clarity and grammar."""
        t = re.sub(r"\s+", " ", text).strip()
        if t and not t[0].isupper():
            t = t[0].upper() + t[1:]
        t = re.sub(r"[,:;\s]+$", "", t)
        return t

    def _generate_highlights(
        self, 
        content: str, 
        summary: str, 
        user_query: Optional[str] = None
    ) -> List[HighlightedText]:
        """Generate intelligent highlights with enhanced relevance scoring and semantic analysis."""
        
        highlights = []
        
        try:
            # Find query-relevant highlights if user query exists
            if user_query:
                query_highlights = self._find_query_relevant_highlights(content, user_query)
                highlights.extend(query_highlights)
            
            # Find summary-relevant highlights
            summary_highlights = self._find_summary_relevant_highlights(content, summary)
            highlights.extend(summary_highlights)
            
            # Add semantic highlights based on content analysis
            semantic_highlights = self._find_semantic_highlights(content, summary, user_query)
            highlights.extend(semantic_highlights)
            
            # Enhanced deduplication with similarity checking
            unique_highlights = self._deduplicate_highlights(highlights)
            
            # Advanced relevance scoring
            scored_highlights = self._score_highlight_relevance(unique_highlights, content, summary, user_query)
            
            # Sort by enhanced relevance score
            scored_highlights.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Convert back to HighlightedText objects
            final_highlights = []
            for item in scored_highlights[:15]:  # Limit to top 15 highlights
                highlight = item['highlight']
                # Update relevance based on score
                if item['relevance_score'] >= 0.8:
                    highlight.relevance = 'high'
                elif item['relevance_score'] >= 0.5:
                    highlight.relevance = 'medium'
                else:
                    highlight.relevance = 'low'
                
                final_highlights.append(highlight)
            
            return final_highlights
            
        except Exception as e:
            logger.error(f"Enhanced highlight generation failed: {e}")
            # Fallback to basic highlighting
            return self._generate_basic_highlights(content, summary, user_query)

    def _generate_basic_highlights(self, content: str, summary: str, user_query: Optional[str] = None) -> List[HighlightedText]:
        """Fallback basic highlighting method."""
        
        highlights = []
        
        try:
            # If user query provided, find relevant segments
            if user_query:
                highlights.extend(self._find_query_relevant_highlights(content, user_query))
            
            # Find segments mentioned in summary
            highlights.extend(self._find_summary_relevant_highlights(content, summary))
            
            # Remove duplicates and sort by relevance
            unique_highlights = {}
            for highlight in highlights:
                key = highlight.text.lower()
                if key not in unique_highlights or highlight.relevance == 'high':
                    unique_highlights[key] = highlight
            
            return list(unique_highlights.values())[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Basic highlight generation failed: {e}")
            return []

    def _find_query_relevant_highlights(self, content: str, user_query: str) -> List[HighlightedText]:
        """Find text segments relevant to user query using semantic similarity."""
        
        highlights = []
        
        try:
            if not self.sentence_transformer:
                return highlights
            
            # Split content into segments
            segments = re.split(r'[.!?]\s+', content)
            segments = [seg.strip() for seg in segments if len(seg.strip()) > 20]
            
            if not segments:
                return highlights
            
            # Encode query and segments
            query_embedding = self.sentence_transformer.encode([user_query])
            segment_embeddings = self.sentence_transformer.encode(segments)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, segment_embeddings)[0]
            
            # Find most relevant segments
            for i, similarity in enumerate(similarities):
                if similarity > 0.3:  # Threshold for relevance
                    relevance = 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                    highlights.append(HighlightedText(
                        text=segments[i],
                        relevance=relevance,
                        context=f"Relevant to query: '{user_query}'"
                    ))
        
        except Exception as e:
            logger.error(f"Query-based highlighting failed: {e}")
        
        return highlights

    def _find_summary_relevant_highlights(self, content: str, summary: str) -> List[HighlightedText]:
        """Find text segments that support the summary."""
        
        highlights = []
        
        try:
            # Extract key phrases from summary
            summary_words = set(word.lower() for word in re.findall(r'\b\w+\b', summary) if len(word) > 3)
            
            # Split content into segments
            segments = re.split(r'[.!?]\s+', content)
            segments = [seg.strip() for seg in segments if len(seg.strip()) > 20]
            
            for segment in segments:
                segment_words = set(word.lower() for word in re.findall(r'\b\w+\b', segment))
                
                # Calculate word overlap
                overlap = len(summary_words.intersection(segment_words))
                overlap_ratio = overlap / len(summary_words) if summary_words else 0
                
                if overlap_ratio > 0.2:  # Threshold for relevance
                    relevance = 'high' if overlap_ratio > 0.5 else 'medium' if overlap_ratio > 0.3 else 'low'
                    highlights.append(HighlightedText(
                        text=segment,
                        relevance=relevance,
                        context="Supports main summary points"
                    ))
        
        except Exception as e:
            logger.error(f"Summary-based highlighting failed: {e}")
        
        return highlights

    def _find_semantic_highlights(self, content: str, summary: str, user_query: Optional[str] = None) -> List[HighlightedText]:
        """Find semantically important highlights using advanced NLP techniques."""
        
        highlights = []
        
        try:
            # Extract sentences for analysis
            sentences = self._extract_sentences(content)
            
            # Calculate semantic importance scores
            semantic_scores = self._calculate_semantic_importance(sentences, summary, user_query)
            
            # Select top semantic highlights
            for i, sentence in enumerate(sentences):
                score = semantic_scores.get(i, 0)
                
                if score > 0.6:  # High semantic relevance threshold
                    relevance = 'high' if score > 0.8 else 'medium'
                    
                    # Find position in original content
                    start_pos = content.find(sentence)
                    end_pos = start_pos + len(sentence) if start_pos != -1 else None
                    
                    highlights.append(HighlightedText(
                        text=sentence,
                        relevance=relevance,
                        context=f"Semantic relevance score: {score:.2f}",
                        start_pos=start_pos,
                        end_pos=end_pos
                    ))
            
        except Exception as e:
            logger.error(f"Semantic highlighting failed: {e}")
        
        return highlights

    def _calculate_semantic_importance(self, sentences: List[str], summary: str, user_query: Optional[str] = None) -> Dict[int, float]:
        """Calculate semantic importance scores for sentences."""
        
        scores = {}
        
        try:
            # Combine summary and user query for relevance calculation
            reference_text = summary
            if user_query:
                reference_text += " " + user_query
            
            # Use sentence transformer if available
            if self.sentence_transformer:
                # Encode sentences and reference text
                sentence_embeddings = self.sentence_transformer.encode(sentences)
                reference_embedding = self.sentence_transformer.encode([reference_text])
                
                # Calculate cosine similarities
                similarities = cosine_similarity(sentence_embeddings, reference_embedding).flatten()
                
                for i, similarity in enumerate(similarities):
                    scores[i] = float(similarity)
            else:
                # Fallback to keyword-based scoring
                reference_words = set(reference_text.lower().split())
                
                for i, sentence in enumerate(sentences):
                    sentence_words = set(sentence.lower().split())
                    overlap = len(reference_words.intersection(sentence_words))
                    scores[i] = overlap / max(len(reference_words), len(sentence_words), 1)
        
        except Exception as e:
            logger.error(f"Semantic scoring failed: {e}")
        
        return scores

    def _deduplicate_highlights(self, highlights: List[HighlightedText]) -> List[HighlightedText]:
        """Enhanced deduplication with similarity checking."""
        
        unique_highlights = []
        seen_texts = set()
        
        for highlight in highlights:
            # Simple text-based deduplication
            text_key = highlight.text.lower().strip()
            
            if text_key not in seen_texts and len(text_key) > 10:  # Minimum length filter
                unique_highlights.append(highlight)
                seen_texts.add(text_key)
        
        return unique_highlights

    def _score_highlight_relevance(
        self, 
        highlights: List[HighlightedText], 
        content: str, 
        summary: str, 
        user_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Advanced relevance scoring for highlights."""
        
        scored_highlights = []
        
        for highlight in highlights:
            score = 0.0
            
            # Base relevance score
            if highlight.relevance == 'high':
                score += 0.4
            elif highlight.relevance == 'medium':
                score += 0.2
            else:
                score += 0.1
            
            # Length-based scoring (prefer moderate length)
            text_length = len(highlight.text.split())
            if 5 <= text_length <= 25:
                score += 0.2
            elif text_length > 25:
                score += 0.1
            
            # Position-based scoring (beginning and end are important)
            if highlight.start_pos is not None:
                content_length = len(content)
                relative_pos = highlight.start_pos / content_length
                
                if relative_pos < 0.2 or relative_pos > 0.8:  # First 20% or last 20%
                    score += 0.1
            
            # Content quality indicators
            text_lower = highlight.text.lower()
            
            # Important keywords boost
            if any(keyword in text_lower for keyword in [
                'important', 'key', 'main', 'significant', 'critical',
                'essential', 'primary', 'major', 'fundamental'
            ]):
                score += 0.2
            
            # Data/numbers boost
            if re.search(r'\d+%|\$\d+|\d+\.\d+|\d{4}', highlight.text):
                score += 0.1
            
            # User query relevance boost
            if user_query:
                query_words = set(user_query.lower().split())
                highlight_words = set(text_lower.split())
                overlap = len(query_words.intersection(highlight_words))
                if overlap > 0:
                    score += 0.3 * (overlap / len(query_words))
            
            scored_highlights.append({
                'highlight': highlight,
                'relevance_score': min(score, 1.0)  # Cap at 1.0
            })
        
        return scored_highlights

    def _calculate_confidence(self, content: str, summary: str, key_points: List[str]) -> float:
        """Calculate confidence score for the generated summary using comprehensive quality metrics."""
        
        try:
            # Calculate individual quality metrics
            quality_metrics = self._calculate_quality_metrics(content, summary, key_points)
            
            # Weighted combination of quality metrics
            weights = {
                'content_coverage': 0.25,
                'length_appropriateness': 0.15,
                'coherence': 0.20,
                'information_density': 0.15,
                'key_points_quality': 0.15,
                'factual_consistency': 0.10
            }
            
            confidence = sum(
                quality_metrics.get(metric, 0.5) * weight 
                for metric, weight in weights.items()
            )
            
            return min(1.0, max(0.1, confidence))  # Ensure between 0.1 and 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.6

    def _calculate_quality_metrics(self, content: str, summary: str, key_points: List[str]) -> Dict[str, float]:
        """Calculate detailed quality metrics for summary assessment."""
        
        metrics = {}
        
        try:
            # Content Coverage Score
            metrics['content_coverage'] = self._calculate_content_coverage(content, summary)
            
            # Length Appropriateness Score
            metrics['length_appropriateness'] = self._calculate_length_appropriateness(content, summary)
            
            # Coherence Score
            metrics['coherence'] = self._calculate_coherence_score(summary)
            
            # Information Density Score
            metrics['information_density'] = self._calculate_information_density(summary, key_points)
            
            # Key Points Quality Score
            metrics['key_points_quality'] = self._calculate_key_points_quality(key_points)
            
            # Factual Consistency Score
            metrics['factual_consistency'] = self._calculate_factual_consistency(content, summary)
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            # Return default scores if calculation fails
            metrics = {
                'content_coverage': 0.6,
                'length_appropriateness': 0.6,
                'coherence': 0.6,
                'information_density': 0.6,
                'key_points_quality': 0.6,
                'factual_consistency': 0.6
            }
        
        return metrics

    def _calculate_content_coverage(self, content: str, summary: str) -> float:
        """Calculate how well the summary covers the original content."""
        
        try:
            # Extract key terms from content and summary
            content_words = set(re.findall(r'\b\w+\b', content.lower()))
            summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            content_keywords = content_words - stop_words
            summary_keywords = summary_words - stop_words
            
            if not content_keywords:
                return 0.5
            
            # Calculate coverage ratio
            coverage = len(summary_keywords.intersection(content_keywords)) / len(content_keywords)
            
            # Normalize to 0-1 scale with reasonable thresholds
            if coverage >= 0.3:
                return min(1.0, coverage * 2)  # Scale up good coverage
            else:
                return coverage * 1.5  # Give some credit for partial coverage
                
        except:
            return 0.5

    def _calculate_length_appropriateness(self, content: str, summary: str) -> float:
        """Calculate if summary length is appropriate for content length."""
        
        try:
            content_length = len(content.split())
            summary_length = len(summary.split())
            
            if content_length == 0:
                return 0.5
            
            # Calculate compression ratio
            compression_ratio = summary_length / content_length
            
            # Ideal compression ratios based on content length
            if content_length < 200:
                ideal_ratio = 0.4  # Less compression for short content
            elif content_length < 1000:
                ideal_ratio = 0.2  # Moderate compression
            else:
                ideal_ratio = 0.1  # High compression for long content
            
            # Score based on how close to ideal ratio
            ratio_diff = abs(compression_ratio - ideal_ratio)
            score = max(0.0, 1.0 - (ratio_diff * 5))  # Penalize deviation
            
            return score
            
        except:
            return 0.5

    def _calculate_coherence_score(self, summary: str) -> float:
        """Calculate coherence and readability of the summary."""
        
        try:
            score = 0.0
            
            # Basic structure checks
            if summary and len(summary.strip()) > 0:
                score += 0.2
            
            # Proper capitalization
            if summary and summary[0].isupper():
                score += 0.1
            
            # Proper ending punctuation
            if summary and summary.rstrip()[-1] in '.!?':
                score += 0.1
            
            # Sentence structure
            sentences = re.split(r'[.!?]+', summary)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            if len(valid_sentences) >= 2:
                score += 0.2
            
            # Check for transition words/phrases
            transitions = ['however', 'therefore', 'furthermore', 'additionally', 'moreover', 'consequently', 'meanwhile', 'subsequently']
            if any(trans in summary.lower() for trans in transitions):
                score += 0.1
            
            # Avoid repetition
            words = summary.lower().split()
            unique_words = set(words)
            if len(words) > 0:
                repetition_ratio = len(unique_words) / len(words)
                if repetition_ratio > 0.7:
                    score += 0.2
                elif repetition_ratio > 0.5:
                    score += 0.1
            
            # Check for balanced sentence lengths
            if valid_sentences:
                avg_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
                if 8 <= avg_length <= 25:  # Reasonable sentence length
                    score += 0.1
            
            return min(1.0, score)
            
        except:
            return 0.5

    def _calculate_information_density(self, summary: str, key_points: List[str]) -> float:
        """Calculate information density and value of the summary."""
        
        try:
            score = 0.0
            
            # Check for specific information types
            summary_lower = summary.lower()
            
            # Numbers and data points
            if re.search(r'\d+%|\$\d+|\d+\.\d+|\d{4}', summary):
                score += 0.2
            
            # Specific names, places, or entities
            if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', summary):  # Proper nouns
                score += 0.1
            
            # Action words and verbs
            action_words = ['increase', 'decrease', 'improve', 'reduce', 'implement', 'develop', 'create', 'establish', 'achieve', 'result', 'cause', 'lead', 'show', 'demonstrate']
            if any(word in summary_lower for word in action_words):
                score += 0.2
            
            # Technical or domain-specific terms
            if len(re.findall(r'\b[a-z]{6,}\b', summary_lower)) >= 3:  # Longer words often more specific
                score += 0.1
            
            # Key points alignment
            if key_points:
                key_points_text = ' '.join(key_points).lower()
                summary_words = set(summary_lower.split())
                key_words = set(key_points_text.split())
                
                if key_words:
                    alignment = len(summary_words.intersection(key_words)) / len(key_words)
                    score += min(0.4, alignment)
            
            return min(1.0, score)
            
        except:
            return 0.5

    def _calculate_key_points_quality(self, key_points: List[str]) -> float:
        """Calculate quality of extracted key points."""
        
        try:
            if not key_points:
                return 0.0
            
            score = 0.0
            
            # Number of key points
            num_points = len(key_points)
            if 3 <= num_points <= 7:
                score += 0.3
            elif 2 <= num_points <= 8:
                score += 0.2
            elif num_points >= 1:
                score += 0.1
            
            # Quality of individual points
            valid_points = 0
            for point in key_points:
                point_score = 0
                
                # Length check
                if 10 <= len(point.split()) <= 30:
                    point_score += 0.3
                elif 5 <= len(point.split()) <= 40:
                    point_score += 0.2
                
                # Structure check
                if point.strip() and point.strip()[0].isupper():
                    point_score += 0.1
                
                # Content quality
                if any(word in point.lower() for word in ['important', 'key', 'significant', 'main', 'primary']):
                    point_score += 0.1
                
                if point_score >= 0.3:
                    valid_points += 1
            
            # Bonus for having mostly valid points
            if num_points > 0:
                validity_ratio = valid_points / num_points
                score += validity_ratio * 0.4
            
            # Diversity check (avoid repetition)
            if len(key_points) > 1:
                all_words = []
                for point in key_points:
                    all_words.extend(point.lower().split())
                
                if all_words:
                    unique_ratio = len(set(all_words)) / len(all_words)
                    if unique_ratio > 0.7:
                        score += 0.3
                    elif unique_ratio > 0.5:
                        score += 0.2
            
            return min(1.0, score)
            
        except:
            return 0.5

    def _calculate_factual_consistency(self, content: str, summary: str) -> float:
        """Calculate factual consistency between content and summary."""
        
        try:
            score = 0.5  # Base score
            
            # Extract numbers and dates from both
            content_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', content))
            summary_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', summary))
            
            content_years = set(re.findall(r'\b(19|20)\d{2}\b', content))
            summary_years = set(re.findall(r'\b(19|20)\d{2}\b', summary))
            
            # Check if summary numbers are consistent with content
            if summary_numbers:
                consistent_numbers = summary_numbers.intersection(content_numbers)
                number_consistency = len(consistent_numbers) / len(summary_numbers)
                score += number_consistency * 0.3
            
            # Check year consistency
            if summary_years:
                consistent_years = summary_years.intersection(content_years)
                year_consistency = len(consistent_years) / len(summary_years)
                score += year_consistency * 0.2
            
            # Check for contradictory statements (basic)
            contradictory_patterns = [
                (r'\bincrease\b', r'\bdecrease\b'),
                (r'\brise\b', r'\bfall\b'),
                (r'\bgrow\b', r'\bshrink\b'),
                (r'\bimprove\b', r'\bworsen\b')
            ]
            
            content_lower = content.lower()
            summary_lower = summary.lower()
            
            for pos_pattern, neg_pattern in contradictory_patterns:
                content_has_pos = bool(re.search(pos_pattern, content_lower))
                content_has_neg = bool(re.search(neg_pattern, content_lower))
                summary_has_pos = bool(re.search(pos_pattern, summary_lower))
                summary_has_neg = bool(re.search(neg_pattern, summary_lower))
                
                # Penalize if summary contradicts content
                if (content_has_pos and summary_has_neg) or (content_has_neg and summary_has_pos):
                    score -= 0.1
            
            return min(1.0, max(0.0, score))
            
        except:
            return 0.5

    def _generate_fallback_summary(
        self, 
        content: str, 
        title: str, 
        customization: SummaryCustomization
    ) -> EnhancedSummary:
        """Generate a basic fallback summary when all else fails."""
        
        # Simple extractive summary
        sentences = content.split('.')[:3]
        summary_text = '. '.join(sentence.strip() for sentence in sentences if sentence.strip()) + '.'
        
        # Basic key points
        key_points = [
            "Content analysis completed with basic processing",
            "Summary generated using fallback method",
            "Limited advanced features available"
        ]
        
        return EnhancedSummary(
            text=summary_text,
            key_points=key_points,
            highlights=[],
            keywords=[],
            metadata={'method': 'fallback', 'confidence': 0.3},
            confidence_score=0.3
        )