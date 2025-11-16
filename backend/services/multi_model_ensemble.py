"""
Multi-Model Ensemble Service for Enhanced AI Summarization
Combines multiple AI models for consensus-based summaries
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    model_name: str
    summary: str
    confidence: float
    processing_time: float
    tokens_used: int
    metadata: Dict[str, Any]

@dataclass
class ConsensusSummary:
    final_summary: str
    model_results: List[ModelResult]
    consensus_score: float
    agreement_level: str
    processing_time: float
    metadata: Dict[str, Any]

class MultiModelEnsemble:
    """Ensemble of multiple AI models for robust summarization"""
    
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.openai_client = None
        self.anthropic_client = None
        self.local_model = None
        self.summarization_pipeline = None
        
        # Initialize OpenAI
        if OpenAI and openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        # Initialize Anthropic
        if anthropic and anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.warning(f"Anthropic initialization failed: {e}")
        
        # Initialize local models (disabled for quick startup)
        # self._initialize_local_models()
        logger.info("Local models disabled for quick startup")
    
    def _initialize_local_models(self):
        """Initialize local models for fallback"""
        try:
            if SentenceTransformer:
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer initialized successfully")
        except Exception as e:
            logger.warning(f"SentenceTransformer initialization failed: {e}")
        
        try:
            if pipeline:
                # Use a smaller model to avoid download issues
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1,  # CPU
                    tokenizer_kwargs={"trust_remote_code": True},
                    model_kwargs={"trust_remote_code": True}
                )
                logger.info("Summarization pipeline initialized successfully")
        except Exception as e:
            logger.warning(f"Summarization pipeline initialization failed: {e}")
            logger.info("Continuing without local summarization pipeline - will use API models only")
    
    async def generate_consensus_summary(
        self,
        content: str,
        title: str = "",
        url: str = "",
        max_length: int = 500,
        domain: str = "general"
    ) -> ConsensusSummary:
        """Generate summary using multiple models and consensus"""
        
        start_time = datetime.now()
        
        # Prepare content for processing
        processed_content = self._preprocess_content(content, title, url)
        
        # Run models in parallel
        tasks = []
        
        if self.openai_client:
            tasks.append(self._gpt4o_summary(processed_content, title, max_length, domain))
        
        if self.anthropic_client:
            tasks.append(self._claude_summary(processed_content, title, max_length, domain))
        
        if self.local_model or self.summarization_pipeline:
            tasks.append(self._local_summary(processed_content, title, max_length, domain))
        
        # Wait for all models to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and get valid results
        valid_results = []
        for result in results:
            if isinstance(result, ModelResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Model failed: {result}")
        
        if not valid_results:
            raise Exception("All models failed to generate summaries")
        
        # Generate consensus
        consensus = self._generate_consensus(valid_results, processed_content)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ConsensusSummary(
            final_summary=consensus['summary'],
            model_results=valid_results,
            consensus_score=consensus['score'],
            agreement_level=consensus['agreement'],
            processing_time=processing_time,
            metadata={
                'domain': domain,
                'content_length': len(processed_content),
                'models_used': len(valid_results),
                'timestamp': datetime.now().isoformat()
            }
        )
    
    async def _gpt4o_summary(
        self,
        content: str,
        title: str,
        max_length: int,
        domain: str
    ) -> ModelResult:
        """Generate summary using GPT-4o"""
        
        start_time = datetime.now()
        
        prompt = self._build_domain_prompt(content, title, domain, max_length)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an expert {domain} content analyst with advanced NLP capabilities. "
                            "Generate highly accurate, comprehensive summaries that capture the essence, "
                            "nuances, and actionable insights of the content. Focus on precision, "
                            "completeness, and relevance to user requirements."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_length * 2,  # Allow for longer summaries
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            summary = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(summary, content)
            
            return ModelResult(
                model_name="gpt-4o",
                summary=summary,
                confidence=confidence,
                processing_time=processing_time,
                tokens_used=tokens_used,
                metadata={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'domain': domain
                }
            )
            
        except Exception as e:
            logger.error(f"GPT-4o summary generation failed: {e}")
            raise e
    
    async def _claude_summary(
        self,
        content: str,
        title: str,
        max_length: int,
        domain: str
    ) -> ModelResult:
        """Generate summary using Claude"""
        
        start_time = datetime.now()
        
        prompt = self._build_domain_prompt(content, title, domain, max_length)
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_length * 2,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": f"{self._get_claude_system_prompt(domain)}\n\n{prompt}"
                    }
                ]
            )
            
            summary = response.content[0].text.strip()
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            processing_time = (datetime.now() - start_time).total_seconds()
            
            confidence = self._calculate_confidence(summary, content)
            
            return ModelResult(
                model_name="claude-3-sonnet",
                summary=summary,
                confidence=confidence,
                processing_time=processing_time,
                tokens_used=tokens_used,
                metadata={
                    'temperature': 0.1,
                    'domain': domain
                }
            )
            
        except Exception as e:
            logger.error(f"Claude summary generation failed: {e}")
            raise e
    
    async def _local_summary(
        self,
        content: str,
        title: str,
        max_length: int,
        domain: str
    ) -> ModelResult:
        """Generate summary using local models"""
        
        start_time = datetime.now()
        
        try:
            if self.summarization_pipeline:
                # Use BART for summarization
                summary = self.summarization_pipeline(
                    content,
                    max_length=max_length,
                    min_length=max_length // 3,
                    do_sample=False
                )[0]['summary_text']
                
                processing_time = (datetime.now() - start_time).total_seconds()
                confidence = self._calculate_confidence(summary, content)
                
                return ModelResult(
                    model_name="bart-large-cnn",
                    summary=summary,
                    confidence=confidence,
                    processing_time=processing_time,
                    tokens_used=0,
                    metadata={
                        'model_type': 'local',
                        'domain': domain
                    }
                )
            
            elif self.local_model:
                # Use sentence transformer for extractive summarization
                sentences = self._split_into_sentences(content)
                if len(sentences) <= 3:
                    summary = content[:max_length]
                else:
                    # Get sentence embeddings
                    embeddings = self.local_model.encode(sentences)
                    
                    # Calculate sentence importance
                    importance_scores = self._calculate_sentence_importance(embeddings)
                    
                    # Select top sentences
                    top_indices = np.argsort(importance_scores)[-max_length//50:]  # Rough estimate
                    top_indices = sorted(top_indices)
                    
                    summary = " ".join([sentences[i] for i in top_indices])
                
                processing_time = (datetime.now() - start_time).total_seconds()
                confidence = self._calculate_confidence(summary, content)
                
                return ModelResult(
                    model_name="sentence-transformer",
                    summary=summary,
                    confidence=confidence,
                    processing_time=processing_time,
                    tokens_used=0,
                    metadata={
                        'model_type': 'local',
                        'domain': domain
                    }
                )
            
            else:
                raise Exception("No local models available")
                
        except Exception as e:
            logger.error(f"Local summary generation failed: {e}")
            raise e
    
    def _generate_consensus(self, results: List[ModelResult], content: str) -> Dict[str, Any]:
        """Generate consensus from multiple model results"""
        
        if len(results) == 1:
            return {
                'summary': results[0].summary,
                'score': results[0].confidence,
                'agreement': 'single_model'
            }
        
        # Calculate similarity between summaries
        summaries = [result.summary for result in results]
        similarities = self._calculate_summary_similarities(summaries)
        
        # Weight summaries by confidence
        weights = [result.confidence for result in results]
        weighted_summaries = [(summary, weight) for summary, weight in zip(summaries, weights)]
        
        # Generate consensus summary
        if similarities.mean() > 0.7:  # High agreement
            # Use weighted average of summaries
            consensus_summary = self._weighted_summary_merge(weighted_summaries)
            agreement_level = "high"
        elif similarities.mean() > 0.5:  # Medium agreement
            # Use most confident summary as base, enhance with others
            best_result = max(results, key=lambda x: x.confidence)
            consensus_summary = self._enhance_summary_with_others(best_result.summary, summaries)
            agreement_level = "medium"
        else:  # Low agreement
            # Use most confident summary
            best_result = max(results, key=lambda x: x.confidence)
            consensus_summary = best_result.summary
            agreement_level = "low"
        
        consensus_score = np.mean([result.confidence for result in results])
        
        return {
            'summary': consensus_summary,
            'score': consensus_score,
            'agreement': agreement_level
        }
    
    def _calculate_summary_similarities(self, summaries: List[str]) -> np.ndarray:
        """Calculate pairwise similarities between summaries"""
        
        if len(summaries) < 2:
            return np.array([1.0])
        
        if self.local_model:
            embeddings = self.local_model.encode(summaries)
            similarities = cosine_similarity(embeddings)
            return similarities
        else:
            # Fallback to simple text similarity
            similarities = []
            for i in range(len(summaries)):
                for j in range(i + 1, len(summaries)):
                    sim = self._simple_text_similarity(summaries[i], summaries[j])
                    similarities.append(sim)
            return np.array(similarities)
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _weighted_summary_merge(self, weighted_summaries: List[Tuple[str, float]]) -> str:
        """Merge summaries using weighted approach"""
        
        # For now, return the highest weighted summary
        # In a more sophisticated implementation, you could merge sentences
        best_summary, best_weight = max(weighted_summaries, key=lambda x: x[1])
        return best_summary
    
    def _enhance_summary_with_others(self, base_summary: str, other_summaries: List[str]) -> str:
        """Enhance base summary with information from other summaries"""
        
        # Simple implementation - return base summary
        # Could be enhanced to extract additional insights from other summaries
        return base_summary
    
    def _calculate_confidence(self, summary: str, content: str) -> float:
        """Calculate confidence score for a summary"""
        
        if not summary or not content:
            return 0.0
        
        # Factors for confidence calculation
        length_ratio = len(summary) / len(content)
        word_count = len(summary.split())
        
        # Ideal length ratio (adjust based on requirements)
        ideal_ratio = 0.1  # 10% of original content
        
        # Calculate confidence based on length appropriateness
        length_score = 1.0 - abs(length_ratio - ideal_ratio) / ideal_ratio
        length_score = max(0.0, min(1.0, length_score))
        
        # Word count score (prefer summaries with reasonable word count)
        word_score = 1.0 if 10 <= word_count <= 200 else 0.8
        
        # Combine scores
        confidence = (length_score * 0.6 + word_score * 0.4)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_sentence_importance(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate importance scores for sentences"""
        
        # Use centroid-based importance
        centroid = np.mean(embeddings, axis=0)
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1))
        
        return similarities.flatten()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _preprocess_content(self, content: str, title: str, url: str) -> str:
        """Preprocess content for better summarization"""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Add title context if available
        if title:
            content = f"Title: {title}\n\n{content}"
        
        # Add URL context if available
        if url:
            content = f"Source: {url}\n\n{content}"
        
        return content
    
    def _build_domain_prompt(self, content: str, title: str, domain: str, max_length: int) -> str:
        """Build domain-specific prompt"""
        
        domain_instructions = {
            'tech': "Focus on technical concepts, code examples, APIs, frameworks, and implementation details.",
            'ai': "Emphasize algorithms, models, research findings, methodologies, and technical innovations.",
            'webdev': "Highlight frameworks, libraries, best practices, code snippets, and development workflows.",
            'finance': "Focus on financial metrics, market analysis, investment strategies, and economic insights.",
            'medical': "Emphasize medical procedures, research findings, clinical data, and healthcare insights.",
            'legal': "Highlight legal precedents, regulations, case law, and compliance requirements.",
            'general': "Provide a comprehensive overview covering all key points and insights."
        }
        
        instruction = domain_instructions.get(domain, domain_instructions['general'])
        
        prompt = f"""
        Please analyze and summarize the following content with a focus on {domain} domain.
        
        {instruction}
        
        Content to summarize:
        {content[:8000]}  # Limit content length
        
        Requirements:
        - Maximum {max_length} words
        - Maintain accuracy and factual correctness
        - Include key technical details and insights
        - Use clear, professional language
        - Structure the summary logically
        
        Please provide a comprehensive summary that captures the essential information and insights.
        """
        
        return prompt
    
    def _get_claude_system_prompt(self, domain: str) -> str:
        """Get system prompt for Claude"""
        
        return f"""
        You are an expert {domain} content analyst with advanced NLP capabilities. 
        Your expertise includes deep semantic understanding, factual accuracy verification, and 
        contextual analysis. Generate highly accurate, comprehensive summaries that capture 
        the essence, nuances, and actionable insights of the content. Focus on precision, 
        completeness, and relevance to user requirements.
        """
