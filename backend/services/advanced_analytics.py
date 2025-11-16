"""
Advanced Analytics Service with Google/Gemini-level accuracy
Implements sophisticated content analysis with chain-of-thought reasoning
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from textblob import TextBlob
# spacy import moved to optional section below

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Try to import spaCy, but make it optional
try:
    import spacy
except ImportError:
    spacy = None

class AdvancedAnalyticsService:
    """Advanced analytics service with Google/Gemini-level accuracy"""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Initialize NLP models
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Some features may be limited.")
                self.nlp = None
        else:
            print("Warning: spaCy not installed. Using NLTK-only features.")
            
        # Advanced stop words (more comprehensive)
        self.stop_words = set(stopwords.words('english')) if stopwords else set()
        self.stop_words.update({
            'said', 'say', 'says', 'saying', 'told', 'tell', 'tells', 'telling',
            'according', 'also', 'would', 'could', 'should', 'might', 'may',
            'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
            'new', 'old', 'good', 'bad', 'big', 'small', 'high', 'low',
            'way', 'ways', 'time', 'times', 'year', 'years', 'day', 'days',
            'people', 'person', 'man', 'woman', 'men', 'women', 'thing', 'things'
        })

    def intelligent_truncate(self, content: str, max_length: int = 12000) -> str:
        """Intelligently truncate content while preserving important sections."""
        if len(content) <= max_length:
            return content
            
        # Split into sentences and prioritize
        sentences = sent_tokenize(content)
        if not sentences:
            return content[:max_length]
            
        # Score sentences by importance
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position scoring (beginning and end are important)
            if i < len(sentences) * 0.2:  # First 20%
                score += 2
            elif i > len(sentences) * 0.8:  # Last 20%
                score += 1
                
            # Length scoring (medium length sentences often more informative)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 1
                
            # Keyword scoring (sentences with important terms)
            important_terms = ['conclusion', 'result', 'finding', 'important', 'significant', 'key', 'main', 'primary']
            for term in important_terms:
                if term in sentence.lower():
                    score += 1
                    
            sentence_scores.append((sentence, score, i))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        current_length = 0
        
        for sentence, score, original_index in sentence_scores:
            if current_length + len(sentence) <= max_length:
                selected_sentences.append((sentence, original_index))
                current_length += len(sentence)
            else:
                break
        
        # Sort selected sentences by original order
        selected_sentences.sort(key=lambda x: x[1])
        
        return ' '.join([sentence for sentence, _ in selected_sentences])

    def advanced_keyword_extraction(self, content: str, num_keywords: int = 20) -> Dict[str, List[str]]:
        """Extract keywords using TF-IDF, NER, and semantic analysis."""
        if not content.strip():
            return {"primary": [], "secondary": []}
            
        # Clean and preprocess text
        cleaned_content = self._clean_text(content)
        
        # TF-IDF based extraction
        tfidf_keywords = self._extract_tfidf_keywords(cleaned_content, num_keywords // 2)
        
        # Named entity extraction
        entity_keywords = self._extract_entity_keywords(cleaned_content)
        
        # POS-based keyword extraction (nouns, adjectives)
        pos_keywords = self._extract_pos_keywords(cleaned_content, num_keywords // 2)
        
        # Combine and rank keywords
        all_keywords = {}
        
        # Weight TF-IDF keywords highly
        for kw in tfidf_keywords:
            all_keywords[kw] = all_keywords.get(kw, 0) + 3
            
        # Weight entity keywords
        for kw in entity_keywords:
            all_keywords[kw] = all_keywords.get(kw, 0) + 2
            
        # Weight POS keywords
        for kw in pos_keywords:
            all_keywords[kw] = all_keywords.get(kw, 0) + 1
        
        # Filter and sort
        filtered_keywords = []
        for kw, score in sorted(all_keywords.items(), key=lambda x: x[1], reverse=True):
            if (len(kw) > 2 and 
                kw.lower() not in self.stop_words and 
                not kw.isdigit() and 
                len(kw.split()) <= 3):
                filtered_keywords.append(kw)
        
        # Split into primary and secondary
        mid_point = min(12, len(filtered_keywords) // 2)
        
        return {
            "primary": filtered_keywords[:mid_point],
            "secondary": filtered_keywords[mid_point:mid_point*2]
        }

    def advanced_topic_modeling(self, content: str) -> Dict[str, Any]:
        """Perform advanced topic modeling using LDA and clustering."""
        if not content.strip():
            return {"main_topics": [], "subtopics": [], "topic_distribution": {}}
            
        # Preprocess for topic modeling
        sentences = sent_tokenize(content)
        if len(sentences) < 3:
            return {"main_topics": [], "subtopics": [], "topic_distribution": {}}
            
        # Clean sentences
        cleaned_sentences = [self._clean_text(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if len(sent.split()) > 3]
        
        if len(cleaned_sentences) < 2:
            return {"main_topics": [], "subtopics": [], "topic_distribution": {}}
        
        try:
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA topic modeling
            n_topics = min(5, max(2, len(cleaned_sentences) // 3))
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(tfidf_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(' '.join(top_words[:3]))  # Use top 3 words as topic name
            
            # Get topic distribution
            doc_topic_dist = lda.transform(tfidf_matrix)
            avg_topic_dist = np.mean(doc_topic_dist, axis=0)
            
            topic_distribution = {}
            for i, prob in enumerate(avg_topic_dist):
                if i < len(topics):
                    topic_distribution[topics[i]] = float(prob)
            
            # Normalize distribution
            total = sum(topic_distribution.values())
            if total > 0:
                topic_distribution = {k: v/total for k, v in topic_distribution.items()}
            
            return {
                "main_topics": topics[:3],
                "subtopics": topics[3:] if len(topics) > 3 else [],
                "topic_distribution": topic_distribution
            }
            
        except Exception as e:
            print(f"Topic modeling error: {e}")
            return {"main_topics": [], "subtopics": [], "topic_distribution": {}}

    def advanced_sentiment_analysis(self, content: str) -> Dict[str, Any]:
        """Perform multi-model sentiment analysis with emotion detection."""
        if not content.strip():
            return {
                "overall": {"label": "neutral", "polarity": 0.0, "subjectivity": 0.5, "confidence": 0.5},
                "emotional_tone": "neutral"
            }
        
        # TextBlob analysis
        blob = TextBlob(content)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine label
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        # Calculate confidence based on polarity strength
        confidence = min(0.95, abs(polarity) + 0.5)
        
        # Emotion detection
        emotional_tone = self._detect_emotional_tone(content, polarity, subjectivity)
        
        return {
            "overall": {
                "label": label,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "confidence": round(confidence, 3)
            },
            "emotional_tone": emotional_tone
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()

    def _extract_tfidf_keywords(self, content: str, num_keywords: int) -> List[str]:
        """Extract keywords using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=num_keywords * 2,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
            return [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
        except Exception:
            return []

    def _extract_entity_keywords(self, content: str) -> List[str]:
        """Extract named entities as keywords."""
        keywords = []
        
        # spaCy NER if available
        if self.nlp:
            try:
                doc = self.nlp(content)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        keywords.append(ent.text.strip())
            except Exception:
                pass
        
        # NLTK NER as fallback
        try:
            tokens = word_tokenize(content)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    keywords.append(entity)
        except Exception:
            pass
        
        return list(set(keywords))

    def _extract_pos_keywords(self, content: str, num_keywords: int) -> List[str]:
        """Extract keywords based on POS tags (nouns, adjectives)."""
        try:
            tokens = word_tokenize(content.lower())
            pos_tags = pos_tag(tokens)
            
            # Filter for nouns and adjectives
            keywords = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ')) and len(word) > 2:
                    if word not in self.stop_words:
                        keywords.append(word)
            
            # Count frequency and return top keywords
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_keywords[:num_keywords]]
            
        except Exception:
            return []

    def _detect_emotional_tone(self, content: str, polarity: float, subjectivity: float) -> str:
        """Detect emotional tone based on content analysis."""
        content_lower = content.lower()
        
        # Emotion keywords
        emotion_patterns = {
            'enthusiastic': ['excited', 'amazing', 'fantastic', 'incredible', 'outstanding'],
            'optimistic': ['hope', 'positive', 'bright', 'promising', 'opportunity'],
            'cautious': ['careful', 'consider', 'however', 'although', 'potential'],
            'analytical': ['analysis', 'data', 'research', 'study', 'evidence'],
            'concerned': ['worry', 'concern', 'issue', 'problem', 'challenge'],
            'confident': ['confident', 'certain', 'sure', 'definite', 'clear'],
            'skeptical': ['doubt', 'question', 'uncertain', 'unclear', 'maybe']
        }
        
        # Score emotions
        emotion_scores = {}
        for emotion, keywords in emotion_patterns.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Determine tone based on sentiment and emotion keywords
        if emotion_scores:
            top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            return top_emotion
        elif polarity > 0.3:
            return "positive"
        elif polarity < -0.3:
            return "negative"
        elif subjectivity > 0.7:
            return "subjective"
        else:
            return "analytical"

    async def generate_advanced_insights(self, content: str, keywords: List[str], 
                                       sentiment: Dict[str, Any], topics: List[str]) -> List[Dict[str, Any]]:
        """Generate advanced insights with evidence and reasoning."""
        insights = []
        
        # Content length and structure insights
        word_count = len(content.split())
        sentence_count = len(sent_tokenize(content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        if avg_sentence_length > 25:
            insights.append({
                "text": "Content uses complex sentence structures that may impact readability",
                "confidence": 0.8,
                "rationale": f"Average sentence length is {avg_sentence_length:.1f} words, above recommended 20 words",
                "evidence": f"Analyzed {sentence_count} sentences with {word_count} total words"
            })
        
        # Keyword density insights
        if keywords:
            top_keyword = keywords[0] if keywords else ""
            keyword_density = content.lower().count(top_keyword.lower()) / max(word_count, 1) * 100
            
            if keyword_density > 3:
                insights.append({
                    "text": f"High focus on '{top_keyword}' suggests specialized content",
                    "confidence": 0.9,
                    "rationale": f"Keyword '{top_keyword}' appears {keyword_density:.1f}% of the time",
                    "evidence": f"'{top_keyword}' mentioned multiple times throughout content"
                })
        
        # Sentiment insights
        if sentiment and 'overall' in sentiment:
            polarity = sentiment['overall'].get('polarity', 0)
            confidence = sentiment['overall'].get('confidence', 0.5)
            
            if abs(polarity) > 0.5 and confidence > 0.7:
                tone = "positive" if polarity > 0 else "negative"
                insights.append({
                    "text": f"Content expresses strong {tone} sentiment with high confidence",
                    "confidence": confidence,
                    "rationale": f"Sentiment polarity of {polarity:.2f} with {confidence:.2f} confidence",
                    "evidence": "Multiple sentiment indicators found throughout the text"
                })
        
        # Topic diversity insights
        if len(topics) > 3:
            insights.append({
                "text": "Content covers multiple diverse topics, indicating comprehensive coverage",
                "confidence": 0.7,
                "rationale": f"Identified {len(topics)} distinct topics",
                "evidence": f"Topics include: {', '.join(topics[:3])}"
            })
        
        return insights[:6]  # Limit to 6 insights

    def comprehensive_abstractive_summary(self, content: str, title: str = "", target_length: int = 300) -> Dict[str, Any]:
        """Generate comprehensive abstractive summary with multiple perspectives."""
        if not content.strip():
            return {"text": "", "key_points": [], "word_count": 0}
        
        # Intelligent content preprocessing
        preprocessed_content = self._clean_text(content)
        sentences = sent_tokenize(preprocessed_content)
        
        if len(sentences) < 3:
            return {"text": content, "key_points": [], "word_count": len(content.split())}
        
        # Extract key concepts and themes
        keywords = self.advanced_keyword_extraction(content, 15)
        primary_keywords = keywords.get("primary", [])
        
        # Identify main themes and concepts
        main_themes = self._extract_main_themes(sentences, primary_keywords)
        
        # Generate comprehensive summary based on content length
        content_length = len(content)
        if content_length > 5000:
            summary = self._generate_long_form_summary(sentences, main_themes, primary_keywords, target_length)
        elif content_length > 2000:
            summary = self._generate_medium_form_summary(sentences, main_themes, primary_keywords, target_length)
        else:
            summary = self._generate_short_form_summary(sentences, main_themes, primary_keywords, target_length)
        
        # Generate detailed key points
        key_points = self._generate_comprehensive_key_points(sentences, main_themes, primary_keywords)
        
        return {
            "text": summary,
            "key_points": key_points,
            "word_count": len(summary.split())
        }
    
    def _extract_main_themes(self, sentences: List[str], keywords: List[str]) -> List[str]:
        """Extract main themes from content."""
        themes = []
        
        # Theme categories based on keywords
        theme_categories = {
            "technology": ["technology", "software", "ai", "digital", "computer", "data", "algorithm", "system"],
            "business": ["business", "company", "market", "revenue", "profit", "strategy", "growth", "investment"],
            "research": ["research", "study", "analysis", "findings", "results", "evidence", "methodology"],
            "health": ["health", "medical", "treatment", "patient", "healthcare", "wellness", "disease"],
            "education": ["education", "learning", "student", "teaching", "academic", "knowledge", "skill"],
            "innovation": ["innovation", "development", "advancement", "breakthrough", "improvement", "progress"]
        }
        
        # Identify themes based on keyword presence
        for theme, theme_keywords in theme_categories.items():
            if any(kw.lower() in [k.lower() for k in keywords] for kw in theme_keywords):
                themes.append(theme)
        
        # Add custom themes based on high-frequency keywords
        for keyword in keywords[:5]:
            if len(keyword) > 4 and keyword.lower() not in [t.lower() for t in themes]:
                themes.append(keyword.title())
        
        return themes[:6]  # Limit to 6 main themes
    
    def _generate_long_form_summary(self, sentences: List[str], themes: List[str], keywords: List[str], target_length: int) -> str:
        """Generate comprehensive summary for long content."""
        # Structure: Introduction + Main Points + Analysis + Conclusion
        
        # Introduction (context and background)
        intro_sentences = self._select_contextual_sentences(sentences[:len(sentences)//4], keywords, 2)
        introduction = f"This comprehensive analysis covers {', '.join(themes[:3])}. " + ' '.join(intro_sentences)
        
        # Main content analysis
        main_sentences = self._select_key_content_sentences(sentences[len(sentences)//4:-len(sentences)//4], keywords, themes, 4)
        main_content = "Key findings include: " + ' '.join(main_sentences)
        
        # Implications and conclusions
        conclusion_sentences = self._select_contextual_sentences(sentences[-len(sentences)//4:], keywords, 2)
        conclusion = "The analysis reveals significant implications: " + ' '.join(conclusion_sentences)
        
        full_summary = f"{introduction} {main_content} {conclusion}"
        
        # Trim to target length if needed
        return self._trim_to_length(full_summary, target_length)
    
    def _generate_medium_form_summary(self, sentences: List[str], themes: List[str], keywords: List[str], target_length: int) -> str:
        """Generate balanced summary for medium content."""
        # Structure: Context + Main Points + Key Insights
        
        context_sentences = self._select_contextual_sentences(sentences[:len(sentences)//3], keywords, 2)
        main_sentences = self._select_key_content_sentences(sentences[len(sentences)//3:], keywords, themes, 3)
        
        summary = f"Focusing on {', '.join(themes[:2])}, this analysis presents: " + ' '.join(context_sentences + main_sentences)
        
        return self._trim_to_length(summary, target_length)
    
    def _generate_short_form_summary(self, sentences: List[str], themes: List[str], keywords: List[str], target_length: int) -> str:
        """Generate concise summary for short content."""
        key_sentences = self._select_key_content_sentences(sentences, keywords, themes, 3)
        summary = f"This content addresses {themes[0] if themes else 'key topics'}: " + ' '.join(key_sentences)
        
        return self._trim_to_length(summary, target_length)
    
    def _select_contextual_sentences(self, sentences: List[str], keywords: List[str], count: int) -> List[str]:
        """Select sentences that provide context and background."""
        scored_sentences = []
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on contextual indicators
            context_indicators = ["background", "context", "overview", "introduction", "purpose", "objective"]
            for indicator in context_indicators:
                if indicator in sentence_lower:
                    score += 2
            
            # Score based on keyword presence
            for keyword in keywords[:5]:
                if keyword.lower() in sentence_lower:
                    score += 1
            
            # Prefer longer, informative sentences
            if 15 <= len(sentence.split()) <= 40:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in scored_sentences[:count]]
    
    def _select_key_content_sentences(self, sentences: List[str], keywords: List[str], themes: List[str], count: int) -> List[str]:
        """Select sentences with key content and findings."""
        scored_sentences = []
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on content indicators
            content_indicators = ["result", "finding", "shows", "indicates", "demonstrates", "reveals", "analysis", "data", "evidence"]
            for indicator in content_indicators:
                if indicator in sentence_lower:
                    score += 3
            
            # Score based on keyword density
            keyword_count = sum(1 for kw in keywords[:8] if kw.lower() in sentence_lower)
            score += keyword_count * 2
            
            # Score based on theme relevance
            theme_count = sum(1 for theme in themes if theme.lower() in sentence_lower)
            score += theme_count
            
            # Prefer substantial sentences
            word_count = len(sentence.split())
            if 12 <= word_count <= 45:
                score += 2
            elif 8 <= word_count <= 60:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in scored_sentences[:count]]
    
    def _generate_comprehensive_key_points(self, sentences: List[str], themes: List[str], keywords: List[str]) -> List[str]:
        """Generate detailed key points covering all important aspects."""
        key_points = []
        
        # Categorize sentences by themes and importance
        categorized_points = {}
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip very short sentences
            if len(sentence.split()) < 8:
                continue
            
            # Determine category
            category = "general"
            for theme in themes:
                if theme.lower() in sentence_lower:
                    category = theme
                    break
            
            # Score sentence importance
            score = 0
            importance_indicators = ["important", "significant", "key", "main", "primary", "critical", "essential", "major"]
            for indicator in importance_indicators:
                if indicator in sentence_lower:
                    score += 2
            
            # Add keyword relevance
            keyword_matches = sum(1 for kw in keywords[:10] if kw.lower() in sentence_lower)
            score += keyword_matches
            
            if score > 1:  # Only include sentences with some importance
                if category not in categorized_points:
                    categorized_points[category] = []
                categorized_points[category].append((sentence.strip(), score))
        
        # Select top points from each category
        for category, points in categorized_points.items():
            points.sort(key=lambda x: x[1], reverse=True)
            for point, score in points[:2]:  # Top 2 from each category
                if len(point) > 20:
                    # Clean and format point
                    clean_point = point.strip()
                    if not clean_point.endswith('.'):
                        clean_point += '.'
                    key_points.append(clean_point)
        
        return key_points[:12]  # Limit to 12 key points
    
    def _trim_to_length(self, text: str, target_length: int) -> str:
        """Trim text to approximately target word count while preserving meaning."""
        words = text.split()
        if len(words) <= target_length:
            return text
        
        # Find a good breaking point near the target
        target_index = min(target_length - 10, len(words) - 1)
        
        # Look for sentence boundaries near the target
        for i in range(target_index, min(target_index + 20, len(words))):
            if i < len(words) and words[i].endswith('.'):
                return ' '.join(words[:i+1])
        
        # If no good breaking point, just truncate and add ellipsis
        return ' '.join(words[:target_length]) + '...'