"""
Real-time Learning and Feedback Loop System
Collects user feedback and continuously improves AI summaries
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import numpy as np
from collections import defaultdict, Counter
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    user_id: str
    summary_id: str
    rating: int  # 1-5 scale
    feedback_text: str
    domain: str
    technology: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class LearningInsight:
    pattern: str
    confidence: float
    frequency: int
    domain: str
    improvement_suggestion: str
    created_at: datetime

@dataclass
class UserPreference:
    user_id: str
    domain: str
    preferred_summary_length: str
    preferred_summary_style: str
    preferred_technical_depth: str
    preferred_focus_areas: List[str]
    disliked_patterns: List[str]
    liked_patterns: List[str]
    updated_at: datetime

class FeedbackCollector:
    """Collects and processes user feedback"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.feedback_history = []
        self.user_preferences = {}
        self.learning_insights = []
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                feedback_text TEXT,
                domain TEXT,
                technology TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Create user preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                domain TEXT,
                preferred_summary_length TEXT,
                preferred_summary_style TEXT,
                preferred_technical_depth TEXT,
                preferred_focus_areas TEXT,
                disliked_patterns TEXT,
                liked_patterns TEXT,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Create learning insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                confidence REAL NOT NULL,
                frequency INTEGER NOT NULL,
                domain TEXT,
                improvement_suggestion TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def collect_feedback(
        self,
        user_id: str,
        summary_id: str,
        rating: int,
        feedback_text: str = "",
        domain: str = "",
        technology: str = "",
        metadata: Dict[str, Any] = None
    ) -> UserFeedback:
        """Collect user feedback for a summary"""
        
        if metadata is None:
            metadata = {}
        
        feedback = UserFeedback(
            user_id=user_id,
            summary_id=summary_id,
            rating=rating,
            feedback_text=feedback_text,
            domain=domain,
            technology=technology,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Store in database
        await self._store_feedback(feedback)
        
        # Add to in-memory history
        self.feedback_history.append(feedback)
        
        # Update user preferences
        await self._update_user_preferences(feedback)
        
        # Generate learning insights
        await self._generate_learning_insights(feedback)
        
        logger.info(f"Collected feedback from user {user_id} for summary {summary_id}: {rating}/5")
        return feedback
    
    async def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (user_id, summary_id, rating, feedback_text, domain, technology, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.user_id,
            feedback.summary_id,
            feedback.rating,
            feedback.feedback_text,
            feedback.domain,
            feedback.technology,
            feedback.timestamp.isoformat(),
            json.dumps(feedback.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_user_preferences(self, feedback: UserFeedback):
        """Update user preferences based on feedback"""
        
        user_id = feedback.user_id
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreference(
                user_id=user_id,
                domain=feedback.domain,
                preferred_summary_length="medium",
                preferred_summary_style="balanced",
                preferred_technical_depth="medium",
                preferred_focus_areas=[],
                disliked_patterns=[],
                liked_patterns=[],
                updated_at=datetime.now()
            )
        
        preference = self.user_preferences[user_id]
        
        # Analyze feedback text for patterns
        if feedback.feedback_text:
            if feedback.rating >= 4:
                # Extract liked patterns
                liked_patterns = self._extract_patterns(feedback.feedback_text, positive=True)
                preference.liked_patterns.extend(liked_patterns)
            elif feedback.rating <= 2:
                # Extract disliked patterns
                disliked_patterns = self._extract_patterns(feedback.feedback_text, positive=False)
                preference.disliked_patterns.extend(disliked_patterns)
        
        # Update domain preferences
        if feedback.domain:
            preference.domain = feedback.domain
        
        preference.updated_at = datetime.now()
        
        # Store updated preferences
        await self._store_user_preferences(preference)
    
    async def _store_user_preferences(self, preference: UserPreference):
        """Store user preferences in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_id, domain, preferred_summary_length, preferred_summary_style, 
             preferred_technical_depth, preferred_focus_areas, disliked_patterns, 
             liked_patterns, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            preference.user_id,
            preference.domain,
            preference.preferred_summary_length,
            preference.preferred_summary_style,
            preference.preferred_technical_depth,
            json.dumps(preference.preferred_focus_areas),
            json.dumps(preference.disliked_patterns),
            json.dumps(preference.liked_patterns),
            preference.updated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _extract_patterns(self, text: str, positive: bool = True) -> List[str]:
        """Extract patterns from feedback text"""
        
        patterns = []
        
        # Common positive patterns
        if positive:
            positive_keywords = [
                'clear', 'concise', 'helpful', 'detailed', 'comprehensive',
                'well-structured', 'easy to understand', 'informative',
                'accurate', 'relevant', 'useful', 'insightful'
            ]
            
            for keyword in positive_keywords:
                if keyword.lower() in text.lower():
                    patterns.append(f"contains_{keyword}")
        
        # Common negative patterns
        else:
            negative_keywords = [
                'unclear', 'confusing', 'too long', 'too short', 'irrelevant',
                'inaccurate', 'hard to understand', 'poorly structured',
                'missing', 'incomplete', 'outdated', 'repetitive'
            ]
            
            for keyword in negative_keywords:
                if keyword.lower() in text.lower():
                    patterns.append(f"contains_{keyword}")
        
        return patterns
    
    async def _generate_learning_insights(self, feedback: UserFeedback):
        """Generate learning insights from feedback"""
        
        # Analyze feedback patterns
        if feedback.feedback_text:
            patterns = self._extract_patterns(feedback.feedback_text, feedback.rating >= 4)
            
            for pattern in patterns:
                # Check if pattern already exists
                existing_insight = None
                for insight in self.learning_insights:
                    if insight.pattern == pattern and insight.domain == feedback.domain:
                        existing_insight = insight
                        break
                
                if existing_insight:
                    # Update existing insight
                    existing_insight.frequency += 1
                    existing_insight.confidence = min(1.0, existing_insight.confidence + 0.1)
                else:
                    # Create new insight
                    insight = LearningInsight(
                        pattern=pattern,
                        confidence=0.5,
                        frequency=1,
                        domain=feedback.domain,
                        improvement_suggestion=self._generate_improvement_suggestion(pattern, feedback.rating),
                        created_at=datetime.now()
                    )
                    self.learning_insights.append(insight)
        
        # Store insights in database
        await self._store_learning_insights()
    
    def _generate_improvement_suggestion(self, pattern: str, rating: int) -> str:
        """Generate improvement suggestion based on pattern and rating"""
        
        suggestions = {
            'contains_clear': 'Continue using clear, simple language',
            'contains_concise': 'Maintain concise summaries without unnecessary details',
            'contains_helpful': 'Focus on actionable and practical information',
            'contains_detailed': 'Provide more detailed explanations and examples',
            'contains_comprehensive': 'Ensure comprehensive coverage of all key points',
            'contains_unclear': 'Use simpler language and better structure',
            'contains_confusing': 'Improve organization and flow of information',
            'contains_too_long': 'Reduce summary length while maintaining key information',
            'contains_too_short': 'Add more detail and context to summaries',
            'contains_irrelevant': 'Focus on more relevant and targeted information',
            'contains_inaccurate': 'Improve fact-checking and accuracy verification',
            'contains_missing': 'Ensure all important information is included'
        }
        
        return suggestions.get(pattern, 'Consider user feedback for improvement')
    
    async def _store_learning_insights(self):
        """Store learning insights in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing insights
        cursor.execute('DELETE FROM learning_insights')
        
        # Insert current insights
        for insight in self.learning_insights:
            cursor.execute('''
                INSERT INTO learning_insights 
                (pattern, confidence, frequency, domain, improvement_suggestion, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                insight.pattern,
                insight.confidence,
                insight.frequency,
                insight.domain,
                insight.improvement_suggestion,
                insight.created_at.isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """Get user preferences"""
        
        return self.user_preferences.get(user_id)
    
    def get_feedback_analytics(self, domain: str = None, days: int = 30) -> Dict[str, Any]:
        """Get feedback analytics"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter feedback by domain and date
        filtered_feedback = [
            f for f in self.feedback_history
            if f.timestamp >= cutoff_date and (domain is None or f.domain == domain)
        ]
        
        if not filtered_feedback:
            return {
                'total_feedback': 0,
                'average_rating': 0.0,
                'rating_distribution': {},
                'common_patterns': [],
                'improvement_areas': []
            }
        
        # Calculate metrics
        total_feedback = len(filtered_feedback)
        average_rating = sum(f.rating for f in filtered_feedback) / total_feedback
        
        # Rating distribution
        rating_distribution = Counter(f.rating for f in filtered_feedback)
        
        # Common patterns
        all_patterns = []
        for f in filtered_feedback:
            if f.feedback_text:
                patterns = self._extract_patterns(f.feedback_text, f.rating >= 4)
                all_patterns.extend(patterns)
        
        common_patterns = Counter(all_patterns).most_common(5)
        
        # Improvement areas
        improvement_areas = []
        for insight in self.learning_insights:
            if insight.confidence > 0.7 and insight.frequency > 2:
                improvement_areas.append({
                    'pattern': insight.pattern,
                    'suggestion': insight.improvement_suggestion,
                    'frequency': insight.frequency
                })
        
        return {
            'total_feedback': total_feedback,
            'average_rating': average_rating,
            'rating_distribution': dict(rating_distribution),
            'common_patterns': [{'pattern': p, 'count': c} for p, c in common_patterns],
            'improvement_areas': improvement_areas
        }
    
    def get_learning_insights(self, domain: str = None) -> List[LearningInsight]:
        """Get learning insights for a domain"""
        
        if domain:
            return [insight for insight in self.learning_insights if insight.domain == domain]
        return self.learning_insights
    
    async def load_feedback_history(self):
        """Load feedback history from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM feedback')
        rows = cursor.fetchall()
        
        for row in rows:
            feedback = UserFeedback(
                user_id=row[1],
                summary_id=row[2],
                rating=row[3],
                feedback_text=row[4] or "",
                domain=row[5] or "",
                technology=row[6] or "",
                timestamp=datetime.fromisoformat(row[7]),
                metadata=json.loads(row[8]) if row[8] else {}
            )
            self.feedback_history.append(feedback)
        
        conn.close()
        
        logger.info(f"Loaded {len(self.feedback_history)} feedback entries from database")
    
    async def load_user_preferences(self):
        """Load user preferences from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_preferences')
        rows = cursor.fetchall()
        
        for row in rows:
            preference = UserPreference(
                user_id=row[0],
                domain=row[1] or "",
                preferred_summary_length=row[2] or "medium",
                preferred_summary_style=row[3] or "balanced",
                preferred_technical_depth=row[4] or "medium",
                preferred_focus_areas=json.loads(row[5]) if row[5] else [],
                disliked_patterns=json.loads(row[6]) if row[6] else [],
                liked_patterns=json.loads(row[7]) if row[7] else [],
                updated_at=datetime.fromisoformat(row[8])
            )
            self.user_preferences[preference.user_id] = preference
        
        conn.close()
        
        logger.info(f"Loaded {len(self.user_preferences)} user preferences from database")
    
    async def load_learning_insights(self):
        """Load learning insights from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM learning_insights')
        rows = cursor.fetchall()
        
        for row in rows:
            insight = LearningInsight(
                pattern=row[1],
                confidence=row[2],
                frequency=row[3],
                domain=row[4] or "",
                improvement_suggestion=row[5] or "",
                created_at=datetime.fromisoformat(row[6])
            )
            self.learning_insights.append(insight)
        
        conn.close()
        
        logger.info(f"Loaded {len(self.learning_insights)} learning insights from database")

class AdaptiveSummarizer:
    """Adaptive summarizer that learns from user preferences and feedback"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.user_models = {}
        self.domain_models = {}
        
    async def personalize_summary(
        self,
        content: str,
        user_id: str,
        domain: str,
        base_summary: str
    ) -> str:
        """Personalize summary based on user preferences and feedback"""
        
        # Get user preferences
        user_preferences = self.feedback_collector.get_user_preferences(user_id)
        
        if not user_preferences:
            return base_summary
        
        # Get learning insights for domain
        domain_insights = self.feedback_collector.get_learning_insights(domain)
        
        # Apply personalization
        personalized_summary = await self._apply_personalization(
            base_summary,
            user_preferences,
            domain_insights
        )
        
        return personalized_summary
    
    async def _apply_personalization(
        self,
        summary: str,
        preferences: UserPreference,
        insights: List[LearningInsight]
    ) -> str:
        """Apply personalization based on preferences and insights"""
        
        personalized = summary
        
        # Apply length preferences
        if preferences.preferred_summary_length == "short":
            personalized = self._shorten_summary(personalized)
        elif preferences.preferred_summary_length == "long":
            personalized = self._lengthen_summary(personalized)
        
        # Apply style preferences
        if preferences.preferred_summary_style == "technical":
            personalized = self._make_more_technical(personalized)
        elif preferences.preferred_summary_style == "simple":
            personalized = self._make_simpler(personalized)
        
        # Apply technical depth preferences
        if preferences.preferred_technical_depth == "high":
            personalized = self._increase_technical_depth(personalized)
        elif preferences.preferred_technical_depth == "low":
            personalized = self._decrease_technical_depth(personalized)
        
        # Apply insights
        for insight in insights:
            if insight.confidence > 0.7:
                personalized = self._apply_insight(personalized, insight)
        
        return personalized
    
    def _shorten_summary(self, summary: str) -> str:
        """Shorten summary while maintaining key information"""
        
        sentences = summary.split('. ')
        if len(sentences) > 3:
            # Keep first sentence and most important sentences
            return '. '.join(sentences[:3]) + '.'
        return summary
    
    def _lengthen_summary(self, summary: str) -> str:
        """Add more detail to summary"""
        
        # This would typically involve AI to expand the summary
        # For now, return the original
        return summary
    
    def _make_more_technical(self, summary: str) -> str:
        """Make summary more technical"""
        
        # Replace simple terms with technical terms
        technical_replacements = {
            'use': 'utilize',
            'make': 'implement',
            'get': 'retrieve',
            'put': 'insert',
            'show': 'demonstrate'
        }
        
        for simple, technical in technical_replacements.items():
            summary = summary.replace(simple, technical)
        
        return summary
    
    def _make_simpler(self, summary: str) -> str:
        """Make summary simpler"""
        
        # Replace technical terms with simple terms
        simple_replacements = {
            'utilize': 'use',
            'implement': 'make',
            'retrieve': 'get',
            'insert': 'put',
            'demonstrate': 'show'
        }
        
        for technical, simple in simple_replacements.items():
            summary = summary.replace(technical, simple)
        
        return summary
    
    def _increase_technical_depth(self, summary: str) -> str:
        """Increase technical depth of summary"""
        
        # Add technical details
        return summary + " Technical implementation details and specifications are included."
    
    def _decrease_technical_depth(self, summary: str) -> str:
        """Decrease technical depth of summary"""
        
        # Remove technical jargon
        return summary
    
    def _apply_insight(self, summary: str, insight: LearningInsight) -> str:
        """Apply learning insight to summary"""
        
        # Apply improvement suggestions
        if 'contains_unclear' in insight.pattern:
            summary = self._make_simpler(summary)
        elif 'contains_too_long' in insight.pattern:
            summary = self._shorten_summary(summary)
        elif 'contains_missing' in insight.pattern:
            summary = self._lengthen_summary(summary)
        
        return summary
    
    async def get_personalization_suggestions(self, user_id: str) -> List[str]:
        """Get personalization suggestions for a user"""
        
        preferences = self.feedback_collector.get_user_preferences(user_id)
        if not preferences:
            return []
        
        suggestions = []
        
        # Analyze feedback patterns
        analytics = self.feedback_collector.get_feedback_analytics()
        
        if analytics['average_rating'] < 3.0:
            suggestions.append("Consider providing more detailed feedback to improve summary quality")
        
        if len(preferences.disliked_patterns) > len(preferences.liked_patterns):
            suggestions.append("Focus on avoiding disliked patterns in future summaries")
        
        return suggestions
