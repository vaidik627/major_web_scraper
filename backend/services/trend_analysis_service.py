import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from urllib.parse import urlparse

# Import models
from models import ScrapedData, TrendData, ExtractedEntity, ContentCategory, User

class TrendAnalysisService:
    def __init__(self, db: Session):
        self.db = db

    async def analyze_trends(
        self, 
        user_id: int, 
        domain: Optional[str] = None,
        time_range: int = 30,  # days
        trend_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze trends for a user's scraped data"""
        
        if trend_types is None:
            trend_types = ["sentiment", "mentions", "categories", "entities"]
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range)
        
        trends = {
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": time_range
            },
            "domain": domain,
            "sentiment_trends": [],
            "mention_trends": [],
            "category_trends": [],
            "entity_trends": [],
            "price_trends": [],
            "volume_trends": [],
            "insights": []
        }
        
        try:
            # Get base query for scraped data
            base_query = self.db.query(ScrapedData).join(ScrapedData.job).filter(
                ScrapedData.job.has(user_id=user_id),
                ScrapedData.scraped_at >= start_date,
                ScrapedData.scraped_at <= end_date
            )
            
            if domain:
                base_query = base_query.filter(ScrapedData.url.like(f"%{domain}%"))
            
            scraped_data = base_query.all()
            
            if "sentiment" in trend_types:
                trends["sentiment_trends"] = await self._analyze_sentiment_trends(scraped_data, time_range)
            
            if "mentions" in trend_types:
                trends["mention_trends"] = await self._analyze_mention_trends(scraped_data, time_range)
            
            if "categories" in trend_types:
                trends["category_trends"] = await self._analyze_category_trends(user_id, domain, start_date, end_date)
            
            if "entities" in trend_types:
                trends["entity_trends"] = await self._analyze_entity_trends(user_id, domain, start_date, end_date)
            
            # Volume trends (always included)
            trends["volume_trends"] = await self._analyze_volume_trends(scraped_data, time_range)
            
            # Generate insights
            trends["insights"] = await self._generate_trend_insights(trends)
            
        except Exception as e:
            print(f"Error in trend analysis: {e}")
            trends["error"] = str(e)
        
        return trends

    async def _analyze_sentiment_trends(self, scraped_data: List[ScrapedData], time_range: int) -> List[Dict[str, Any]]:
        """Analyze sentiment trends over time"""
        sentiment_data = []
        
        try:
            # Group data by day
            daily_sentiments = defaultdict(list)
            
            for data in scraped_data:
                if data.ai_analysis and "sentiment" in data.ai_analysis:
                    date_key = data.scraped_at.date().isoformat()
                    sentiment = data.ai_analysis["sentiment"]
                    
                    if "polarity" in sentiment:
                        daily_sentiments[date_key].append(sentiment["polarity"])
            
            # Calculate daily averages
            for date, polarities in daily_sentiments.items():
                avg_polarity = np.mean(polarities)
                sentiment_label = "positive" if avg_polarity > 0.1 else "negative" if avg_polarity < -0.1 else "neutral"
                
                sentiment_data.append({
                    "date": date,
                    "average_polarity": round(avg_polarity, 3),
                    "sentiment_label": sentiment_label,
                    "sample_count": len(polarities),
                    "polarity_range": {
                        "min": round(min(polarities), 3),
                        "max": round(max(polarities), 3)
                    }
                })
            
            # Sort by date
            sentiment_data.sort(key=lambda x: x["date"])
            
        except Exception as e:
            print(f"Error in sentiment trend analysis: {e}")
        
        return sentiment_data

    async def _analyze_mention_trends(self, scraped_data: List[ScrapedData], time_range: int) -> List[Dict[str, Any]]:
        """Analyze mention frequency trends"""
        mention_data = []
        
        try:
            # Group by day and count mentions
            daily_mentions = defaultdict(int)
            daily_domains = defaultdict(set)
            
            for data in scraped_data:
                date_key = data.scraped_at.date().isoformat()
                daily_mentions[date_key] += 1
                
                # Extract domain from URL
                try:
                    domain = urlparse(data.url).netloc
                    daily_domains[date_key].add(domain)
                except:
                    pass
            
            for date, count in daily_mentions.items():
                mention_data.append({
                    "date": date,
                    "mention_count": count,
                    "unique_domains": len(daily_domains[date]),
                    "domains": list(daily_domains[date])
                })
            
            # Sort by date
            mention_data.sort(key=lambda x: x["date"])
            
        except Exception as e:
            print(f"Error in mention trend analysis: {e}")
        
        return mention_data

    async def _analyze_category_trends(self, user_id: int, domain: Optional[str], start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Analyze category trends over time"""
        category_data = []
        
        try:
            # Query category data
            query = self.db.query(ContentCategory).join(ContentCategory.scraped_data).join(ScrapedData.job).filter(
                ScrapedData.job.has(user_id=user_id),
                ContentCategory.created_at >= start_date,
                ContentCategory.created_at <= end_date
            )
            
            if domain:
                query = query.filter(ScrapedData.url.like(f"%{domain}%"))
            
            categories = query.all()
            
            # Group by date and category
            daily_categories = defaultdict(lambda: defaultdict(int))
            
            for cat in categories:
                date_key = cat.created_at.date().isoformat()
                daily_categories[date_key][cat.category] += 1
            
            # Convert to trend format
            all_dates = sorted(daily_categories.keys())
            all_categories = set()
            for date_data in daily_categories.values():
                all_categories.update(date_data.keys())
            
            for category in all_categories:
                trend_points = []
                for date in all_dates:
                    count = daily_categories[date].get(category, 0)
                    trend_points.append({
                        "date": date,
                        "count": count
                    })
                
                if trend_points:
                    category_data.append({
                        "category": category,
                        "trend_points": trend_points,
                        "total_mentions": sum(point["count"] for point in trend_points),
                        "peak_date": max(trend_points, key=lambda x: x["count"])["date"],
                        "peak_count": max(trend_points, key=lambda x: x["count"])["count"]
                    })
            
            # Sort by total mentions
            category_data.sort(key=lambda x: x["total_mentions"], reverse=True)
            
        except Exception as e:
            print(f"Error in category trend analysis: {e}")
        
        return category_data

    async def _analyze_entity_trends(self, user_id: int, domain: Optional[str], start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Analyze entity mention trends"""
        entity_data = []
        
        try:
            # Query entity data
            query = self.db.query(ExtractedEntity).join(ExtractedEntity.scraped_data).join(ScrapedData.job).filter(
                ScrapedData.job.has(user_id=user_id),
                ExtractedEntity.created_at >= start_date,
                ExtractedEntity.created_at <= end_date
            )
            
            if domain:
                query = query.filter(ScrapedData.url.like(f"%{domain}%"))
            
            entities = query.all()
            
            # Group by entity type and text
            entity_counts = defaultdict(lambda: defaultdict(int))
            entity_dates = defaultdict(lambda: defaultdict(list))
            
            for entity in entities:
                entity_key = f"{entity.entity_type}:{entity.entity_text}"
                date_key = entity.created_at.date().isoformat()
                
                entity_counts[entity.entity_type][entity.entity_text] += 1
                entity_dates[entity.entity_type][entity.entity_text].append(date_key)
            
            # Convert to trend format
            for entity_type, entities_of_type in entity_counts.items():
                type_data = {
                    "entity_type": entity_type,
                    "entities": []
                }
                
                # Get top entities for this type
                sorted_entities = sorted(entities_of_type.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for entity_text, count in sorted_entities:
                    dates = entity_dates[entity_type][entity_text]
                    date_counts = Counter(dates)
                    
                    trend_points = [{"date": date, "count": count} for date, count in date_counts.items()]
                    trend_points.sort(key=lambda x: x["date"])
                    
                    type_data["entities"].append({
                        "entity_text": entity_text,
                        "total_mentions": count,
                        "trend_points": trend_points,
                        "first_mention": min(dates),
                        "last_mention": max(dates)
                    })
                
                if type_data["entities"]:
                    entity_data.append(type_data)
            
        except Exception as e:
            print(f"Error in entity trend analysis: {e}")
        
        return entity_data

    async def _analyze_volume_trends(self, scraped_data: List[ScrapedData], time_range: int) -> List[Dict[str, Any]]:
        """Analyze scraping volume trends"""
        volume_data = []
        
        try:
            # Group by day
            daily_volumes = defaultdict(lambda: {
                "total_scraped": 0,
                "successful": 0,
                "failed": 0,
                "avg_processing_time": 0,
                "processing_times": []
            })
            
            for data in scraped_data:
                date_key = data.scraped_at.date().isoformat()
                daily_volumes[date_key]["total_scraped"] += 1
                
                if data.status == "success":
                    daily_volumes[date_key]["successful"] += 1
                else:
                    daily_volumes[date_key]["failed"] += 1
                
                if data.processing_time:
                    daily_volumes[date_key]["processing_times"].append(data.processing_time)
            
            # Calculate averages and create trend data
            for date, volume_info in daily_volumes.items():
                processing_times = volume_info["processing_times"]
                avg_processing_time = np.mean(processing_times) if processing_times else 0
                
                volume_data.append({
                    "date": date,
                    "total_scraped": volume_info["total_scraped"],
                    "successful": volume_info["successful"],
                    "failed": volume_info["failed"],
                    "success_rate": round(volume_info["successful"] / volume_info["total_scraped"] * 100, 2) if volume_info["total_scraped"] > 0 else 0,
                    "avg_processing_time": round(avg_processing_time, 2)
                })
            
            # Sort by date
            volume_data.sort(key=lambda x: x["date"])
            
        except Exception as e:
            print(f"Error in volume trend analysis: {e}")
        
        return volume_data

    async def _generate_trend_insights(self, trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from trend analysis"""
        insights = []
        
        try:
            # Sentiment insights
            if trends["sentiment_trends"]:
                sentiment_data = trends["sentiment_trends"]
                if len(sentiment_data) >= 2:
                    recent_sentiment = sentiment_data[-1]["average_polarity"]
                    previous_sentiment = sentiment_data[-2]["average_polarity"]
                    
                    if recent_sentiment > previous_sentiment + 0.1:
                        insights.append({
                            "type": "sentiment_improvement",
                            "insight": "Sentiment has improved in recent data",
                            "confidence": 0.8,
                            "data": {
                                "current": recent_sentiment,
                                "previous": previous_sentiment,
                                "change": round(recent_sentiment - previous_sentiment, 3)
                            }
                        })
                    elif recent_sentiment < previous_sentiment - 0.1:
                        insights.append({
                            "type": "sentiment_decline",
                            "insight": "Sentiment has declined in recent data",
                            "confidence": 0.8,
                            "data": {
                                "current": recent_sentiment,
                                "previous": previous_sentiment,
                                "change": round(recent_sentiment - previous_sentiment, 3)
                            }
                        })
            
            # Volume insights
            if trends["volume_trends"]:
                volume_data = trends["volume_trends"]
                if len(volume_data) >= 7:  # At least a week of data
                    recent_week = volume_data[-7:]
                    avg_recent_volume = np.mean([day["total_scraped"] for day in recent_week])
                    
                    if len(volume_data) >= 14:
                        previous_week = volume_data[-14:-7]
                        avg_previous_volume = np.mean([day["total_scraped"] for day in previous_week])
                        
                        if avg_recent_volume > avg_previous_volume * 1.2:
                            insights.append({
                                "type": "volume_increase",
                                "insight": f"Scraping volume increased by {round((avg_recent_volume - avg_previous_volume) / avg_previous_volume * 100, 1)}%",
                                "confidence": 0.9,
                                "data": {
                                    "recent_avg": round(avg_recent_volume, 1),
                                    "previous_avg": round(avg_previous_volume, 1)
                                }
                            })
            
            # Category insights
            if trends["category_trends"]:
                top_category = trends["category_trends"][0] if trends["category_trends"] else None
                if top_category:
                    insights.append({
                        "type": "dominant_category",
                        "insight": f"'{top_category['category']}' is the most frequently scraped category",
                        "confidence": 0.9,
                        "data": {
                            "category": top_category["category"],
                            "mentions": top_category["total_mentions"]
                        }
                    })
            
            # Entity insights
            if trends["entity_trends"]:
                for entity_type_data in trends["entity_trends"]:
                    if entity_type_data["entities"]:
                        top_entity = entity_type_data["entities"][0]
                        insights.append({
                            "type": "top_entity",
                            "insight": f"Most mentioned {entity_type_data['entity_type'].lower()}: '{top_entity['entity_text']}'",
                            "confidence": 0.8,
                            "data": {
                                "entity_type": entity_type_data["entity_type"],
                                "entity_text": top_entity["entity_text"],
                                "mentions": top_entity["total_mentions"]
                            }
                        })
            
        except Exception as e:
            print(f"Error generating trend insights: {e}")
        
        return insights

    async def store_trend_data(
        self, 
        user_id: int, 
        domain: str, 
        trend_type: str, 
        metric_name: str, 
        metric_value: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store trend data point"""
        try:
            trend_data = TrendData(
                user_id=user_id,
                domain=domain,
                trend_type=trend_type,
                metric_name=metric_name,
                metric_value=metric_value,
                trend_metadata=metadata or {}
            )
            
            self.db.add(trend_data)
            self.db.commit()
            return True
            
        except Exception as e:
            print(f"Error storing trend data: {e}")
            self.db.rollback()
            return False

    async def get_historical_trends(
        self, 
        user_id: int, 
        trend_type: str, 
        metric_name: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical trend data"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            trends = self.db.query(TrendData).filter(
                TrendData.user_id == user_id,
                TrendData.trend_type == trend_type,
                TrendData.metric_name == metric_name,
                TrendData.recorded_at >= start_date,
                TrendData.recorded_at <= end_date
            ).order_by(TrendData.recorded_at).all()
            
            return [
                {
                    "date": trend.recorded_at.isoformat(),
                    "value": trend.metric_value,
                    "domain": trend.domain,
                    "metadata": trend.trend_metadata
                }
                for trend in trends
            ]
            
        except Exception as e:
            print(f"Error getting historical trends: {e}")
            return []

    async def compare_domains(
        self, 
        user_id: int, 
        domains: List[str], 
        metric: str = "sentiment",
        days: int = 30
    ) -> Dict[str, Any]:
        """Compare trends across different domains"""
        comparison = {
            "domains": domains,
            "metric": metric,
            "time_range": days,
            "comparison_data": {},
            "insights": []
        }
        
        try:
            for domain in domains:
                domain_trends = await self.analyze_trends(
                    user_id=user_id,
                    domain=domain,
                    time_range=days,
                    trend_types=[metric]
                )
                comparison["comparison_data"][domain] = domain_trends
            
            # Generate comparison insights
            if metric == "sentiment" and len(domains) >= 2:
                domain_sentiments = {}
                for domain, data in comparison["comparison_data"].items():
                    if data["sentiment_trends"]:
                        avg_sentiment = np.mean([point["average_polarity"] for point in data["sentiment_trends"]])
                        domain_sentiments[domain] = avg_sentiment
                
                if domain_sentiments:
                    best_domain = max(domain_sentiments.items(), key=lambda x: x[1])
                    worst_domain = min(domain_sentiments.items(), key=lambda x: x[1])
                    
                    comparison["insights"].append({
                        "type": "domain_comparison",
                        "insight": f"{best_domain[0]} has the most positive sentiment ({best_domain[1]:.3f}), while {worst_domain[0]} has the least positive ({worst_domain[1]:.3f})",
                        "confidence": 0.8
                    })
            
        except Exception as e:
            print(f"Error in domain comparison: {e}")
            comparison["error"] = str(e)
        
        return comparison