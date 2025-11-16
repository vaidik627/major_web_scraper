"""
Enhanced AI Integration Service
Integrates all enhanced AI components for comprehensive content analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .multi_model_ensemble import MultiModelEnsemble, ConsensusSummary
from .domain_aware_classifier import DomainAwareClassifier, DomainSpecificSummarizer, ContentDomain
from .tech_specialized_scraper import TechDomainScraper, TechDomain, TechScrapedContent
from .tech_knowledge_graph import TechnologyKnowledgeGraph, TechNodeType
from .feedback_learning_system import FeedbackCollector, AdaptiveSummarizer

logger = logging.getLogger(__name__)

@dataclass
class EnhancedAnalysisResult:
    summary: str
    domain: str
    confidence: float
    technical_elements: Dict[str, Any]
    related_technologies: List[Dict[str, Any]]
    learning_insights: List[Dict[str, Any]]
    personalized_for_user: bool
    metadata: Dict[str, Any]

class EnhancedAIIntegrationService:
    """Main service that integrates all enhanced AI components"""
    
    def __init__(
        self,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        feedback_db_path: str = "feedback.db"
    ):
        # Initialize all components
        self.multi_model_ensemble = MultiModelEnsemble(openai_api_key, anthropic_api_key)
        self.domain_classifier = DomainAwareClassifier()
        self.domain_summarizer = DomainSpecificSummarizer()
        self.tech_scraper = TechDomainScraper()
        self.knowledge_graph = TechnologyKnowledgeGraph()
        self.feedback_collector = FeedbackCollector(feedback_db_path)
        self.adaptive_summarizer = AdaptiveSummarizer(self.feedback_collector)
        
        # Initialize feedback system (disabled for quick startup)
        # asyncio.create_task(self._initialize_feedback_system())
        logger.info("Feedback system initialization disabled for quick startup")
    
    async def _initialize_feedback_system(self):
        """Initialize feedback system by loading existing data"""
        
        try:
            await self.feedback_collector.load_feedback_history()
            await self.feedback_collector.load_user_preferences()
            await self.feedback_collector.load_learning_insights()
            logger.info("Feedback system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize feedback system: {e}")
    
    async def analyze_content_enhanced(
        self,
        content: str,
        title: str = "",
        url: str = "",
        user_id: str = None,
        max_length: int = 500,
        enable_personalization: bool = True
    ) -> EnhancedAnalysisResult:
        """Perform comprehensive enhanced content analysis"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Domain Classification
            domain_classification = self.domain_classifier.classify_content(content, title, url)
            domain = domain_classification.domain.value
            
            logger.info(f"Content classified as: {domain} (confidence: {domain_classification.confidence:.2f})")
            
            # Step 2: Technology Detection (if applicable)
            tech_domain = None
            related_technologies = []
            
            if domain in ['tech', 'ai', 'webdev', 'mobile', 'devops']:
                tech_domain = self._map_domain_to_tech_domain(domain)
                related_technologies = await self._find_related_technologies(content, tech_domain)
            
            # Step 3: Multi-Model Summary Generation
            consensus_summary = await self.multi_model_ensemble.generate_consensus_summary(
                content=content,
                title=title,
                url=url,
                max_length=max_length,
                domain=domain
            )
            
            # Step 4: Domain-Specific Enhancement
            domain_summary = self.domain_summarizer.generate_domain_summary(
                content=content,
                title=title,
                url=url,
                max_length=max_length
            )
            
            # Step 5: Personalization (if user_id provided)
            final_summary = consensus_summary.final_summary
            personalized_for_user = False
            
            if enable_personalization and user_id:
                final_summary = await self.adaptive_summarizer.personalize_summary(
                    content=content,
                    user_id=user_id,
                    domain=domain,
                    base_summary=consensus_summary.final_summary
                )
                personalized_for_user = True
            
            # Step 6: Extract Technical Elements
            technical_elements = await self._extract_technical_elements(
                content, domain, tech_domain
            )
            
            # Step 7: Get Learning Insights
            learning_insights = self._get_relevant_learning_insights(domain, user_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedAnalysisResult(
                summary=final_summary,
                domain=domain,
                confidence=consensus_summary.consensus_score,
                technical_elements=technical_elements,
                related_technologies=related_technologies,
                learning_insights=learning_insights,
                personalized_for_user=personalized_for_user,
                metadata={
                    'processing_time': processing_time,
                    'models_used': len(consensus_summary.model_results),
                    'consensus_agreement': consensus_summary.agreement_level,
                    'domain_confidence': domain_classification.confidence,
                    'technical_elements_count': len(technical_elements),
                    'related_technologies_count': len(related_technologies),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            raise e
    
    async def scrape_and_analyze_tech_content(
        self,
        urls: List[str],
        tech_domain: TechDomain,
        user_id: str = None,
        enable_personalization: bool = True
    ) -> List[EnhancedAnalysisResult]:
        """Scrape technology-specific content and analyze it"""
        
        results = []
        
        # Scrape content with domain-specific configuration
        scraped_content = await self.tech_scraper.scrape_tech_content(urls, tech_domain)
        
        for content in scraped_content:
            try:
                # Analyze scraped content
                analysis = await self.analyze_content_enhanced(
                    content=content.content,
                    title=content.title,
                    url=content.url,
                    user_id=user_id,
                    enable_personalization=enable_personalization
                )
                
                # Enhance with scraped technical elements
                analysis.technical_elements.update({
                    'code_blocks': content.code_blocks,
                    'api_endpoints': content.api_endpoints,
                    'documentation': content.documentation,
                    'scraped_metadata': content.metadata
                })
                
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze scraped content from {content.url}: {e}")
                continue
        
        return results
    
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
        
        await self.feedback_collector.collect_feedback(
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
        
        preferences = self.feedback_collector.get_user_preferences(user_id)
        analytics = self.feedback_collector.get_feedback_analytics()
        suggestions = await self.adaptive_summarizer.get_personalization_suggestions(user_id)
        
        return {
            'preferences': preferences.__dict__ if preferences else None,
            'analytics': analytics,
            'suggestions': suggestions,
            'learning_insights': self.feedback_collector.get_learning_insights()
        }
    
    async def get_technology_ecosystem(self, technology: str) -> Dict[str, Any]:
        """Get technology ecosystem information"""
        
        return self.knowledge_graph.get_technology_ecosystem(technology)
    
    async def find_learning_path(
        self,
        start_technology: str,
        target_technology: str,
        max_steps: int = 5
    ) -> List[Dict[str, Any]]:
        """Find learning path between technologies"""
        
        return self.knowledge_graph.find_learning_path(
            start_technology, target_technology, max_steps
        )
    
    async def suggest_technologies(
        self,
        current_technologies: List[str],
        target_domain: str = None,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Suggest technologies based on current stack"""
        
        return self.knowledge_graph.suggest_technologies(
            current_technologies, target_domain, max_suggestions
        )
    
    def _map_domain_to_tech_domain(self, domain: str) -> Optional[TechDomain]:
        """Map content domain to technology domain"""
        
        mapping = {
            'tech': TechDomain.WEB_DEVELOPMENT,
            'ai': TechDomain.AI_ML,
            'webdev': TechDomain.WEB_DEVELOPMENT,
            'mobile': TechDomain.MOBILE_DEVELOPMENT,
            'devops': TechDomain.DEVOPS,
            'cybersecurity': TechDomain.CYBERSECURITY,
            'data_science': TechDomain.DATA_SCIENCE,
            'blockchain': TechDomain.BLOCKCHAIN,
            'cloud': TechDomain.CLOUD_COMPUTING,
            'game_dev': TechDomain.GAME_DEVELOPMENT,
            'embedded': TechDomain.EMBEDDED_SYSTEMS
        }
        
        return mapping.get(domain)
    
    async def _find_related_technologies(
        self,
        content: str,
        tech_domain: TechDomain
    ) -> List[Dict[str, Any]]:
        """Find technologies related to the content"""
        
        if not tech_domain:
            return []
        
        # Extract technology mentions from content
        tech_mentions = self._extract_technology_mentions(content)
        
        related_tech = []
        for mention in tech_mentions:
            # Find related technologies in knowledge graph
            related = self.knowledge_graph.find_related_technologies(mention)
            
            for tech_id, relationship, strength in related[:3]:  # Top 3 related
                if tech_id in self.knowledge_graph.nodes:
                    node = self.knowledge_graph.nodes[tech_id]
                    related_tech.append({
                        'id': node.id,
                        'name': node.name,
                        'type': node.node_type.value,
                        'relationship': relationship,
                        'strength': strength,
                        'description': node.description
                    })
        
        return related_tech
    
    def _extract_technology_mentions(self, content: str) -> List[str]:
        """Extract technology mentions from content"""
        
        mentions = []
        
        # Check against known technologies in knowledge graph
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.name.lower() in content.lower():
                mentions.append(node_id)
        
        return mentions
    
    async def _extract_technical_elements(
        self,
        content: str,
        domain: str,
        tech_domain: Optional[TechDomain]
    ) -> Dict[str, Any]:
        """Extract technical elements from content"""
        
        elements = {
            'code_blocks': [],
            'api_endpoints': [],
            'technical_terms': [],
            'frameworks': [],
            'libraries': [],
            'tools': []
        }
        
        # Extract code blocks
        import re
        code_patterns = [
            r'```[\s\S]*?```',
            r'<code>[\s\S]*?</code>',
            r'<pre>[\s\S]*?</pre>'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, content)
            elements['code_blocks'].extend(matches)
        
        # Extract API endpoints
        api_patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+[^\s]+',
            r'https?://[^\s]+/api/[^\s]+',
            r'/[a-zA-Z0-9/_-]+\.json'
        ]
        
        for pattern in api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            elements['api_endpoints'].extend(matches)
        
        # Extract technical terms based on domain
        if tech_domain:
            domain_keywords = self.tech_scraper.domain_configs.get(tech_domain)
            if domain_keywords:
                for keyword in domain_keywords.keywords:
                    if keyword.lower() in content.lower():
                        elements['technical_terms'].append(keyword)
        
        # Extract frameworks and libraries
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.node_type in [TechNodeType.FRAMEWORK, TechNodeType.LIBRARY]:
                if node.name.lower() in content.lower():
                    elements['frameworks'].append({
                        'name': node.name,
                        'type': node.node_type.value,
                        'description': node.description
                    })
        
        return elements
    
    def _get_relevant_learning_insights(
        self,
        domain: str,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """Get relevant learning insights for domain and user"""
        
        insights = self.feedback_collector.get_learning_insights(domain)
        
        # Filter by confidence and frequency
        relevant_insights = [
            insight for insight in insights
            if insight.confidence > 0.6 and insight.frequency > 1
        ]
        
        return [
            {
                'pattern': insight.pattern,
                'confidence': insight.confidence,
                'frequency': insight.frequency,
                'suggestion': insight.improvement_suggestion
            }
            for insight in relevant_insights
        ]
    
    async def get_domain_analytics(self, domain: str = None) -> Dict[str, Any]:
        """Get analytics for a specific domain"""
        
        analytics = self.feedback_collector.get_feedback_analytics(domain)
        
        # Add technology ecosystem data if domain is tech-related
        if domain in ['tech', 'ai', 'webdev', 'mobile', 'devops']:
            tech_domain = self._map_domain_to_tech_domain(domain)
            if tech_domain:
                domain_technologies = self.knowledge_graph.get_domain_technologies(tech_domain.value)
                trending_tech = self.knowledge_graph.find_trending_technologies(tech_domain.value)
                
                analytics.update({
                    'domain_technologies': domain_technologies,
                    'trending_technologies': trending_tech
                })
        
        return analytics
    
    async def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export the knowledge graph"""
        
        return self.knowledge_graph.export_graph()
    
    async def import_knowledge_graph(self, graph_data: Dict[str, Any]):
        """Import knowledge graph data"""
        
        self.knowledge_graph.import_graph(graph_data)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics"""
        
        return {
            'multi_model_ensemble': {
                'openai_available': self.multi_model_ensemble.openai_client is not None,
                'anthropic_available': self.multi_model_ensemble.anthropic_client is not None,
                'local_models_available': (
                    self.multi_model_ensemble.local_model is not None or
                    self.multi_model_ensemble.summarization_pipeline is not None
                )
            },
            'knowledge_graph': {
                'total_nodes': len(self.knowledge_graph.nodes),
                'total_relationships': len(self.knowledge_graph.relationships),
                'domains_covered': len(self.knowledge_graph.domain_clusters)
            },
            'feedback_system': {
                'total_feedback': len(self.feedback_collector.feedback_history),
                'total_users': len(self.feedback_collector.user_preferences),
                'learning_insights': len(self.feedback_collector.learning_insights)
            },
            'tech_scraper': {
                'domains_supported': len(self.tech_scraper.domain_configs),
                'playwright_available': True  # Would check actual availability
            }
        }
