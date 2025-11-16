"""
Domain-Aware Content Classifier and Summarizer
Automatically detects content domain and applies specialized processing
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import Counter
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class ContentDomain(Enum):
    TECHNOLOGY = "tech"
    AI_ML = "ai"
    WEB_DEVELOPMENT = "webdev"
    MOBILE_DEVELOPMENT = "mobile"
    DEVOPS = "devops"
    CYBERSECURITY = "cybersecurity"
    FINANCE = "finance"
    MEDICAL = "medical"
    LEGAL = "legal"
    ACADEMIC = "academic"
    BUSINESS = "business"
    GENERAL = "general"

@dataclass
class DomainClassification:
    domain: ContentDomain
    confidence: float
    keywords: List[str]
    indicators: List[str]
    metadata: Dict[str, Any]

@dataclass
class DomainSpecificSummary:
    summary: str
    domain: ContentDomain
    technical_terms: List[str]
    key_concepts: List[str]
    actionable_insights: List[str]
    metadata: Dict[str, Any]

class DomainAwareClassifier:
    """Classifies content into specific domains for targeted processing"""
    
    def __init__(self):
        self.domain_patterns = self._build_domain_patterns()
        self.domain_keywords = self._build_domain_keywords()
        self.domain_indicators = self._build_domain_indicators()
        
    def _build_domain_patterns(self) -> Dict[ContentDomain, List[str]]:
        """Build regex patterns for domain detection"""
        
        return {
            ContentDomain.TECHNOLOGY: [
                r'\b(?:API|SDK|framework|library|algorithm|database|server|client)\b',
                r'\b(?:programming|coding|development|software|hardware)\b',
                r'\b(?:code|function|class|method|variable|parameter)\b',
                r'\b(?:git|github|repository|version control|deployment)\b'
            ],
            ContentDomain.AI_ML: [
                r'\b(?:machine learning|ML|artificial intelligence|AI|neural network)\b',
                r'\b(?:deep learning|reinforcement learning|supervised|unsupervised)\b',
                r'\b(?:model|training|inference|prediction|classification|regression)\b',
                r'\b(?:tensorflow|pytorch|keras|scikit-learn|pandas|numpy)\b',
                r'\b(?:algorithm|optimization|gradient descent|backpropagation)\b'
            ],
            ContentDomain.WEB_DEVELOPMENT: [
                r'\b(?:HTML|CSS|JavaScript|React|Vue|Angular|Node\.js)\b',
                r'\b(?:frontend|backend|full-stack|responsive|mobile-first)\b',
                r'\b(?:REST|GraphQL|API|endpoint|HTTP|HTTPS|JSON|XML)\b',
                r'\b(?:bootstrap|tailwind|webpack|babel|npm|yarn)\b',
                r'\b(?:DOM|virtual DOM|component|state|props|hooks)\b'
            ],
            ContentDomain.MOBILE_DEVELOPMENT: [
                r'\b(?:iOS|Android|React Native|Flutter|Xamarin|Swift|Kotlin)\b',
                r'\b(?:mobile app|native|hybrid|cross-platform|App Store|Play Store)\b',
                r'\b(?:UI|UX|gesture|touch|swipe|pinch|zoom)\b',
                r'\b(?:push notification|location|camera|sensor|accelerometer)\b'
            ],
            ContentDomain.DEVOPS: [
                r'\b(?:Docker|Kubernetes|Jenkins|CI\/CD|pipeline|deployment)\b',
                r'\b(?:AWS|Azure|GCP|cloud|infrastructure|microservices)\b',
                r'\b(?:monitoring|logging|scaling|load balancing|container)\b',
                r'\b(?:terraform|ansible|chef|puppet|infrastructure as code)\b'
            ],
            ContentDomain.CYBERSECURITY: [
                r'\b(?:security|vulnerability|exploit|penetration testing|firewall)\b',
                r'\b(?:encryption|authentication|authorization|SSL|TLS|HTTPS)\b',
                r'\b(?:malware|virus|trojan|phishing|social engineering)\b',
                r'\b(?:OWASP|NIST|ISO 27001|GDPR|compliance|audit)\b'
            ],
            ContentDomain.FINANCE: [
                r'\b(?:investment|portfolio|trading|stocks|bonds|mutual funds)\b',
                r'\b(?:ROI|ROE|P\/E ratio|market cap|dividend|yield)\b',
                r'\b(?:financial|economic|revenue|profit|loss|balance sheet)\b',
                r'\b(?:banking|fintech|blockchain|cryptocurrency|bitcoin)\b'
            ],
            ContentDomain.MEDICAL: [
                r'\b(?:medical|healthcare|clinical|patient|diagnosis|treatment)\b',
                r'\b(?:drug|medication|therapy|surgery|procedure|symptom)\b',
                r'\b(?:FDA|clinical trial|research|study|evidence-based)\b',
                r'\b(?:anatomy|physiology|pathology|pharmacology|immunology)\b'
            ],
            ContentDomain.LEGAL: [
                r'\b(?:legal|law|attorney|lawyer|court|judge|jury)\b',
                r'\b(?:contract|agreement|liability|damages|settlement)\b',
                r'\b(?:regulation|compliance|statute|precedent|jurisdiction)\b',
                r'\b(?:intellectual property|copyright|patent|trademark)\b'
            ],
            ContentDomain.ACADEMIC: [
                r'\b(?:research|study|experiment|hypothesis|methodology)\b',
                r'\b(?:peer review|journal|publication|citation|reference)\b',
                r'\b(?:university|college|academic|scholarly|thesis|dissertation)\b',
                r'\b(?:data analysis|statistics|correlation|significance)\b'
            ],
            ContentDomain.BUSINESS: [
                r'\b(?:business|strategy|marketing|sales|customer|revenue)\b',
                r'\b(?:startup|entrepreneur|venture capital|funding|IPO)\b',
                r'\b(?:management|leadership|team|organization|culture)\b',
                r'\b(?:product|service|market|competition|brand|growth)\b'
            ]
        }
    
    def _build_domain_keywords(self) -> Dict[ContentDomain, List[str]]:
        """Build keyword lists for each domain"""
        
        return {
            ContentDomain.TECHNOLOGY: [
                'programming', 'coding', 'software', 'hardware', 'computer',
                'technology', 'digital', 'system', 'application', 'platform',
                'interface', 'database', 'server', 'client', 'network'
            ],
            ContentDomain.AI_ML: [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'algorithm', 'model', 'training', 'data',
                'prediction', 'classification', 'regression', 'clustering',
                'optimization', 'tensorflow', 'pytorch', 'keras'
            ],
            ContentDomain.WEB_DEVELOPMENT: [
                'web development', 'frontend', 'backend', 'full-stack',
                'HTML', 'CSS', 'JavaScript', 'React', 'Vue', 'Angular',
                'Node.js', 'API', 'REST', 'GraphQL', 'responsive'
            ],
            ContentDomain.MOBILE_DEVELOPMENT: [
                'mobile', 'app', 'iOS', 'Android', 'React Native', 'Flutter',
                'native', 'hybrid', 'cross-platform', 'mobile-first'
            ],
            ContentDomain.DEVOPS: [
                'devops', 'deployment', 'CI/CD', 'Docker', 'Kubernetes',
                'AWS', 'Azure', 'cloud', 'infrastructure', 'monitoring'
            ],
            ContentDomain.CYBERSECURITY: [
                'security', 'cybersecurity', 'vulnerability', 'encryption',
                'authentication', 'firewall', 'malware', 'penetration testing'
            ],
            ContentDomain.FINANCE: [
                'finance', 'financial', 'investment', 'trading', 'banking',
                'revenue', 'profit', 'market', 'economy', 'cryptocurrency'
            ],
            ContentDomain.MEDICAL: [
                'medical', 'healthcare', 'clinical', 'patient', 'treatment',
                'drug', 'therapy', 'diagnosis', 'research', 'study'
            ],
            ContentDomain.LEGAL: [
                'legal', 'law', 'attorney', 'court', 'contract', 'liability',
                'regulation', 'compliance', 'intellectual property'
            ],
            ContentDomain.ACADEMIC: [
                'research', 'study', 'academic', 'university', 'scholarly',
                'publication', 'journal', 'thesis', 'methodology'
            ],
            ContentDomain.BUSINESS: [
                'business', 'strategy', 'marketing', 'sales', 'management',
                'startup', 'entrepreneur', 'revenue', 'growth', 'market'
            ]
        }
    
    def _build_domain_indicators(self) -> Dict[ContentDomain, List[str]]:
        """Build domain-specific indicators"""
        
        return {
            ContentDomain.TECHNOLOGY: ['code', 'function', 'class', 'method', 'variable'],
            ContentDomain.AI_ML: ['model', 'training', 'inference', 'prediction', 'algorithm'],
            ContentDomain.WEB_DEVELOPMENT: ['component', 'state', 'props', 'DOM', 'API'],
            ContentDomain.MOBILE_DEVELOPMENT: ['app', 'native', 'hybrid', 'gesture', 'touch'],
            ContentDomain.DEVOPS: ['container', 'pipeline', 'deployment', 'monitoring', 'scaling'],
            ContentDomain.CYBERSECURITY: ['vulnerability', 'exploit', 'encryption', 'firewall', 'malware'],
            ContentDomain.FINANCE: ['investment', 'portfolio', 'trading', 'ROI', 'market'],
            ContentDomain.MEDICAL: ['patient', 'diagnosis', 'treatment', 'clinical', 'drug'],
            ContentDomain.LEGAL: ['contract', 'liability', 'regulation', 'compliance', 'court'],
            ContentDomain.ACADEMIC: ['research', 'study', 'publication', 'methodology', 'hypothesis'],
            ContentDomain.BUSINESS: ['strategy', 'marketing', 'revenue', 'growth', 'customer']
        }
    
    def classify_content(self, content: str, title: str = "", url: str = "") -> DomainClassification:
        """Classify content into specific domain"""
        
        # Combine content sources
        full_text = f"{title} {content}".lower()
        
        domain_scores = {}
        detected_keywords = {}
        detected_indicators = {}
        
        # Score each domain
        for domain in ContentDomain:
            if domain == ContentDomain.GENERAL:
                continue
                
            score = 0.0
            keywords_found = []
            indicators_found = []
            
            # Check patterns
            patterns = self.domain_patterns.get(domain, [])
            for pattern in patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                score += len(matches) * 0.1
            
            # Check keywords
            keywords = self.domain_keywords.get(domain, [])
            for keyword in keywords:
                if keyword.lower() in full_text:
                    score += 0.2
                    keywords_found.append(keyword)
            
            # Check indicators
            indicators = self.domain_indicators.get(domain, [])
            for indicator in indicators:
                if indicator.lower() in full_text:
                    score += 0.3
                    indicators_found.append(indicator)
            
            # URL-based scoring
            if url:
                url_score = self._score_url_domain(url, domain)
                score += url_score
            
            domain_scores[domain] = score
            detected_keywords[domain] = keywords_found
            detected_indicators[domain] = indicators_found
        
        # Find best domain
        if not domain_scores or max(domain_scores.values()) == 0:
            best_domain = ContentDomain.GENERAL
            confidence = 0.1
        else:
            best_domain = max(domain_scores, key=domain_scores.get)
            max_score = domain_scores[best_domain]
            confidence = min(1.0, max_score / 5.0)  # Normalize to 0-1
        
        return DomainClassification(
            domain=best_domain,
            confidence=confidence,
            keywords=detected_keywords.get(best_domain, []),
            indicators=detected_indicators.get(best_domain, []),
            metadata={
                'all_scores': domain_scores,
                'url': url,
                'title': title,
                'content_length': len(content)
            }
        )
    
    def _score_url_domain(self, url: str, domain: ContentDomain) -> float:
        """Score domain based on URL patterns"""
        
        url_lower = url.lower()
        
        url_patterns = {
            ContentDomain.TECHNOLOGY: ['github.com', 'stackoverflow.com', 'dev.to', 'medium.com'],
            ContentDomain.AI_ML: ['arxiv.org', 'paperswithcode.com', 'openai.com', 'huggingface.co'],
            ContentDomain.WEB_DEVELOPMENT: ['mdn.mozilla.org', 'w3schools.com', 'css-tricks.com'],
            ContentDomain.MOBILE_DEVELOPMENT: ['developer.apple.com', 'developer.android.com'],
            ContentDomain.DEVOPS: ['kubernetes.io', 'docker.com', 'aws.amazon.com'],
            ContentDomain.CYBERSECURITY: ['owasp.org', 'nist.gov', 'cve.mitre.org'],
            ContentDomain.FINANCE: ['bloomberg.com', 'reuters.com', 'finance.yahoo.com'],
            ContentDomain.MEDICAL: ['pubmed.ncbi.nlm.nih.gov', 'who.int', 'cdc.gov'],
            ContentDomain.LEGAL: ['law.cornell.edu', 'justia.com', 'findlaw.com'],
            ContentDomain.ACADEMIC: ['scholar.google.com', 'jstor.org', 'ieee.org'],
            ContentDomain.BUSINESS: ['forbes.com', 'businessinsider.com', 'wsj.com']
        }
        
        patterns = url_patterns.get(domain, [])
        for pattern in patterns:
            if pattern in url_lower:
                return 0.5
        
        return 0.0

class DomainSpecificSummarizer:
    """Generates domain-specific summaries with specialized processing"""
    
    def __init__(self):
        self.classifier = DomainAwareClassifier()
        self.domain_processors = self._initialize_domain_processors()
    
    def _initialize_domain_processors(self) -> Dict[ContentDomain, Any]:
        """Initialize domain-specific processors"""
        
        return {
            ContentDomain.TECHNOLOGY: TechContentProcessor(),
            ContentDomain.AI_ML: AIMLContentProcessor(),
            ContentDomain.WEB_DEVELOPMENT: WebDevContentProcessor(),
            ContentDomain.MOBILE_DEVELOPMENT: MobileDevContentProcessor(),
            ContentDomain.DEVOPS: DevOpsContentProcessor(),
            ContentDomain.CYBERSECURITY: SecurityContentProcessor(),
            ContentDomain.FINANCE: FinanceContentProcessor(),
            ContentDomain.MEDICAL: MedicalContentProcessor(),
            ContentDomain.LEGAL: LegalContentProcessor(),
            ContentDomain.ACADEMIC: AcademicContentProcessor(),
            ContentDomain.BUSINESS: BusinessContentProcessor(),
            ContentDomain.GENERAL: GeneralContentProcessor()
        }
    
    def generate_domain_summary(
        self,
        content: str,
        title: str = "",
        url: str = "",
        max_length: int = 500
    ) -> DomainSpecificSummary:
        """Generate domain-specific summary"""
        
        # Classify content
        classification = self.classifier.classify_content(content, title, url)
        
        # Get domain processor
        processor = self.domain_processors.get(classification.domain, self.domain_processors[ContentDomain.GENERAL])
        
        # Generate domain-specific summary
        summary_data = processor.process_content(content, title, url, max_length)
        
        return DomainSpecificSummary(
            summary=summary_data['summary'],
            domain=classification.domain,
            technical_terms=summary_data.get('technical_terms', []),
            key_concepts=summary_data.get('key_concepts', []),
            actionable_insights=summary_data.get('actionable_insights', []),
            metadata={
                'classification': classification,
                'processing_time': summary_data.get('processing_time', 0),
                'confidence': classification.confidence
            }
        )

class BaseContentProcessor:
    """Base class for domain-specific content processors"""
    
    def process_content(self, content: str, title: str, url: str, max_length: int) -> Dict[str, Any]:
        """Process content and generate summary"""
        
        # Extract technical terms
        technical_terms = self._extract_technical_terms(content)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(content)
        
        # Generate actionable insights
        actionable_insights = self._extract_actionable_insights(content)
        
        # Generate summary
        summary = self._generate_summary(content, title, max_length)
        
        return {
            'summary': summary,
            'technical_terms': technical_terms,
            'key_concepts': key_concepts,
            'actionable_insights': actionable_insights,
            'processing_time': 0
        }
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content"""
        # Base implementation - can be overridden by subclasses
        return []
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Base implementation - can be overridden by subclasses
        return []
    
    def _extract_actionable_insights(self, content: str) -> List[str]:
        """Extract actionable insights from content"""
        # Base implementation - can be overridden by subclasses
        return []
    
    def _generate_summary(self, content: str, title: str, max_length: int) -> str:
        """Generate summary"""
        # Base implementation - can be overridden by subclasses
        return content[:max_length]

class TechContentProcessor(BaseContentProcessor):
    """Processor for technology content"""
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technology-specific terms"""
        
        tech_patterns = [
            r'\b(?:API|SDK|framework|library|algorithm|database|server|client)\b',
            r'\b(?:programming|coding|development|software|hardware)\b',
            r'\b(?:code|function|class|method|variable|parameter)\b'
        ]
        
        terms = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key technology concepts"""
        
        concepts = []
        
        # Look for code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        if code_blocks:
            concepts.append("Code examples provided")
        
        # Look for API references
        api_refs = re.findall(r'\b(?:GET|POST|PUT|DELETE)\s+[^\s]+', content)
        if api_refs:
            concepts.append("API endpoints discussed")
        
        # Look for framework mentions
        frameworks = re.findall(r'\b(?:React|Vue|Angular|Node\.js|Python|Java|C\+\+)\b', content)
        if frameworks:
            concepts.append(f"Frameworks: {', '.join(set(frameworks))}")
        
        return concepts
    
    def _extract_actionable_insights(self, content: str) -> List[str]:
        """Extract actionable technology insights"""
        
        insights = []
        
        # Look for implementation steps
        steps = re.findall(r'(?:step \d+|first|second|third|next|then|finally)', content, re.IGNORECASE)
        if steps:
            insights.append("Implementation steps provided")
        
        # Look for best practices
        best_practices = re.findall(r'\b(?:best practice|recommendation|should|avoid|don\'t)\b', content, re.IGNORECASE)
        if best_practices:
            insights.append("Best practices mentioned")
        
        # Look for performance tips
        performance = re.findall(r'\b(?:performance|optimization|efficiency|speed|memory)\b', content, re.IGNORECASE)
        if performance:
            insights.append("Performance considerations discussed")
        
        return insights

class AIMLContentProcessor(BaseContentProcessor):
    """Processor for AI/ML content"""
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract AI/ML-specific terms"""
        
        ai_patterns = [
            r'\b(?:machine learning|ML|artificial intelligence|AI|neural network)\b',
            r'\b(?:deep learning|reinforcement learning|supervised|unsupervised)\b',
            r'\b(?:model|training|inference|prediction|classification|regression)\b',
            r'\b(?:tensorflow|pytorch|keras|scikit-learn|pandas|numpy)\b'
        ]
        
        terms = []
        for pattern in ai_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key AI/ML concepts"""
        
        concepts = []
        
        # Look for algorithm mentions
        algorithms = re.findall(r'\b(?:gradient descent|backpropagation|random forest|SVM|k-means)\b', content, re.IGNORECASE)
        if algorithms:
            concepts.append(f"Algorithms: {', '.join(set(algorithms))}")
        
        # Look for model types
        model_types = re.findall(r'\b(?:CNN|RNN|LSTM|transformer|BERT|GPT)\b', content, re.IGNORECASE)
        if model_types:
            concepts.append(f"Model types: {', '.join(set(model_types))}")
        
        # Look for metrics
        metrics = re.findall(r'\b(?:accuracy|precision|recall|F1|AUC|MSE|MAE)\b', content, re.IGNORECASE)
        if metrics:
            concepts.append(f"Metrics: {', '.join(set(metrics))}")
        
        return concepts
    
    def _extract_actionable_insights(self, content: str) -> List[str]:
        """Extract actionable AI/ML insights"""
        
        insights = []
        
        # Look for implementation guidance
        implementation = re.findall(r'\b(?:implement|train|test|validate|deploy)\b', content, re.IGNORECASE)
        if implementation:
            insights.append("Implementation guidance provided")
        
        # Look for data requirements
        data_req = re.findall(r'\b(?:dataset|training data|test data|validation)\b', content, re.IGNORECASE)
        if data_req:
            insights.append("Data requirements discussed")
        
        # Look for performance benchmarks
        benchmarks = re.findall(r'\b(?:benchmark|comparison|evaluation|results)\b', content, re.IGNORECASE)
        if benchmarks:
            insights.append("Performance benchmarks mentioned")
        
        return insights

# Additional processors for other domains
class WebDevContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        web_patterns = [
            r'\b(?:HTML|CSS|JavaScript|React|Vue|Angular|Node\.js)\b',
            r'\b(?:frontend|backend|full-stack|responsive|mobile-first)\b',
            r'\b(?:REST|GraphQL|API|endpoint|HTTP|HTTPS|JSON|XML)\b'
        ]
        terms = []
        for pattern in web_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class MobileDevContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        mobile_patterns = [
            r'\b(?:iOS|Android|React Native|Flutter|Xamarin|Swift|Kotlin)\b',
            r'\b(?:mobile app|native|hybrid|cross-platform|App Store|Play Store)\b'
        ]
        terms = []
        for pattern in mobile_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class DevOpsContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        devops_patterns = [
            r'\b(?:Docker|Kubernetes|Jenkins|CI\/CD|pipeline|deployment)\b',
            r'\b(?:AWS|Azure|GCP|cloud|infrastructure|microservices)\b'
        ]
        terms = []
        for pattern in devops_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class SecurityContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        security_patterns = [
            r'\b(?:security|vulnerability|exploit|penetration testing|firewall)\b',
            r'\b(?:encryption|authentication|authorization|SSL|TLS|HTTPS)\b'
        ]
        terms = []
        for pattern in security_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class FinanceContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        finance_patterns = [
            r'\b(?:investment|portfolio|trading|stocks|bonds|mutual funds)\b',
            r'\b(?:ROI|ROE|P\/E ratio|market cap|dividend|yield)\b'
        ]
        terms = []
        for pattern in finance_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class MedicalContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        medical_patterns = [
            r'\b(?:medical|healthcare|clinical|patient|diagnosis|treatment)\b',
            r'\b(?:drug|medication|therapy|surgery|procedure|symptom)\b'
        ]
        terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class LegalContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        legal_patterns = [
            r'\b(?:legal|law|attorney|lawyer|court|judge|jury)\b',
            r'\b(?:contract|agreement|liability|damages|settlement)\b'
        ]
        terms = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class AcademicContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        academic_patterns = [
            r'\b(?:research|study|experiment|hypothesis|methodology)\b',
            r'\b(?:peer review|journal|publication|citation|reference)\b'
        ]
        terms = []
        for pattern in academic_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class BusinessContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        business_patterns = [
            r'\b(?:business|strategy|marketing|sales|customer|revenue)\b',
            r'\b(?:startup|entrepreneur|venture capital|funding|IPO)\b'
        ]
        terms = []
        for pattern in business_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        return list(set(terms))

class GeneralContentProcessor(BaseContentProcessor):
    def _extract_technical_terms(self, content: str) -> List[str]:
        # General term extraction
        return []
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        # General concept extraction
        return []
    
    def _extract_actionable_insights(self, content: str) -> List[str]:
        # General insight extraction
        return []
