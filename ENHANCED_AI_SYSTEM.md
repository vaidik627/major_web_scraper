# Enhanced AI Web Scraper System

## 🚀 Overview

This enhanced AI web scraper system implements cutting-edge AI technologies to provide intelligent, domain-aware content analysis with multi-model consensus, technology-specific scraping, and continuous learning capabilities.

## ✨ Key Features

### 1. **Multi-Model Ensemble AI**
- **GPT-4o Integration**: Advanced reasoning and analysis
- **Claude Integration**: Anthropic's powerful language model
- **Local Models**: BART and SentenceTransformer for offline processing
- **Consensus Mechanism**: Combines multiple AI models for robust results
- **Fallback System**: Graceful degradation when services are unavailable

### 2. **Domain-Aware Classification**
- **Automatic Domain Detection**: Classifies content into 12+ domains
- **Technology Domains**: AI/ML, Web Dev, Mobile, DevOps, Cybersecurity, etc.
- **Business Domains**: Finance, Medical, Legal, Academic, Business
- **Specialized Processing**: Domain-specific extraction and analysis

### 3. **Technology-Specific Scraping**
- **Domain-Specific Selectors**: Optimized CSS selectors for each tech domain
- **Code Extraction**: Preserves code blocks, APIs, and technical documentation
- **Playwright Integration**: Handles JavaScript-heavy sites
- **Smart Content Detection**: Identifies technical elements automatically

### 4. **Technology Knowledge Graph**
- **Relationship Mapping**: Maps relationships between technologies
- **Learning Paths**: Finds optimal learning sequences between technologies
- **Technology Suggestions**: Recommends complementary technologies
- **Ecosystem Analysis**: Analyzes technology ecosystems and dependencies

### 5. **Real-Time Learning System**
- **User Feedback Collection**: Collects and processes user ratings and feedback
- **Adaptive Summarization**: Personalizes summaries based on user preferences
- **Learning Insights**: Generates actionable insights from feedback patterns
- **Continuous Improvement**: System learns and improves over time

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced AI Integration                  │
├─────────────────────────────────────────────────────────────┤
│  Multi-Model    │  Domain-Aware   │  Tech-Specific  │  Real-Time │
│  Ensemble       │  Classifier     │  Scraper        │  Learning  │
├─────────────────────────────────────────────────────────────┤
│  GPT-4o         │  Content       │  Playwright     │  Feedback  │
│  Claude         │  Classification│  Requests       │  Collection│
│  Local Models   │  Domain-Specific│  Domain Configs │  User      │
│  Consensus      │  Processing    │  Tech Extraction│  Preferences│
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
backend/services/
├── multi_model_ensemble.py          # Multi-model AI consensus
├── domain_aware_classifier.py       # Domain classification
├── tech_specialized_scraper.py      # Technology-specific scraping
├── tech_knowledge_graph.py          # Technology relationships
├── feedback_learning_system.py      # Learning and feedback
├── enhanced_ai_integration.py       # Main integration service
└── ai_service.py                    # Updated main AI service

backend/routers/
└── enhanced_ai.py                   # Enhanced AI API endpoints
```

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install openai anthropic sentence-transformers transformers
pip install networkx scikit-learn beautifulsoup4 playwright
pip install sqlite3 nltk textblob spacy
```

### 2. **Environment Variables**

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional
FEEDBACK_DB_PATH=feedback.db
```

### 3. **Initialize Services**

```python
from backend.services.enhanced_ai_integration import EnhancedAIIntegrationService

# Initialize the enhanced AI service
ai_service = EnhancedAIIntegrationService(
    openai_api_key="your_key",
    anthropic_api_key="your_key"
)
```

## 📊 API Endpoints

### **Enhanced Content Analysis**

```http
POST /api/enhanced-ai/analyze
Content-Type: application/json

{
  "content": "Your content here...",
  "title": "Content Title",
  "url": "https://example.com",
  "user_id": "user123",
  "max_length": 500,
  "enable_personalization": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": "Enhanced AI-generated summary...",
    "domain": "tech",
    "confidence": 0.95,
    "technical_elements": {
      "code_blocks": ["code1", "code2"],
      "api_endpoints": ["/api/endpoint1"],
      "technical_terms": ["React", "JavaScript"]
    },
    "related_technologies": [
      {
        "name": "React",
        "relationship": "uses",
        "strength": 0.8
      }
    ],
    "personalized_for_user": true,
    "metadata": {
      "processing_time": 2.5,
      "models_used": 3,
      "consensus_agreement": "high"
    }
  }
}
```

### **Technology-Specific Scraping**

```http
POST /api/enhanced-ai/scrape-tech
Content-Type: application/json

{
  "urls": ["https://reactjs.org/docs", "https://vuejs.org/guide"],
  "tech_domain": "web_dev",
  "user_id": "user123",
  "enable_personalization": true
}
```

### **User Feedback Collection**

```http
POST /api/enhanced-ai/feedback
Content-Type: application/json

{
  "user_id": "user123",
  "summary_id": "summary456",
  "rating": 5,
  "feedback_text": "Great summary, very helpful!",
  "domain": "tech",
  "technology": "React"
}
```

### **Learning Path Discovery**

```http
POST /api/enhanced-ai/learning-path
Content-Type: application/json

{
  "start_technology": "javascript",
  "target_technology": "react",
  "max_steps": 5
}
```

### **Technology Suggestions**

```http
POST /api/enhanced-ai/suggest-technologies
Content-Type: application/json

{
  "current_technologies": ["javascript", "html", "css"],
  "target_domain": "web_dev",
  "max_suggestions": 5
}
```

## 🔧 Configuration

### **Domain-Specific Scraping Config**

```python
from backend.services.tech_specialized_scraper import TechDomainScraper, TechDomain

scraper = TechDomainScraper()

# Custom configuration for AI/ML content
custom_config = scraper.create_custom_config(
    domain=TechDomain.AI_ML,
    custom_selectors={
        'content': ['.paper-content', '.research-summary'],
        'code': ['.algorithm-code', '.implementation'],
        'math': ['.equation', '.formula']
    },
    custom_keywords=['neural network', 'machine learning', 'algorithm'],
    wait_time=5
)
```

### **Knowledge Graph Customization**

```python
from backend.services.tech_knowledge_graph import TechnologyKnowledgeGraph, TechNodeType

kg = TechnologyKnowledgeGraph()

# Add custom technology
kg.add_node(
    node_id="custom_framework",
    name="Custom Framework",
    node_type=TechNodeType.FRAMEWORK,
    description="A custom web framework",
    domain="web_dev",
    popularity_score=0.7
)

# Add relationship
kg.add_relationship(
    source_id="custom_framework",
    target_id="javascript",
    relationship_type=RelationshipType.BUILT_WITH,
    strength=0.9
)
```

## 📈 Performance Metrics

### **System Health Check**

```http
GET /api/enhanced-ai/system-health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "multi_model_ensemble": {
      "openai_available": true,
      "anthropic_available": true,
      "local_models_available": true
    },
    "knowledge_graph": {
      "total_nodes": 150,
      "total_relationships": 300,
      "domains_covered": 10
    },
    "feedback_system": {
      "total_feedback": 1250,
      "total_users": 45,
      "learning_insights": 25
    }
  }
}
```

## 🎯 Use Cases

### **1. Technical Documentation Analysis**
- Automatically extracts code examples, API endpoints, and technical concepts
- Provides domain-specific summaries for developers
- Suggests related technologies and learning paths

### **2. Research Paper Summarization**
- Handles mathematical formulas and algorithm descriptions
- Extracts key research findings and methodologies
- Provides academic-level analysis and insights

### **3. Learning Path Generation**
- Maps learning sequences between technologies
- Suggests complementary skills and tools
- Personalizes recommendations based on user feedback

### **4. Content Personalization**
- Learns from user preferences and feedback
- Adapts summary style and technical depth
- Improves over time with continuous learning

## 🔄 Continuous Learning

The system continuously improves through:

1. **User Feedback**: Collects ratings and textual feedback
2. **Pattern Recognition**: Identifies successful and unsuccessful patterns
3. **Preference Learning**: Adapts to individual user preferences
4. **Domain Expertise**: Builds domain-specific knowledge over time
5. **Technology Evolution**: Updates knowledge graph with new technologies

## 🛠️ Advanced Features

### **Multi-Model Consensus**
- Runs multiple AI models in parallel
- Uses voting and similarity analysis for consensus
- Provides confidence scores and agreement levels

### **Domain-Specific Processing**
- 12+ content domains with specialized processing
- Technology-specific extraction patterns
- Domain-aware summarization strategies

### **Real-Time Adaptation**
- Learns from user interactions
- Adapts summarization style and content
- Provides personalized recommendations

### **Technology Ecosystem Analysis**
- Maps relationships between technologies
- Identifies dependencies and alternatives
- Suggests optimal technology stacks

## 📚 Examples

### **Web Development Content**

```python
# Analyze React documentation
result = await ai_service.analyze_content_with_enhanced_ai(
    content=react_docs_content,
    title="React Documentation",
    url="https://reactjs.org/docs",
    user_id="developer123",
    enable_personalization=True
)

# Result includes:
# - React-specific technical elements
# - Related technologies (JavaScript, JSX, etc.)
# - Code examples and API references
# - Personalized summary based on user preferences
```

### **AI/ML Research Paper**

```python
# Analyze machine learning paper
result = await ai_service.analyze_content_with_enhanced_ai(
    content=ml_paper_content,
    title="Deep Learning for NLP",
    url="https://arxiv.org/paper",
    user_id="researcher456"
)

# Result includes:
# - Mathematical formulas preserved
# - Algorithm descriptions extracted
# - Research methodology highlighted
# - Technical terms and concepts identified
```

## 🚀 Future Enhancements

1. **Custom Model Training**: Fine-tune models for specific domains
2. **Multi-Language Support**: Extend to non-English content
3. **Visual Content Analysis**: Process images, diagrams, and charts
4. **Real-Time Collaboration**: Multi-user feedback and learning
5. **API Integration**: Connect with external knowledge sources

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using FastAPI, OpenAI, Anthropic, and modern AI technologies**
