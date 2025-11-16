"""
Technology-Specific Scraper with Domain Context
Specialized scraping for different technology domains with enhanced extraction
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import requests
# Optional dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    webdriver = None
    Options = None
    By = None
    WebDriverWait = None
    expected_conditions = None

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

logger = logging.getLogger(__name__)

class TechDomain(Enum):
    AI_ML = "ai_ml"
    WEB_DEVELOPMENT = "web_dev"
    MOBILE_DEVELOPMENT = "mobile_dev"
    DEVOPS = "devops"
    CYBERSECURITY = "cybersecurity"
    DATA_SCIENCE = "data_science"
    BLOCKCHAIN = "blockchain"
    CLOUD_COMPUTING = "cloud"
    GAME_DEVELOPMENT = "game_dev"
    EMBEDDED_SYSTEMS = "embedded"

@dataclass
class TechScrapingConfig:
    domain: TechDomain
    selectors: Dict[str, List[str]]
    keywords: List[str]
    extraction_patterns: List[str]
    wait_time: int
    use_playwright: bool
    extract_code: bool
    extract_apis: bool
    extract_docs: bool

@dataclass
class TechScrapedContent:
    url: str
    title: str
    content: str
    domain: TechDomain
    technical_elements: Dict[str, Any]
    code_blocks: List[str]
    api_endpoints: List[str]
    documentation: List[str]
    metadata: Dict[str, Any]

class TechDomainScraper:
    """Technology-specific scraper with domain-aware extraction"""
    
    def __init__(self):
        self.domain_configs = self._build_domain_configs()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _build_domain_configs(self) -> Dict[TechDomain, TechScrapingConfig]:
        """Build domain-specific scraping configurations"""
        
        return {
            TechDomain.AI_ML: TechScrapingConfig(
                domain=TechDomain.AI_ML,
                selectors={
                    'content': ['.paper-content', '.research-summary', '.algorithm-description', '.methodology', '.results'],
                    'code': ['.code-block', 'pre code', '.algorithm-code', '.implementation'],
                    'math': ['.math', '.equation', '.formula', '[class*="math"]'],
                    'references': ['.references', '.bibliography', '.citations']
                },
                keywords=['machine learning', 'neural network', 'algorithm', 'model', 'training', 'inference', 'prediction'],
                extraction_patterns=['code blocks', 'mathematical formulas', 'research citations', 'algorithm descriptions'],
                wait_time=5,
                use_playwright=True,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            ),
            
            TechDomain.WEB_DEVELOPMENT: TechScrapingConfig(
                domain=TechDomain.WEB_DEVELOPMENT,
                selectors={
                    'content': ['.tutorial-content', '.guide-content', '.documentation', '.api-docs'],
                    'code': ['.code-example', 'pre code', '.snippet', '.demo-code'],
                    'apis': ['.api-endpoint', '.endpoint', '.route', '.method'],
                    'examples': ['.example', '.demo', '.sample', '.tutorial-step']
                },
                keywords=['framework', 'library', 'API', 'tutorial', 'documentation', 'frontend', 'backend'],
                extraction_patterns=['code snippets', 'API endpoints', 'configuration examples', 'tutorial steps'],
                wait_time=3,
                use_playwright=True,
                extract_code=True,
                extract_apis=True,
                extract_docs=True
            ),
            
            TechDomain.MOBILE_DEVELOPMENT: TechScrapingConfig(
                domain=TechDomain.MOBILE_DEVELOPMENT,
                selectors={
                    'content': ['.mobile-guide', '.app-tutorial', '.development-guide', '.platform-docs'],
                    'code': ['.mobile-code', '.app-code', '.platform-code', 'pre code'],
                    'ui': ['.ui-example', '.screen-shot', '.mockup', '.design'],
                    'platform': ['.ios-guide', '.android-guide', '.cross-platform']
                },
                keywords=['mobile', 'app', 'iOS', 'Android', 'React Native', 'Flutter', 'native', 'hybrid'],
                extraction_patterns=['mobile code', 'UI examples', 'platform-specific guides', 'app configurations'],
                wait_time=4,
                use_playwright=True,
                extract_code=True,
                extract_apis=True,
                extract_docs=True
            ),
            
            TechDomain.DEVOPS: TechScrapingConfig(
                domain=TechDomain.DEVOPS,
                selectors={
                    'content': ['.devops-guide', '.deployment-guide', '.infrastructure-docs', '.tutorial'],
                    'code': ['.yaml', '.dockerfile', '.config', 'pre code'],
                    'commands': ['.command', '.cli', '.terminal', '.bash'],
                    'diagrams': ['.architecture', '.diagram', '.flowchart', '.deployment']
                },
                keywords=['Docker', 'Kubernetes', 'CI/CD', 'deployment', 'infrastructure', 'cloud', 'monitoring'],
                extraction_patterns=['configuration files', 'deployment scripts', 'CLI commands', 'architecture diagrams'],
                wait_time=3,
                use_playwright=True,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            ),
            
            TechDomain.CYBERSECURITY: TechScrapingConfig(
                domain=TechDomain.CYBERSECURITY,
                selectors={
                    'content': ['.security-guide', '.vulnerability-report', '.security-analysis', '.threat-report'],
                    'code': ['.exploit-code', '.security-script', '.penetration-test', 'pre code'],
                    'vulnerabilities': ['.vulnerability', '.exploit', '.threat', '.attack'],
                    'tools': ['.security-tool', '.penetration-tool', '.analysis-tool']
                },
                keywords=['security', 'vulnerability', 'exploit', 'penetration testing', 'firewall', 'encryption'],
                extraction_patterns=['security scripts', 'vulnerability reports', 'penetration tests', 'security tools'],
                wait_time=4,
                use_playwright=True,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            ),
            
            TechDomain.DATA_SCIENCE: TechScrapingConfig(
                domain=TechDomain.DATA_SCIENCE,
                selectors={
                    'content': ['.data-analysis', '.statistical-guide', '.data-tutorial', '.research'],
                    'code': ['.python-code', '.r-code', '.jupyter', 'pre code'],
                    'visualizations': ['.chart', '.graph', '.plot', '.visualization'],
                    'datasets': ['.dataset', '.data-source', '.sample-data']
                },
                keywords=['data science', 'statistics', 'analysis', 'visualization', 'machine learning', 'pandas', 'numpy'],
                extraction_patterns=['data analysis code', 'statistical methods', 'visualizations', 'dataset descriptions'],
                wait_time=3,
                use_playwright=True,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            ),
            
            TechDomain.BLOCKCHAIN: TechScrapingConfig(
                domain=TechDomain.BLOCKCHAIN,
                selectors={
                    'content': ['.blockchain-guide', '.crypto-tutorial', '.smart-contract', '.defi-guide'],
                    'code': ['.solidity', '.smart-contract-code', '.blockchain-code', 'pre code'],
                    'contracts': ['.contract', '.transaction', '.block', '.consensus'],
                    'tokens': ['.token', '.cryptocurrency', '.coin', '.nft']
                },
                keywords=['blockchain', 'cryptocurrency', 'smart contract', 'DeFi', 'NFT', 'consensus', 'mining'],
                extraction_patterns=['smart contracts', 'blockchain transactions', 'cryptocurrency protocols', 'DeFi mechanisms'],
                wait_time=5,
                use_playwright=True,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            ),
            
            TechDomain.CLOUD_COMPUTING: TechScrapingConfig(
                domain=TechDomain.CLOUD_COMPUTING,
                selectors={
                    'content': ['.cloud-guide', '.aws-docs', '.azure-docs', '.gcp-docs'],
                    'code': ['.cloud-config', '.terraform', '.cloudformation', 'pre code'],
                    'services': ['.cloud-service', '.aws-service', '.azure-service', '.gcp-service'],
                    'architecture': ['.cloud-architecture', '.deployment', '.scaling']
                },
                keywords=['cloud', 'AWS', 'Azure', 'GCP', 'serverless', 'microservices', 'container', 'scaling'],
                extraction_patterns=['cloud configurations', 'service definitions', 'deployment scripts', 'architecture diagrams'],
                wait_time=3,
                use_playwright=True,
                extract_code=True,
                extract_apis=True,
                extract_docs=True
            ),
            
            TechDomain.GAME_DEVELOPMENT: TechScrapingConfig(
                domain=TechDomain.GAME_DEVELOPMENT,
                selectors={
                    'content': ['.game-tutorial', '.game-dev-guide', '.game-design', '.game-mechanics'],
                    'code': ['.game-code', '.unity-script', '.unreal-code', 'pre code'],
                    'assets': ['.game-asset', '.sprite', '.texture', '.model'],
                    'mechanics': ['.game-mechanic', '.gameplay', '.level-design', '.ai-behavior']
                },
                keywords=['game development', 'Unity', 'Unreal', 'game design', 'gameplay', 'graphics', 'physics'],
                extraction_patterns=['game code', 'game mechanics', 'asset descriptions', 'gameplay systems'],
                wait_time=4,
                use_playwright=True,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            ),
            
            TechDomain.EMBEDDED_SYSTEMS: TechScrapingConfig(
                domain=TechDomain.EMBEDDED_SYSTEMS,
                selectors={
                    'content': ['.embedded-guide', '.microcontroller-docs', '.iot-guide', '.hardware-guide'],
                    'code': ['.embedded-code', '.c-code', '.assembly', 'pre code'],
                    'hardware': ['.hardware-spec', '.pinout', '.schematic', '.circuit'],
                    'protocols': ['.communication-protocol', '.i2c', '.spi', '.uart']
                },
                keywords=['embedded', 'microcontroller', 'IoT', 'hardware', 'firmware', 'real-time', 'sensor'],
                extraction_patterns=['embedded code', 'hardware specifications', 'communication protocols', 'firmware descriptions'],
                wait_time=3,
                use_playwright=False,
                extract_code=True,
                extract_apis=False,
                extract_docs=True
            )
        }
    
    async def scrape_tech_content(
        self,
        urls: List[str],
        domain: TechDomain,
        custom_config: Optional[TechScrapingConfig] = None
    ) -> List[TechScrapedContent]:
        """Scrape technology-specific content with domain context"""
        
        config = custom_config or self.domain_configs.get(domain)
        if not config:
            raise ValueError(f"No configuration found for domain: {domain}")
        
        results = []
        
        for url in urls:
            try:
                if config.use_playwright:
                    content = await self._scrape_with_playwright(url, config)
                else:
                    content = await self._scrape_with_requests(url, config)
                
                if content:
                    results.append(content)
                    
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                continue
        
        return results
    
    async def _scrape_with_playwright(self, url: str, config: TechScrapingConfig) -> Optional[TechScrapedContent]:
        """Scrape content using Playwright for JavaScript-heavy sites"""
        
        if not async_playwright:
            logger.warning("Playwright not available, falling back to requests")
            return await self._scrape_with_requests(url, config)
        
        try:
            # Initialize Playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to the page
                await page.goto(url, wait_until='networkidle')
                
                # Wait for content to load
                await page.wait_for_timeout(config.wait_time * 1000)
                
                # Extract content using domain-specific selectors
                content_data = await self._extract_content_with_selectors(page, config)
                
                await browser.close()
                
                return TechScrapedContent(
                    url=url,
                    title=content_data.get('title', ''),
                    content=content_data.get('content', ''),
                    domain=config.domain,
                    technical_elements=content_data.get('technical_elements', {}),
                    code_blocks=content_data.get('code_blocks', []),
                    api_endpoints=content_data.get('api_endpoints', []),
                    documentation=content_data.get('documentation', []),
                    metadata={
                        'scraping_method': 'playwright',
                        'wait_time': config.wait_time,
                        'selectors_used': list(config.selectors.keys())
                    }
                )
                
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {e}")
            return None
    
    async def _scrape_with_requests(self, url: str, config: TechScrapingConfig) -> Optional[TechScrapedContent]:
        """Scrape content using requests for static sites"""
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content using domain-specific selectors
            content_data = self._extract_content_with_soup(soup, config)
            
            return TechScrapedContent(
                url=url,
                title=content_data.get('title', ''),
                content=content_data.get('content', ''),
                domain=config.domain,
                technical_elements=content_data.get('technical_elements', {}),
                code_blocks=content_data.get('code_blocks', []),
                api_endpoints=content_data.get('api_endpoints', []),
                documentation=content_data.get('documentation', []),
                metadata={
                    'scraping_method': 'requests',
                    'status_code': response.status_code,
                    'selectors_used': list(config.selectors.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Requests scraping failed for {url}: {e}")
            return None
    
    async def _extract_content_with_selectors(self, page, config: TechScrapingConfig) -> Dict[str, Any]:
        """Extract content using Playwright page object"""
        
        content_data = {
            'title': '',
            'content': '',
            'technical_elements': {},
            'code_blocks': [],
            'api_endpoints': [],
            'documentation': []
        }
        
        try:
            # Extract title
            title_element = await page.query_selector('title')
            if title_element:
                content_data['title'] = await title_element.text_content()
            
            # Extract main content
            content_parts = []
            for selector in config.selectors.get('content', []):
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text and text.strip():
                        content_parts.append(text.strip())
            
            content_data['content'] = '\n\n'.join(content_parts)
            
            # Extract code blocks
            if config.extract_code:
                code_blocks = []
                for selector in config.selectors.get('code', []):
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        code_text = await element.text_content()
                        if code_text and code_text.strip():
                            code_blocks.append(code_text.strip())
                content_data['code_blocks'] = code_blocks
            
            # Extract API endpoints
            if config.extract_apis:
                api_endpoints = []
                for selector in config.selectors.get('apis', []):
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        api_text = await element.text_content()
                        if api_text and api_text.strip():
                            api_endpoints.append(api_text.strip())
                content_data['api_endpoints'] = api_endpoints
            
            # Extract documentation
            if config.extract_docs:
                documentation = []
                for selector in config.selectors.get('documentation', []):
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        doc_text = await element.text_content()
                        if doc_text and doc_text.strip():
                            documentation.append(doc_text.strip())
                content_data['documentation'] = documentation
            
            # Extract domain-specific technical elements
            content_data['technical_elements'] = await self._extract_domain_specific_elements(page, config)
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
        
        return content_data
    
    def _extract_content_with_soup(self, soup: BeautifulSoup, config: TechScrapingConfig) -> Dict[str, Any]:
        """Extract content using BeautifulSoup"""
        
        content_data = {
            'title': '',
            'content': '',
            'technical_elements': {},
            'code_blocks': [],
            'api_endpoints': [],
            'documentation': []
        }
        
        try:
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                content_data['title'] = title_tag.get_text().strip()
            
            # Extract main content
            content_parts = []
            for selector in config.selectors.get('content', []):
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text:
                        content_parts.append(text)
            
            content_data['content'] = '\n\n'.join(content_parts)
            
            # Extract code blocks
            if config.extract_code:
                code_blocks = []
                for selector in config.selectors.get('code', []):
                    elements = soup.select(selector)
                    for element in elements:
                        code_text = element.get_text().strip()
                        if code_text:
                            code_blocks.append(code_text)
                content_data['code_blocks'] = code_blocks
            
            # Extract API endpoints
            if config.extract_apis:
                api_endpoints = []
                for selector in config.selectors.get('apis', []):
                    elements = soup.select(selector)
                    for element in elements:
                        api_text = element.get_text().strip()
                        if api_text:
                            api_endpoints.append(api_text)
                content_data['api_endpoints'] = api_endpoints
            
            # Extract documentation
            if config.extract_docs:
                documentation = []
                for selector in config.selectors.get('documentation', []):
                    elements = soup.select(selector)
                    for element in elements:
                        doc_text = element.get_text().strip()
                        if doc_text:
                            documentation.append(doc_text)
                content_data['documentation'] = documentation
            
            # Extract domain-specific technical elements
            content_data['technical_elements'] = self._extract_domain_specific_elements_soup(soup, config)
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
        
        return content_data
    
    async def _extract_domain_specific_elements(self, page, config: TechScrapingConfig) -> Dict[str, Any]:
        """Extract domain-specific technical elements using Playwright"""
        
        elements = {}
        
        try:
            if config.domain == TechDomain.AI_ML:
                # Extract mathematical formulas
                math_elements = await page.query_selector_all('.math, .equation, .formula')
                elements['mathematical_formulas'] = [await el.text_content() for el in math_elements]
                
                # Extract algorithm descriptions
                algo_elements = await page.query_selector_all('.algorithm, .methodology')
                elements['algorithms'] = [await el.text_content() for el in algo_elements]
                
            elif config.domain == TechDomain.WEB_DEVELOPMENT:
                # Extract framework mentions
                framework_elements = await page.query_selector_all('[class*="react"], [class*="vue"], [class*="angular"]')
                elements['frameworks'] = [await el.text_content() for el in framework_elements]
                
                # Extract API documentation
                api_elements = await page.query_selector_all('.api-doc, .endpoint-doc')
                elements['api_docs'] = [await el.text_content() for el in api_elements]
                
            elif config.domain == TechDomain.DEVOPS:
                # Extract configuration files
                config_elements = await page.query_selector_all('.yaml, .dockerfile, .config')
                elements['configurations'] = [await el.text_content() for el in config_elements]
                
                # Extract deployment scripts
                deploy_elements = await page.query_selector_all('.deployment, .deploy-script')
                elements['deployments'] = [await el.text_content() for el in deploy_elements]
                
        except Exception as e:
            logger.error(f"Domain-specific extraction failed: {e}")
        
        return elements
    
    def _extract_domain_specific_elements_soup(self, soup: BeautifulSoup, config: TechScrapingConfig) -> Dict[str, Any]:
        """Extract domain-specific technical elements using BeautifulSoup"""
        
        elements = {}
        
        try:
            if config.domain == TechDomain.AI_ML:
                # Extract mathematical formulas
                math_elements = soup.select('.math, .equation, .formula')
                elements['mathematical_formulas'] = [el.get_text().strip() for el in math_elements]
                
                # Extract algorithm descriptions
                algo_elements = soup.select('.algorithm, .methodology')
                elements['algorithms'] = [el.get_text().strip() for el in algo_elements]
                
            elif config.domain == TechDomain.WEB_DEVELOPMENT:
                # Extract framework mentions
                framework_elements = soup.select('[class*="react"], [class*="vue"], [class*="angular"]')
                elements['frameworks'] = [el.get_text().strip() for el in framework_elements]
                
                # Extract API documentation
                api_elements = soup.select('.api-doc, .endpoint-doc')
                elements['api_docs'] = [el.get_text().strip() for el in api_elements]
                
            elif config.domain == TechDomain.DEVOPS:
                # Extract configuration files
                config_elements = soup.select('.yaml, .dockerfile, .config')
                elements['configurations'] = [el.get_text().strip() for el in config_elements]
                
                # Extract deployment scripts
                deploy_elements = soup.select('.deployment, .deploy-script')
                elements['deployments'] = [el.get_text().strip() for el in deploy_elements]
                
        except Exception as e:
            logger.error(f"Domain-specific extraction failed: {e}")
        
        return elements
    
    def detect_tech_domain_from_url(self, url: str) -> Optional[TechDomain]:
        """Detect technology domain from URL patterns"""
        
        url_lower = url.lower()
        
        domain_patterns = {
            TechDomain.AI_ML: ['arxiv.org', 'paperswithcode.com', 'openai.com', 'huggingface.co', 'ai.google'],
            TechDomain.WEB_DEVELOPMENT: ['mdn.mozilla.org', 'w3schools.com', 'css-tricks.com', 'reactjs.org', 'vuejs.org'],
            TechDomain.MOBILE_DEVELOPMENT: ['developer.apple.com', 'developer.android.com', 'reactnative.dev', 'flutter.dev'],
            TechDomain.DEVOPS: ['kubernetes.io', 'docker.com', 'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com'],
            TechDomain.CYBERSECURITY: ['owasp.org', 'nist.gov', 'cve.mitre.org', 'security.google'],
            TechDomain.DATA_SCIENCE: ['kaggle.com', 'pandas.pydata.org', 'numpy.org', 'matplotlib.org'],
            TechDomain.BLOCKCHAIN: ['ethereum.org', 'bitcoin.org', 'consensys.net', 'defipulse.com'],
            TechDomain.CLOUD_COMPUTING: ['aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com', 'digitalocean.com'],
            TechDomain.GAME_DEVELOPMENT: ['unity.com', 'unrealengine.com', 'godotengine.org', 'gamedev.net'],
            TechDomain.EMBEDDED_SYSTEMS: ['arduino.cc', 'raspberrypi.org', 'microchip.com', 'st.com']
        }
        
        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                if pattern in url_lower:
                    return domain
        
        return None
    
    def create_custom_config(
        self,
        domain: TechDomain,
        custom_selectors: Dict[str, List[str]] = None,
        custom_keywords: List[str] = None,
        wait_time: int = 3
    ) -> TechScrapingConfig:
        """Create custom scraping configuration for a domain"""
        
        base_config = self.domain_configs.get(domain)
        if not base_config:
            raise ValueError(f"No base configuration found for domain: {domain}")
        
        # Merge custom selectors with base selectors
        selectors = base_config.selectors.copy()
        if custom_selectors:
            selectors.update(custom_selectors)
        
        # Merge custom keywords with base keywords
        keywords = base_config.keywords.copy()
        if custom_keywords:
            keywords.extend(custom_keywords)
        
        return TechScrapingConfig(
            domain=domain,
            selectors=selectors,
            keywords=keywords,
            extraction_patterns=base_config.extraction_patterns,
            wait_time=wait_time,
            use_playwright=base_config.use_playwright,
            extract_code=base_config.extract_code,
            extract_apis=base_config.extract_apis,
            extract_docs=base_config.extract_docs
        )
