import asyncio
import time
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from fake_useragent import UserAgent
import validators
import os

class ScraperService:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random
        })
    
    async def scrape_url(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Main scraping method that handles both static and dynamic content"""
        start_time = time.time()
        
        try:
            # Validate URL
            if not validators.url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Choose scraping method based on config
            if config.get("use_playwright", False):
                result = await self._scrape_with_playwright(url, config)
            else:
                result = await self._scrape_with_requests(url, config)
            
            # Post-process the data
            result = self._post_process_data(result, config)
            
            # Add metadata
            result["metadata"] = {
                "url": url,
                "scraped_at": time.time(),
                "processing_time": time.time() - start_time,
                "method": "playwright" if config.get("use_playwright") else "requests",
                "config": config
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "url": url,
                "processing_time": time.time() - start_time
            }
    
    async def _scrape_with_requests(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape using requests and BeautifulSoup for static content"""
        headers = config.get("custom_headers") or {}
        headers.update({"User-Agent": self.ua.random})
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        result = {
            "title": self._extract_title(soup),
            "content": self._extract_content(soup, config),
            "extracted_data": {},
            "links": [],
            "images": []
        }
        
        # Extract specific data based on selectors
        if config.get("css_selector"):
            result["extracted_data"]["css_selector_data"] = self._extract_by_css(soup, config["css_selector"])
        
        if config.get("xpath"):
            # Convert XPath to CSS selector (basic conversion)
            css_selector = self._xpath_to_css(config["xpath"])
            if css_selector:
                result["extracted_data"]["xpath_data"] = self._extract_by_css(soup, css_selector)
        
        # Extract links if requested
        if config.get("extract_links", False):
            result["links"] = self._extract_links(soup, url)
        
        # Extract images if requested
        if config.get("extract_images", False):
            result["images"] = self._extract_images(soup, url)
        
        # Extract data by type
        data_type = config.get("data_type", "text")
        if data_type == "prices":
            result["extracted_data"]["prices"] = self._extract_prices(soup)
        elif data_type == "emails":
            result["extracted_data"]["emails"] = self._extract_emails(soup)
        elif data_type == "phones":
            result["extracted_data"]["phones"] = self._extract_phones(soup)

        # Optional targeted extractions regardless of data_type
        if config.get("extract_emails", False):
            result["extracted_data"]["emails"] = self._extract_emails(soup)
        if config.get("extract_phones", False):
            result["extracted_data"]["phones"] = self._extract_phones(soup)
        
        return result
    
    async def _scrape_with_playwright(self, url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape using Playwright for dynamic content"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.ua.random,
                extra_http_headers=(config.get("custom_headers") or {})
            )
            page = await context.new_page()
            
            try:
                await page.goto(url, wait_until="networkidle")
                
                # Wait for dynamic content
                wait_time = config.get("wait_time", 3)
                await page.wait_for_timeout(wait_time * 1000)
                
                # Get page content
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                result = {
                    "title": await page.title(),
                    "content": self._extract_content(soup, config),
                    "extracted_data": {},
                    "links": [],
                    "images": []
                }
                
                # Extract specific data based on selectors
                if config.get("css_selector"):
                    elements = await page.query_selector_all(config["css_selector"])
                    result["extracted_data"]["css_selector_data"] = [
                        await elem.text_content() for elem in elements
                    ]
                
                if config.get("xpath"):
                    elements = await page.query_selector_all(f"xpath={config['xpath']}")
                    result["extracted_data"]["xpath_data"] = [
                        await elem.text_content() for elem in elements
                    ]
                
                # Extract links if requested
                if config.get("extract_links", False):
                    result["links"] = self._extract_links(soup, url)
                
                # Extract images if requested
                if config.get("extract_images", False):
                    result["images"] = self._extract_images(soup, url)
                
                # Extract data by type
                data_type = config.get("data_type", "text")
                if data_type == "prices":
                    result["extracted_data"]["prices"] = self._extract_prices(soup)
                elif data_type == "emails":
                    result["extracted_data"]["emails"] = self._extract_emails(soup)
                elif data_type == "phones":
                    result["extracted_data"]["phones"] = self._extract_phones(soup)
                
                return result
                
            finally:
                await browser.close()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def _extract_content(self, soup: BeautifulSoup, config: Dict[str, Any]) -> str:
        """Extract main content from the page.
        Priority order:
        1) Explicit selectors provided in config (css_selector / xpath converted to css)
        2) Common content containers
        3) Fallback to full body text
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # 1) Honor explicit selector as primary content if provided
        selector = None
        if config.get("css_selector"):
            selector = config.get("css_selector")
        elif config.get("xpath"):
            selector = self._xpath_to_css(config.get("xpath"))

        content = ""
        if selector:
            try:
                selected = soup.select(selector)
                if selected:
                    # Join text from all matched nodes
                    content = " ".join([
                        el.get_text(separator=' ', strip=True) for el in selected
                    ]).strip()
            except Exception:
                # Fall through to default heuristics
                content = ""

        # 2) Try common content containers if explicit selector did not yield content
        if not content:
            content_selectors = [
                'main', 'article', '.content', '#content',
                '.post', '.entry', '.article-body'
            ]
            for sel in content_selectors:
                element = soup.select_one(sel)
                if element:
                    content = element.get_text(separator=' ', strip=True)
                    break

        # 3) Fallback to body content
        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)

        # Filter by keywords if provided
        keywords = config.get("keywords", [])
        if keywords and content:
            content = self._filter_by_keywords(content, keywords)

        return content
    
    def _extract_by_css(self, soup: BeautifulSoup, selector: str) -> List[str]:
        """Extract data using CSS selector"""
        elements = soup.select(selector)
        return [elem.get_text(strip=True) for elem in elements]
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            links.append({
                "text": link.get_text(strip=True),
                "url": absolute_url,
                "title": link.get('title', '')
            })
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all images from the page"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(base_url, src)
            images.append({
                "src": absolute_url,
                "alt": img.get('alt', ''),
                "title": img.get('title', '')
            })
        return images
    
    def _extract_prices(self, soup: BeautifulSoup) -> List[str]:
        """Extract price information"""
        price_patterns = [
            r'\$[\d,]+\.?\d*',
            r'€[\d,]+\.?\d*',
            r'£[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(?:USD|EUR|GBP)',
        ]
        
        text = soup.get_text()
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            prices.extend(matches)
        
        return list(set(prices))
    
    def _extract_emails(self, soup: BeautifulSoup) -> List[str]:
        """Extract email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = soup.get_text()
        emails = re.findall(email_pattern, text)
        return list(set(emails))
    
    def _extract_phones(self, soup: BeautifulSoup) -> List[str]:
        """Extract phone numbers"""
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}'
        ]
        
        text = soup.get_text()
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        return list(set(phones))
    
    def _filter_by_keywords(self, content: str, keywords: List[str]) -> str:
        """Filter content by keywords"""
        if not keywords:
            return content
        
        sentences = content.split('.')
        filtered_sentences = []
        
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                filtered_sentences.append(sentence.strip())
        
        return '. '.join(filtered_sentences)
    
    def _xpath_to_css(self, xpath: str) -> Optional[str]:
        """Basic XPath to CSS selector conversion"""
        # This is a simplified conversion for common cases
        xpath = xpath.strip()
        
        # Simple tag conversion
        if xpath.startswith('//'):
            xpath = xpath[2:]
        
        # Convert basic XPath patterns to CSS
        conversions = {
            r'div\[@class="([^"]+)"\]': r'div.\1',
            r'span\[@id="([^"]+)"\]': r'span#\1',
            r'\[@class="([^"]+)"\]': r'.\1',
            r'\[@id="([^"]+)"\]': r'#\1',
        }
        
        css_selector = xpath
        for pattern, replacement in conversions.items():
            css_selector = re.sub(pattern, replacement, css_selector)
        
        return css_selector if css_selector != xpath else None
    
    def _post_process_data(self, result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extracted data"""
        # Clean up text content
        if result.get("content"):
            result["content"] = self._clean_text(result["content"])
        
        # Remove duplicates from extracted data
        for key, value in result.get("extracted_data", {}).items():
            if isinstance(value, list):
                result["extracted_data"][key] = list(set(value))
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()