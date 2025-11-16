import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


async def scrape_and_analyze_url(url: str) -> Tuple[str, Dict[str, Any]]:
    from services.scraper_service import ScraperService
    from services.ai_service import AIService

    scraper = ScraperService()
    ai_service = AIService(db=None)

    # Robust scraping config aiming to handle images/links and dynamic pages
    scraping_config = {
        "use_playwright": False,  # start with requests; fallback to playwright if needed
        "wait_time": 3,
        "extract_links": True,
        "extract_images": True,
        "data_type": "text",
    }

    start = time.time()
    result: Dict[str, Any] = {
        "url": url,
        "scrape_success": False,
        "ai_summary_success": False,
        "enhanced_summary_success": False,
        "error": None,
        "timings": {},
    }

    # Step 1: Scrape (requests first)
    try:
        scraped = await scraper.scrape_url(url, scraping_config)
        if scraped.get("error"):
            raise RuntimeError(scraped["error"]) 
        result["scrape_success"] = True
        result["scraped_title"] = scraped.get("title")
        content = scraped.get("content") or ""
        # Truncate very long content to control token usage
        if len(content) > 30000:
            content = content[:30000]
        result["content_length"] = len(content)
        result["links_found"] = len(scraped.get("links", []))
        result["images_found"] = len(scraped.get("images", []))
    except Exception as e:
        # Fallback: try playwright if available
        try:
            scraping_config["use_playwright"] = True
            scraped = await scraper.scrape_url(url, scraping_config)
            if scraped.get("error"):
                raise RuntimeError(scraped["error"]) 
            result["scrape_success"] = True
            result["scraped_title"] = scraped.get("title")
            content = scraped.get("content") or ""
            if len(content) > 30000:
                content = content[:30000]
            result["content_length"] = len(content)
            result["links_found"] = len(scraped.get("links", []))
            result["images_found"] = len(scraped.get("images", []))
        except Exception as e2:
            result["error"] = f"scrape_failed: {e2}"
            result["timings"]["total_s"] = round(time.time() - start, 2)
            return url, result

    # Step 2: AI summary (standard)
    try:
        std = await ai_service.analyze_content(
            content=content,
            title=result.get("scraped_title") or "",
            url=url,
        )
        result["ai_summary_success"] = bool(std and (std.get("summary") or std.get("readable_summary") or std.get("professional_summary")))
        result["ai_summary_accuracy"] = std.get("accuracy_score") if isinstance(std, dict) else None
    except Exception as e:
        result["ai_summary_success"] = False
        # Don't fail the whole run if AI call fails
    
    # Step 3: Enhanced AI summary (content -> enhanced summarization service)
    try:
        from services.enhanced_summarization_service import (
            EnhancedSummarizationService,
            SummaryCustomization,
            SummaryType,
            DetailLevel,
            OutputFormat,
        )

        enh = EnhancedSummarizationService()
        customization = SummaryCustomization(
            summary_type=SummaryType.BALANCED,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.MIXED,
            focus_areas=["key_points", "insights", "actionable_items", "quantitative_data", "stakeholders"],
            user_query="",
        )
        enhanced_obj = enh.generate_enhanced_summary(
            content=content,
            title=result.get("scraped_title") or "",
            url=url,
            customization=customization,
        )
        # Success if non-empty text or some key_points
        result["enhanced_summary_success"] = bool(
            getattr(enhanced_obj, "text", "") or getattr(enhanced_obj, "key_points", [])
        )
    except Exception:
        result["enhanced_summary_success"] = False

    result["timings"]["total_s"] = round(time.time() - start, 2)
    return url, result


async def main() -> None:
    # 50 diverse URLs (news, docs, blogs, ecom, wiki, gov, dev, image-heavy, etc.)
    urls = [
        "https://www.bbc.com/",
        "https://www.nytimes.com/",
        "https://www.cnn.com/",
        "https://www.theguardian.com/international",
        "https://www.wikipedia.org/",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://developer.mozilla.org/en-US/",
        "https://react.dev/",
        "https://fastapi.tiangolo.com/",
        "https://docs.python.org/3/",
        "https://pytorch.org/",
        "https://tensorflow.org/",
        "https://www.nvidia.com/en-us/",
        "https://www.apple.com/",
        "https://www.microsoft.com/",
        "https://about.google/",
        "https://www.amazon.com/",
        "https://www.flipkart.com/",
        "https://store.steampowered.com/",
        "https://www.imdb.com/",
        "https://www.reuters.com/",
        "https://www.bloomberg.com/",
        "https://www.forbes.com/",
        "https://techcrunch.com/",
        "https://arxiv.org/",
        "https://paperswithcode.com/",
        "https://openai.com/",
        "https://anthropic.com/",
        "https://huggingface.co/",
        "https://kaggle.com/",
        "https://news.ycombinator.com/",
        "https://github.com/",
        "https://docs.github.com/en",
        "https://stackoverflow.com/",
        "https://www.reddit.com/",
        "https://www.instagram.com/",
        "https://www.tiktok.com/",
        "https://www.fifa.com/",
        "https://www.espncricinfo.com/",
        "https://www.espn.com/",
        "https://weather.com/",
        "https://openweathermap.org/",
        "https://www.nature.com/",
        "https://www.science.org/",
        "https://www.coursera.org/",
        "https://www.edx.org/",
        "https://www.ted.com/",
        "https://www.nasa.gov/",
        "https://www.who.int/",
        "https://www.whitehouse.gov/",
        "https://www.un.org/",
        "https://www.adobe.com/",
    ]

    # Shuffle to avoid burst to one domain type
    random.shuffle(urls)

    concurrency = int(os.getenv("BATCH_TEST_CONCURRENCY", "5"))
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(u: str):
        async with semaphore:
            return await scrape_and_analyze_url(u)

    tasks = [asyncio.create_task(guarded(u)) for u in urls]
    results: List[Tuple[str, Dict[str, Any]]] = await asyncio.gather(*tasks)

    total = len(results)
    scrape_ok = sum(1 for _, r in results if r.get("scrape_success"))
    ai_ok = sum(1 for _, r in results if r.get("ai_summary_success"))
    enh_ok = sum(1 for _, r in results if r.get("enhanced_summary_success"))

    summary = {
        "total": total,
        "scrape_success": scrape_ok,
        "ai_summary_success": ai_ok,
        "enhanced_summary_success": enh_ok,
        "scrape_success_rate": round(scrape_ok * 100.0 / total, 2),
        "ai_summary_success_rate": round(ai_ok * 100.0 / total, 2),
        "enhanced_summary_success_rate": round(enh_ok * 100.0 / total, 2),
    }

    # Print per-URL concise results
    for url, r in results:
        print(json.dumps({
            "url": url,
            "scraped": r.get("scrape_success"),
            "ai": r.get("ai_summary_success"),
            "enh": r.get("enhanced_summary_success"),
            "len": r.get("content_length"),
            "links": r.get("links_found"),
            "images": r.get("images_found"),
            "err": r.get("error"),
            "time_s": r.get("timings", {}).get("total_s"),
        }))

    print("BATCH_SUMMARY::" + json.dumps(summary))


if __name__ == "__main__":
    # Run with: python backend/scripts/batch_test.py
    asyncio.run(main())


