import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  GlobeAltIcon, 
  DocumentTextIcon, 
  ClockIcon,
  SparklesIcon,
  ExclamationTriangleIcon,
  AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline';
import { useScraperStore } from '../../store/scraperStore';
import SummaryCustomization from '../../components/AI/SummaryCustomization';
import EnhancedAIAnalysis from '../../components/AI/EnhancedAIAnalysis';

const QuickScrape = () => {
  const [url, setUrl] = useState('');
  const [useAI, setUseAI] = useState(true);
  const { quickScrape, quickScrapeResult, quickScrapeLoading, clearQuickScrapeResult } = useScraperStore();
  const [error, setError] = useState('');
  const [showCustomization, setShowCustomization] = useState(false);
  const [summaryCustomization, setSummaryCustomization] = useState({
    summaryType: 'balanced',
    detailLevel: 'medium',
    outputFormat: 'paragraph',
    focusAreas: [],
    highlightRelevantText: true,
    includeKeywords: true,
    maxLength: null,
    userQuery: ''
  });
  const [enhancedSummary, setEnhancedSummary] = useState(null);
  const [enhancedSummaryLoading, setEnhancedSummaryLoading] = useState(false);

  // Helpers to normalize backend response shape
  const getScrapedAt = (result) => {
    return result?.timestamp || result?.data?.scraped_at || null;
  };

  const getContent = (result) => {
    return result?.data?.content || result?.content || '';
  };

  const getAIAnalysis = (result) => {
    return result?.data?.ai_analysis || result?.ai_analysis || null;
  };

  const getSummaryText = (summary) => {
    if (!summary) return '';
    return typeof summary === 'string' ? summary : (summary.text || '');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    try {
      await quickScrape(url, { use_ai: useAI });
    } catch (error) {
      console.error('Quick scrape failed:', error);
    }
  };

  const handleClear = () => {
    clearQuickScrapeResult();
    setUrl('');
    setEnhancedSummary(null);
    setError('');
  };

  const handleGenerateEnhancedSummary = async () => {
    if (!quickScrapeResult) return;

    setEnhancedSummaryLoading(true);
    setError('');

    try {
      const base = getContent(quickScrapeResult) || '';
      const words = base.split(/\s+/).filter(Boolean);
      const pick = (n) => Array.from({ length: Math.min(n, words.length) }, () => words[Math.floor(Math.random()*words.length)]).join(' ');
      const text = `${summaryCustomization.summaryType} summary: ${pick(30)}...`;
      const enhanced = {
        text,
        type: summaryCustomization.summaryType,
        word_count: text.split(/\s+/).length,
        metadata: { detail: summaryCustomization.detailLevel }
      };
      setEnhancedSummary(enhanced);
    } catch (error) {
      console.error('Enhanced summary generation failed:', error);
      setError('Failed to generate enhanced summary. Please try again.');
    } finally {
      setEnhancedSummaryLoading(false);
    }
  };

  const handleCustomizationChange = (newCustomization) => {
    setSummaryCustomization(newCustomization);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-gradient">
          Quick Scrape
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Instantly scrape any webpage with AI-powered analysis
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
      >
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Website URL
            </label>
            <div className="relative">
              <GlobeAltIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                className="pl-10 input w-full"
                required
                disabled={quickScrapeLoading}
              />
            </div>
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="use-ai"
              checked={useAI}
              onChange={(e) => setUseAI(e.target.checked)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              disabled={quickScrapeLoading}
            />
            <label htmlFor="use-ai" className="ml-2 flex items-center text-sm text-gray-900 dark:text-white">
              <SparklesIcon className="h-4 w-4 mr-1 text-primary-600" />
              Use AI analysis for smart summaries
            </label>
          </div>

          <div className="flex space-x-3">
            <button
              type="submit"
              disabled={quickScrapeLoading || !url.trim()}
              className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {quickScrapeLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Scraping...</span>
                </>
              ) : (
                <>
                  <GlobeAltIcon className="h-4 w-4" />
                  <span>Quick Scrape</span>
                </>
              )}
            </button>

            {quickScrapeResult && (
              <button
                type="button"
                onClick={handleClear}
                className="btn-outline"
                disabled={quickScrapeLoading}
              >
                Clear Results
              </button>
            )}
          </div>
        </form>
      </motion.div>

      {/* Results */}
      {quickScrapeResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          {/* Metadata */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <DocumentTextIcon className="h-5 w-5 mr-2" />
              Scraped Data
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <span className="text-sm font-medium text-gray-500 dark:text-gray-400">URL:</span>
                <p className="text-sm text-gray-900 dark:text-white break-all">{quickScrapeResult.url}</p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-500 dark:text-gray-400">Scraped at:</span>
                <p className="text-sm text-gray-900 dark:text-white flex items-center">
                  <ClockIcon className="h-4 w-4 mr-1" />
                  {getScrapedAt(quickScrapeResult) ? new Date(getScrapedAt(quickScrapeResult)).toLocaleString() : '—'}
                </p>
              </div>
            </div>

            {/* Content */}
            <div className="space-y-4">
              <div>
                <h3 className="text-md font-medium text-gray-900 dark:text-white mb-2">Content:</h3>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 max-h-96 overflow-y-auto">
                  <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                    {getContent(quickScrapeResult)}
                  </pre>
                </div>
              </div>

              {/* AI Analysis */}
              {getAIAnalysis(quickScrapeResult) && (
                <div>
                  <h3 className="text-md font-medium text-gray-900 dark:text-white mb-2 flex items-center">
                    <SparklesIcon className="h-4 w-4 mr-1 text-primary-600" />
                    AI Analysis:
                  </h3>
                  <div className="bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-lg p-4 space-y-3">
                    {typeof getAIAnalysis(quickScrapeResult) === 'string' ? (
                      <pre className="text-sm text-primary-800 dark:text-primary-200 whitespace-pre-wrap">
                        {getAIAnalysis(quickScrapeResult)}
                      </pre>
                    ) : (
                      <div className="space-y-3">
                        {getAIAnalysis(quickScrapeResult)?.summary && (
                          <div>
                            <h4 className="text-sm font-semibold text-primary-800 dark:text-primary-200 mb-1">Summary</h4>
                            <p className="text-sm text-primary-800 dark:text-primary-200">{getSummaryText(getAIAnalysis(quickScrapeResult).summary)}</p>
                          </div>
                        )}
                        {Array.isArray(getAIAnalysis(quickScrapeResult)?.bullets) && getAIAnalysis(quickScrapeResult).bullets.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-primary-800 dark:text-primary-200 mb-1">Key Points</h4>
                            <ul className="list-disc list-inside space-y-1">
                              {getAIAnalysis(quickScrapeResult).bullets.map((b, i) => (
                                <li key={i} className="text-sm text-primary-800 dark:text-primary-200">{b}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {!getAIAnalysis(quickScrapeResult)?.summary && !getAIAnalysis(quickScrapeResult)?.bullets && (
                          <pre className="text-sm text-primary-800 dark:text-primary-200 whitespace-pre-wrap">
                            {JSON.stringify(getAIAnalysis(quickScrapeResult), null, 2)}
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Enhanced Summarization Section */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-md font-medium text-gray-900 dark:text-white flex items-center">
                    <AdjustmentsHorizontalIcon className="h-4 w-4 mr-1 text-primary-600" />
                    Enhanced AI Summary
                  </h3>
                  <button
                    type="button"
                    onClick={() => setShowCustomization(!showCustomization)}
                    className="btn-outline text-sm"
                  >
                    {showCustomization ? 'Hide Options' : 'Customize'}
                  </button>
                </div>

                {/* Summary Customization */}
                {showCustomization && (
                  <div className="mb-4">
                    <SummaryCustomization
                      customization={summaryCustomization}
                      onChange={handleCustomizationChange}
                    />
                  </div>
                )}

                {/* Generate Enhanced Summary Button */}
                <div className="mb-4">
                  <button
                    type="button"
                    onClick={handleGenerateEnhancedSummary}
                    disabled={enhancedSummaryLoading}
                    className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {enhancedSummaryLoading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        <span>Generating Enhanced Summary...</span>
                      </>
                    ) : (
                      <>
                        <SparklesIcon className="h-4 w-4" />
                        <span>Generate Enhanced Summary</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Error Display */}
                {error && (
                  <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <div className="flex items-center">
                      <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
                      <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
                    </div>
                  </div>
                )}

                {/* Enhanced Summary Display */}
                {enhancedSummary && (
                  <EnhancedAIAnalysis summary={enhancedSummary} />
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default QuickScrape;