import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { PlusIcon, TrashIcon, GlobeAltIcon, QuestionMarkCircleIcon, InformationCircleIcon, AdjustmentsHorizontalIcon } from '@heroicons/react/24/outline';
import { useScraperStore } from '../../store/scraperStore';
import SummaryCustomization from '../../components/AI/SummaryCustomization';

const Scraper = () => {
  const [urls, setUrls] = useState(['']);
  const [config, setConfig] = useState({
    css_selector: '',
    xpath: '',
    keywords: [],
    data_type: 'text',
    use_playwright: false,
    wait_time: 3,
    extract_images: false,
    extract_links: false,
    use_ai: true,
    use_enhanced_ai: false
  });
  const [showTooltips, setShowTooltips] = useState({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [showEnhancedOptions, setShowEnhancedOptions] = useState(false);
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

  const { createJob, quickScrapeLoading } = useScraperStore();

  const addUrl = () => {
    setUrls([...urls, '']);
  };

  const removeUrl = (index) => {
    setUrls(urls.filter((_, i) => i !== index));
  };

  const updateUrl = (index, value) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const toggleTooltip = (field) => {
    setShowTooltips(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  const validateInputs = () => {
    const validUrls = urls.filter(url => url.trim() !== '');
    
    if (validUrls.length === 0) {
      alert('⚠️ Please add at least one valid URL to scrape');
      return false;
    }

    // Validate URL format
    const urlPattern = /^https?:\/\/.+/;
    const invalidUrls = validUrls.filter(url => !urlPattern.test(url));
    if (invalidUrls.length > 0) {
      alert('⚠️ Please ensure all URLs start with http:// or https://');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateInputs()) {
      return;
    }

    const validUrls = urls.filter(url => url.trim() !== '');
    
    setIsProcessing(true);
    
    try {
      // Add realistic processing delay to show user their input is being processed
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const jobConfig = {
        ...config,
        enhanced_summarization: config.use_enhanced_ai ? summaryCustomization : null
      };

      await createJob({
        name: `Scraping Job ${new Date().toLocaleString()}`,
        urls: validUrls,
        config: jobConfig,
        use_ai: config.use_ai
      });
      
      // Show success message with delay
      setTimeout(() => {
        alert('✅ Scraping job created successfully! Check the Jobs page to monitor progress.');
        setIsProcessing(false);
      }, 1000);
      
    } catch (error) {
      setIsProcessing(false);
      alert('❌ Failed to create scraping job. Please try again.');
    }
  };

  const Tooltip = ({ field, children }) => (
    <div className="relative inline-block">
      <button
        type="button"
        onClick={() => toggleTooltip(field)}
        className="ml-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
      >
        <QuestionMarkCircleIcon className="h-4 w-4" />
      </button>
      {showTooltips[field] && (
        <div className="absolute z-10 w-64 p-3 mt-1 text-sm bg-gray-900 text-white rounded-lg shadow-lg dark:bg-gray-700">
          {children}
          <div className="absolute top-0 left-4 w-2 h-2 bg-gray-900 dark:bg-gray-700 transform rotate-45 -translate-y-1"></div>
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-gradient">
          Web Scraper
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Configure and start your web scraping job with AI-powered analysis
        </p>
      </div>

      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        onSubmit={handleSubmit}
        className="space-y-6"
      >
        {/* URLs Section */}
        <div className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
          <div className="flex items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              URLs to Scrape
            </h2>
            <Tooltip field="urls">
              <p><strong>Examples:</strong></p>
              <p>• https://example.com/products</p>
              <p>• https://news.site.com/articles</p>
              <p>• https://blog.company.com</p>
              <br />
              <p>Make sure URLs are publicly accessible and start with http:// or https://</p>
            </Tooltip>
          </div>
          <div className="space-y-3">
            {urls.map((url, index) => (
              <div key={index} className="flex items-center space-x-3">
                <GlobeAltIcon className="h-5 w-5 text-gray-400" />
                <input
                  type="url"
                  value={url}
                  onChange={(e) => updateUrl(index, e.target.value)}
                  placeholder="https://example.com/page-to-scrape"
                  className="flex-1 input"
                  required={index === 0}
                />
                {urls.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeUrl(index)}
                    className="p-2 text-red-600 hover:text-red-800"
                  >
                    <TrashIcon className="h-5 w-5" />
                  </button>
                )}
              </div>
            ))}
            <button
              type="button"
              onClick={addUrl}
              className="flex items-center space-x-2 text-primary-600 hover:text-primary-800"
            >
              <PlusIcon className="h-4 w-4" />
              <span>Add Another URL</span>
            </button>
          </div>
        </div>

        {/* Configuration Section */}
        <div className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Scraping Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex items-center">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  CSS Selector (Optional)
                </label>
                <Tooltip field="css_selector">
                  <p><strong>Target specific content:</strong></p>
                  <p>• .content (class selector)</p>
                  <p>• #main-article (ID selector)</p>
                  <p>• article p (element selector)</p>
                  <p>• .post-content, .article-body</p>
                  <br />
                  <p>Leave empty to scrape entire page content</p>
                </Tooltip>
              </div>
              <input
                type="text"
                value={config.css_selector}
                onChange={(e) => setConfig({...config, css_selector: e.target.value})}
                placeholder="e.g., .content, #main, article"
                className="mt-1 input"
              />
            </div>
            <div>
              <div className="flex items-center">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  XPath (Advanced)
                </label>
                <Tooltip field="xpath">
                  <p><strong>Advanced targeting:</strong></p>
                  <p>• //div[@class='content']</p>
                  <p>• //article//p</p>
                  <p>• //span[contains(@class,'price')]</p>
                  <br />
                  <p>Use when CSS selectors aren't sufficient</p>
                </Tooltip>
              </div>
              <input
                type="text"
                value={config.xpath}
                onChange={(e) => setConfig({...config, xpath: e.target.value})}
                placeholder="e.g., //div[@class='content']"
                className="mt-1 input"
              />
            </div>
            <div>
              <div className="flex items-center">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Content Type
                </label>
                <Tooltip field="data_type">
                  <p><strong>What to extract:</strong></p>
                  <p>• Text: Articles, descriptions</p>
                  <p>• Images: Product photos, galleries</p>
                  <p>• Links: Navigation, references</p>
                  <p>• Prices: E-commerce data</p>
                  <p>• Emails: Contact information</p>
                </Tooltip>
              </div>
              <select
                value={config.data_type}
                onChange={(e) => setConfig({...config, data_type: e.target.value})}
                className="mt-1 input"
              >
                <option value="text">📄 Text Content</option>
                <option value="images">🖼️ Images</option>
                <option value="links">🔗 Links</option>
                <option value="prices">💰 Prices</option>
                <option value="emails">📧 Email Addresses</option>
              </select>
            </div>
            <div>
              <div className="flex items-center">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Wait Time (seconds)
                </label>
                <Tooltip field="wait_time">
                  <p><strong>Page loading time:</strong></p>
                  <p>• 1-3s: Fast static sites</p>
                  <p>• 3-5s: Standard websites</p>
                  <p>• 5-10s: Heavy dynamic content</p>
                  <p>• 10+s: Very slow sites</p>
                  <br />
                  <p>Longer wait = more content loaded</p>
                </Tooltip>
              </div>
              <input
                type="number"
                value={config.wait_time}
                onChange={(e) => setConfig({...config, wait_time: parseInt(e.target.value)})}
                min="1"
                max="30"
                className="mt-1 input"
              />
            </div>
          </div>

          <div className="mt-6 space-y-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={config.use_playwright}
                onChange={(e) => setConfig({...config, use_playwright: e.target.checked})}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label className="ml-2 block text-sm text-gray-900 dark:text-white">
                🎭 Use Playwright (for JavaScript-heavy sites)
              </label>
              <Tooltip field="playwright">
                <p>Enable for sites that load content with JavaScript (SPAs, React apps, etc.)</p>
              </Tooltip>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={config.extract_images}
                onChange={(e) => setConfig({...config, extract_images: e.target.checked})}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label className="ml-2 block text-sm text-gray-900 dark:text-white">
                🖼️ Extract image URLs and metadata
              </label>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={config.extract_links}
                onChange={(e) => setConfig({...config, extract_links: e.target.checked})}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label className="ml-2 block text-sm text-gray-900 dark:text-white">
                🔗 Extract all links from the page
              </label>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={config.use_ai}
                onChange={(e) => setConfig({...config, use_ai: e.target.checked})}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label className="ml-2 block text-sm text-gray-900 dark:text-white">
                🤖 Use AI analysis for smart summaries and insights
              </label>
              <Tooltip field="ai_analysis">
                <p>AI will analyze content and provide:</p>
                <p>• Intelligent summaries</p>
                <p>• Key insights and bullet points</p>
                <p>• Sentiment analysis</p>
                <p>• Content categorization</p>
              </Tooltip>
            </div>

            {/* Enhanced AI Summarization */}
            {config.use_ai && (
              <div className="ml-6 space-y-3 border-l-2 border-primary-200 dark:border-primary-700 pl-4">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.use_enhanced_ai}
                    onChange={(e) => setConfig({...config, use_enhanced_ai: e.target.checked})}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                  />
                  <label className="ml-2 block text-sm text-gray-900 dark:text-white">
                    ✨ Enhanced AI Summarization with customization
                  </label>
                  <Tooltip field="enhanced_ai">
                    <p>Enhanced AI provides:</p>
                    <p>• Customizable summary types and detail levels</p>
                    <p>• Intelligent text highlighting</p>
                    <p>• Focus area selection</p>
                    <p>• Multiple output formats</p>
                    <p>• Advanced keyword extraction</p>
                  </Tooltip>
                </div>

                {config.use_enhanced_ai && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Summary Customization
                      </span>
                      <button
                        type="button"
                        onClick={() => setShowEnhancedOptions(!showEnhancedOptions)}
                        className="flex items-center space-x-1 text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
                      >
                        <AdjustmentsHorizontalIcon className="h-4 w-4" />
                        <span>{showEnhancedOptions ? 'Hide Options' : 'Customize'}</span>
                      </button>
                    </div>

                    {showEnhancedOptions && (
                      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                        <SummaryCustomization
                          customization={summaryCustomization}
                          onChange={setSummaryCustomization}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isProcessing || quickScrapeLoading}
            className={`btn-primary flex items-center space-x-2 ${
              isProcessing ? 'opacity-75 cursor-not-allowed' : ''
            }`}
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Processing your request...</span>
              </>
            ) : (
              <>
                <span>🚀 Start Scraping</span>
              </>
            )}
          </button>
        </div>

        {/* Processing Info */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4"
          >
            <div className="flex items-center">
              <InformationCircleIcon className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
              <div>
                <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  Processing Your Scraping Job
                </p>
                <p className="text-sm text-blue-600 dark:text-blue-300">
                  We're setting up your scraping configuration and validating the URLs. This may take a few seconds...
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </motion.form>
    </div>
  );
};

export default Scraper;