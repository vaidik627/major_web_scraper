import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  SparklesIcon, 
  AdjustmentsHorizontalIcon,
  DocumentTextIcon,
  ListBulletIcon,
  ChatBubbleLeftRightIcon,
  EyeIcon,
  ClockIcon,
  TagIcon,
  CpuChipIcon,
  ChartBarIcon,
  BeakerIcon,
  PlayIcon
} from '@heroicons/react/24/outline';

const SummaryCustomization = ({ 
  isVisible, 
  onCustomize, 
  onClose, 
  isLoading = false,
  defaultOptions = {},
  content = '',
  onPreview = null
}) => {
  const [options, setOptions] = useState({
    summaryType: defaultOptions.summaryType || 'balanced',
    detailLevel: defaultOptions.detailLevel || 'medium',
    format: defaultOptions.format || 'paragraph',
    focusAreas: defaultOptions.focusAreas || [],
    highlightRelevant: defaultOptions.highlightRelevant !== false,
    includeKeywords: defaultOptions.includeKeywords !== false,
    maxLength: defaultOptions.maxLength || 200,
    aiModel: defaultOptions.aiModel || 'gpt-4o',
    tone: defaultOptions.tone || 'professional',
    creativity: defaultOptions.creativity || 0.3,
    includeMetrics: defaultOptions.includeMetrics !== false,
    userQuery: defaultOptions.userQuery || '',
    ...defaultOptions
  });

  const [activeTab, setActiveTab] = useState('basic');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewResult, setPreviewResult] = useState(null);

  const summaryTypes = [
    { 
      id: 'brief', 
      label: 'Brief', 
      description: 'Quick overview with essential points only',
      icon: ClockIcon,
      color: 'text-blue-600'
    },
    { 
      id: 'balanced', 
      label: 'Balanced', 
      description: 'Comprehensive yet concise summary',
      icon: DocumentTextIcon,
      color: 'text-green-600'
    },
    { 
      id: 'detailed', 
      label: 'Detailed', 
      description: 'In-depth analysis with comprehensive coverage',
      icon: EyeIcon,
      color: 'text-purple-600'
    },
    { 
      id: 'executive', 
      label: 'Executive', 
      description: 'Business-focused summary for decision makers',
      icon: ChatBubbleLeftRightIcon,
      color: 'text-orange-600'
    }
  ];

  const detailLevels = [
    { id: 'short', label: 'Short', words: '50-100 words' },
    { id: 'medium', label: 'Medium', words: '100-200 words' },
    { id: 'long', label: 'Long', words: '200-400 words' },
    { id: 'comprehensive', label: 'Comprehensive', words: '400+ words' }
  ];

  const formats = [
    { 
      id: 'paragraph', 
      label: 'Paragraph', 
      description: 'Flowing narrative text',
      icon: DocumentTextIcon 
    },
    { 
      id: 'bullets', 
      label: 'Bullet Points', 
      description: 'Structured key points',
      icon: ListBulletIcon 
    },
    { 
      id: 'mixed', 
      label: 'Mixed', 
      description: 'Summary + bullet points',
      icon: AdjustmentsHorizontalIcon 
    }
  ];

  const focusAreaOptions = [
    { id: 'main_content', label: 'Main Content', description: 'Core article/page content' },
    { id: 'key_facts', label: 'Key Facts', description: 'Important data and statistics' },
    { id: 'actionable_items', label: 'Action Items', description: 'Tasks and recommendations' },
    { id: 'technical_details', label: 'Technical Details', description: 'Specifications and technical info' },
    { id: 'business_insights', label: 'Business Insights', description: 'Market and business implications' },
    { id: 'trends', label: 'Trends', description: 'Patterns and emerging trends' }
  ];

  const aiModels = [
    { 
      id: 'gpt-4o', 
      label: 'GPT-4o', 
      description: 'Latest OpenAI model with superior reasoning',
      icon: CpuChipIcon,
      color: 'text-green-600',
      speed: 'Fast',
      quality: 'Excellent'
    },
    { 
      id: 'gpt-4-turbo', 
      label: 'GPT-4 Turbo', 
      description: 'Balanced performance and cost',
      icon: CpuChipIcon,
      color: 'text-blue-600',
      speed: 'Medium',
      quality: 'Very Good'
    },
    { 
      id: 'claude-3-sonnet', 
      label: 'Claude 3 Sonnet', 
      description: 'Anthropic\'s advanced reasoning model',
      icon: BeakerIcon,
      color: 'text-purple-600',
      speed: 'Medium',
      quality: 'Excellent'
    },
    { 
      id: 'local', 
      label: 'Local Processing', 
      description: 'Privacy-focused local summarization',
      icon: TagIcon,
      color: 'text-gray-600',
      speed: 'Very Fast',
      quality: 'Good'
    }
  ];

  const toneOptions = [
    { id: 'professional', label: 'Professional', description: 'Formal business tone', icon: '💼' },
    { id: 'casual', label: 'Casual', description: 'Conversational and friendly', icon: '😊' },
    { id: 'academic', label: 'Academic', description: 'Scholarly and precise', icon: '🎓' },
    { id: 'technical', label: 'Technical', description: 'Detailed and specific', icon: '⚙️' },
    { id: 'executive', label: 'Executive', description: 'High-level strategic focus', icon: '📊' },
    { id: 'creative', label: 'Creative', description: 'Engaging and innovative', icon: '🎨' }
  ];

  const customizationTabs = [
    { id: 'basic', label: 'Basic', icon: DocumentTextIcon },
    { id: 'advanced', label: 'Advanced', icon: AdjustmentsHorizontalIcon },
    { id: 'ai_model', label: 'AI Model', icon: CpuChipIcon },
    { id: 'preview', label: 'Preview', icon: EyeIcon }
  ];

  const handleOptionChange = (key, value) => {
    setOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleFocusAreaToggle = (areaId) => {
    setOptions(prev => ({
      ...prev,
      focusAreas: prev.focusAreas.includes(areaId)
        ? prev.focusAreas.filter(id => id !== areaId)
        : [...prev.focusAreas, areaId]
    }));
  };

  const handleCustomize = () => {
    onCustomize(options);
  };

  const handlePreview = useCallback(async () => {
    if (!onPreview || !content) return;
    
    setPreviewLoading(true);
    try {
      const result = await onPreview(options, content);
      setPreviewResult(result);
      setActiveTab('preview');
    } catch (error) {
      console.error('Preview failed:', error);
    } finally {
      setPreviewLoading(false);
    }
  }, [onPreview, content, options]);

  // Auto-preview when options change (debounced)
  useEffect(() => {
    if (activeTab === 'preview' && content && onPreview) {
      const timeoutId = setTimeout(() => {
        handlePreview();
      }, 1000);
      return () => clearTimeout(timeoutId);
    }
  }, [options, activeTab, content, onPreview, handlePreview]);

  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <SparklesIcon className="h-6 w-6 text-primary-600" />
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Advanced AI Summary Customization
              </h2>
            </div>
            <div className="flex items-center space-x-3">
              {content && onPreview && (
                <button
                  onClick={handlePreview}
                  disabled={previewLoading}
                  className="flex items-center space-x-2 px-3 py-1.5 text-sm bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-lg hover:bg-primary-200 dark:hover:bg-primary-900/50 transition-colors"
                >
                  {previewLoading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
                  ) : (
                    <PlayIcon className="h-4 w-4" />
                  )}
                  <span>Quick Preview</span>
                </button>
              )}
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Configure AI models, tone, focus areas, and preview results in real-time.
          </p>
          
          {/* Tab Navigation */}
          <div className="mt-4 flex space-x-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
            {customizationTabs.map((tab) => {
              const IconComponent = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-white dark:bg-gray-600 text-primary-600 dark:text-primary-400 shadow-sm'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                  }`}
                >
                  <IconComponent className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="p-6">
          {/* Basic Tab */}
          {activeTab === 'basic' && (
            <div className="space-y-8">
              {/* Summary Type */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Summary Type</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {summaryTypes.map((type) => {
                    const IconComponent = type.icon;
                    return (
                      <div
                        key={type.id}
                        className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all ${
                          options.summaryType === type.id
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                        }`}
                        onClick={() => handleOptionChange('summaryType', type.id)}
                      >
                        <div className="flex items-start space-x-3">
                          <IconComponent className={`h-5 w-5 mt-0.5 ${type.color}`} />
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 dark:text-white">{type.label}</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{type.description}</p>
                          </div>
                        </div>
                        {options.summaryType === type.id && (
                          <div className="absolute top-2 right-2">
                            <div className="h-2 w-2 bg-primary-500 rounded-full"></div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Detail Level */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Detail Level</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {detailLevels.map((level) => (
                    <button
                      key={level.id}
                      onClick={() => handleOptionChange('detailLevel', level.id)}
                      className={`p-3 text-center border-2 rounded-lg transition-all ${
                        options.detailLevel === level.id
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <div className="font-medium text-sm">{level.label}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{level.words}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Format */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Output Format</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {formats.map((format) => {
                    const IconComponent = format.icon;
                    return (
                      <div
                        key={format.id}
                        className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                          options.format === format.id
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                        }`}
                        onClick={() => handleOptionChange('format', format.id)}
                      >
                        <div className="flex items-center space-x-3">
                          <IconComponent className="h-5 w-5 text-primary-600" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">{format.label}</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">{format.description}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Tone Selection */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Tone & Style</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {toneOptions.map((tone) => (
                    <button
                      key={tone.id}
                      onClick={() => handleOptionChange('tone', tone.id)}
                      className={`p-3 text-left border-2 rounded-lg transition-all ${
                        options.tone === tone.id
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{tone.icon}</span>
                        <div>
                          <div className="font-medium text-sm text-gray-900 dark:text-white">{tone.label}</div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">{tone.description}</div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* User Query */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Focus Query (Optional)</h3>
                <textarea
                  value={options.userQuery}
                  onChange={(e) => handleOptionChange('userQuery', e.target.value)}
                  placeholder="Enter specific questions or topics you want the summary to focus on..."
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white resize-none"
                  rows={3}
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  AI will prioritize content related to your query and highlight relevant sections.
                </p>
              </div>
            </div>
          )}

          {/* Advanced Tab */}
          {activeTab === 'advanced' && (
            <div className="space-y-8">
              {/* Focus Areas */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Focus Areas</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Select specific areas to emphasize in the summary
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {focusAreaOptions.map((area) => (
                    <label
                      key={area.id}
                      className={`flex items-start space-x-3 p-3 border rounded-lg cursor-pointer transition-all ${
                        options.focusAreas.includes(area.id)
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={options.focusAreas.includes(area.id)}
                        onChange={() => handleFocusAreaToggle(area.id)}
                        className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                      />
                      <div>
                        <div className="font-medium text-sm text-gray-900 dark:text-white">{area.label}</div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">{area.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Creativity Level */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Creativity Level</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Conservative</span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Creative</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={options.creativity}
                    onChange={(e) => handleOptionChange('creativity', parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  />
                  <div className="text-center">
                    <span className="text-sm font-medium text-primary-600 dark:text-primary-400">
                      {Math.round(options.creativity * 100)}% Creative
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Higher creativity allows for more interpretive and engaging summaries, while lower values focus on factual extraction.
                  </p>
                </div>
              </div>

              {/* Advanced Options */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Advanced Features</h3>
                <div className="space-y-4">
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={options.highlightRelevant}
                      onChange={(e) => handleOptionChange('highlightRelevant', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <div>
                      <span className="font-medium text-gray-900 dark:text-white">Smart Highlighting</span>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Highlight text based on relevance and user query</p>
                    </div>
                  </label>

                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={options.includeKeywords}
                      onChange={(e) => handleOptionChange('includeKeywords', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <div>
                      <span className="font-medium text-gray-900 dark:text-white">Extract Keywords</span>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Identify and display key terms and phrases</p>
                    </div>
                  </label>

                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={options.includeMetrics}
                      onChange={(e) => handleOptionChange('includeMetrics', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <div>
                      <span className="font-medium text-gray-900 dark:text-white">Quality Metrics</span>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Show confidence scores and quality indicators</p>
                    </div>
                  </label>

                  <div>
                    <label className="block text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Maximum Summary Length
                    </label>
                    <select
                      value={options.maxLength}
                      onChange={(e) => handleOptionChange('maxLength', parseInt(e.target.value))}
                      className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                    >
                      <option value={100}>100 words</option>
                      <option value={200}>200 words</option>
                      <option value={300}>300 words</option>
                      <option value={500}>500 words</option>
                      <option value={1000}>1000 words</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* AI Model Tab */}
          {activeTab === 'ai_model' && (
            <div className="space-y-8">
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">AI Model Selection</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                  Choose the AI model that best fits your needs for speed, quality, and privacy.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {aiModels.map((model) => {
                    const IconComponent = model.icon;
                    return (
                      <div
                        key={model.id}
                        className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all ${
                          options.aiModel === model.id
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                        }`}
                        onClick={() => handleOptionChange('aiModel', model.id)}
                      >
                        <div className="flex items-start space-x-3">
                          <IconComponent className={`h-6 w-6 mt-0.5 ${model.color}`} />
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 dark:text-white">{model.label}</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{model.description}</p>
                            <div className="flex items-center space-x-4 mt-2">
                              <div className="flex items-center space-x-1">
                                <ClockIcon className="h-3 w-3 text-gray-400" />
                                <span className="text-xs text-gray-500">{model.speed}</span>
                              </div>
                              <div className="flex items-center space-x-1">
                                <ChartBarIcon className="h-3 w-3 text-gray-400" />
                                <span className="text-xs text-gray-500">{model.quality}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                        {options.aiModel === model.id && (
                          <div className="absolute top-2 right-2">
                            <div className="h-2 w-2 bg-primary-500 rounded-full"></div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Model-specific settings */}
              {(options.aiModel === 'gpt-4o' || options.aiModel === 'gpt-4-turbo') && (
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">OpenAI Settings</h4>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    Using OpenAI's advanced language models for high-quality summarization with superior reasoning capabilities.
                  </p>
                </div>
              )}

              {options.aiModel === 'claude-3-sonnet' && (
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-2">Anthropic Claude Settings</h4>
                  <p className="text-sm text-purple-800 dark:text-purple-200">
                    Using Anthropic's Claude for thoughtful analysis with strong reasoning and safety features.
                  </p>
                </div>
              )}

              {options.aiModel === 'local' && (
                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Local Processing</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    Processing content locally for maximum privacy. No data is sent to external services.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Preview Tab */}
          {activeTab === 'preview' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">Live Preview</h3>
                <button
                  onClick={handlePreview}
                  disabled={previewLoading || !content}
                  className="flex items-center space-x-2 px-3 py-1.5 text-sm bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {previewLoading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <PlayIcon className="h-4 w-4" />
                  )}
                  <span>Refresh Preview</span>
                </button>
              </div>

              {!content && (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <EyeIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No content available for preview</p>
                </div>
              )}

              {content && !previewResult && !previewLoading && (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <PlayIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>Click "Refresh Preview" to see how your summary will look</p>
                </div>
              )}

              {previewLoading && (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-3"></div>
                  <p className="text-gray-600 dark:text-gray-400">Generating preview...</p>
                </div>
              )}

              {previewResult && (
                <div className="space-y-4">
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Preview Summary</h4>
                    <div className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                      {typeof previewResult === 'string' ? (
                        <p>{previewResult}</p>
                      ) : (
                        <div>
                          {previewResult.summary && <p>{previewResult.summary}</p>}
                          {previewResult.key_points && (
                            <div className="mt-3">
                              <h5 className="font-medium mb-2">Key Points:</h5>
                              <ul className="list-disc list-inside space-y-1">
                                {previewResult.key_points.map((point, index) => (
                                  <li key={index}>{point}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {previewResult.confidence_score && (
                    <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <ChartBarIcon className="h-4 w-4 text-green-600" />
                        <span className="text-sm font-medium text-green-800 dark:text-green-200">
                          Confidence: {Math.round(previewResult.confidence_score * 100)}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            onClick={handleCustomize}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <SparklesIcon className="h-4 w-4" />
                <span>Generate Custom Summary</span>
              </>
            )}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default SummaryCustomization;