import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowDownTrayIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  SparklesIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline';
import { useScraperStore } from '../../store/scraperStore';
import SummaryCustomization from '../../components/AI/SummaryCustomization';
import EnhancedAIAnalysis from '../../components/AI/EnhancedAIAnalysis';

const JobDetail = () => {
  const { id } = useParams();
  const { currentJob, scrapedData, dataLoading, fetchJob, fetchJobData, exportData } = useScraperStore();
  const [activeTab, setActiveTab] = useState('data');
  const [enhancedSummaries, setEnhancedSummaries] = useState({});
  const [loadingEnhanced, setLoadingEnhanced] = useState({});
  const [showCustomization, setShowCustomization] = useState({});
  const [summaryCustomization, setSummaryCustomization] = useState({
    summaryType: 'balanced',
    detailLevel: 'medium',
    outputFormat: 'mixed',
    focusAreas: [],
    highlightRelevantText: true,
    includeKeywords: true,
    maxLength: null,
    userQuery: ''
  });

  useEffect(() => {
    if (id) {
      fetchJob(parseInt(id));
      fetchJobData(parseInt(id));
    }
  }, [id, fetchJob, fetchJobData]);

  const handleExport = async (format) => {
    try {
      await exportData(parseInt(id), format);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const handleGenerateEnhancedSummary = async (itemId, content, title, url) => {
    setLoadingEnhanced(prev => ({ ...prev, [itemId]: true }));
    
    try {
      const response = await fetch('/api/scraper/enhanced-summary', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          content,
          title: title || '',
          url: url || '',
          summary_type: summaryCustomization.summaryType,
          detail_level: summaryCustomization.detailLevel,
          output_format: summaryCustomization.outputFormat,
          focus_areas: summaryCustomization.focusAreas,
          highlight_relevant_text: summaryCustomization.highlightRelevantText,
          include_keywords: summaryCustomization.includeKeywords,
          max_length: summaryCustomization.maxLength,
          user_query: summaryCustomization.userQuery
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate enhanced summary');
      }

      const result = await response.json();
      setEnhancedSummaries(prev => ({ ...prev, [itemId]: result }));
    } catch (error) {
      console.error('Enhanced summary generation failed:', error);
      // Show error message to user
      alert(`Failed to generate enhanced summary: ${error.message}`);
    } finally {
      setLoadingEnhanced(prev => ({ ...prev, [itemId]: false }));
    }
  };

  const handleCustomizationChange = (field, value) => {
    setSummaryCustomization(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const toggleCustomization = (itemId) => {
    setShowCustomization(prev => ({
      ...prev,
      [itemId]: !prev[itemId]
    }));
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-success-500" />;
      case 'running':
        return <ClockIcon className="h-5 w-5 text-warning-500" />;
      case 'failed':
        return <ExclamationTriangleIcon className="h-5 w-5 text-error-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  if (!currentJob) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner h-8 w-8"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center space-x-3">
            {getStatusIcon(currentJob.status)}
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {currentJob.name}
            </h1>
          </div>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Created on {new Date(currentJob.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-2">
          <button
            onClick={() => handleExport('csv')}
            className="btn-outline flex items-center space-x-2"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            <span>CSV</span>
          </button>
          <button
            onClick={() => handleExport('json')}
            className="btn-outline flex items-center space-x-2"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            <span>JSON</span>
          </button>
          <button
            onClick={() => handleExport('excel')}
            className="btn-outline flex items-center space-x-2"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            <span>Excel</span>
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {currentJob.total_urls}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Total URLs</div>
        </div>
        <div className="card p-6">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {currentJob.processed_urls}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Processed</div>
        </div>
        <div className="card p-6">
          <div className="text-2xl font-bold text-success-600">
            {scrapedData.filter(item => item.status === 'success').length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Successful</div>
        </div>
        <div className="card p-6">
          <div className="text-2xl font-bold text-error-600">
            {scrapedData.filter(item => item.status === 'failed').length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Failed</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="card">
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8 px-6">
            <button
              onClick={() => setActiveTab('data')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'data'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400'
              }`}
            >
              Scraped Data
            </button>
            <button
              onClick={() => setActiveTab('config')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'config'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400'
              }`}
            >
              Configuration
            </button>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'data' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              {dataLoading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="loading-spinner h-6 w-6"></div>
                </div>
              ) : scrapedData.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-gray-500 dark:text-gray-400">No data available</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {scrapedData.map((item) => (
                    <div
                      key={item.id}
                      className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2">
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-primary-600 hover:text-primary-800 font-medium"
                            >
                              {item.title || item.url}
                            </a>
                            <span className={`badge ${
                              item.status === 'success' ? 'badge-success' : 'badge-error'
                            }`}>
                              {item.status}
                            </span>
                          </div>
                          {item.content && (
                            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 line-clamp-3">
                              {item.content.substring(0, 200)}...
                            </p>
                          )}
                          {item.ai_analysis?.summary && (
                            <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100">
                                AI Summary
                              </h4>
                              <p className="mt-1 text-sm text-blue-800 dark:text-blue-200">
                                {item.ai_analysis.summary}
                              </p>
                            </div>
                          )}

                          {/* Enhanced Summarization Section */}
                          <div className="mt-4 border-t border-gray-200 dark:border-gray-700 pt-4">
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-sm font-medium text-gray-900 dark:text-white flex items-center">
                                <SparklesIcon className="h-4 w-4 mr-2 text-purple-500" />
                                Enhanced AI Analysis
                              </h4>
                              <div className="flex space-x-2">
                                <button
                                  onClick={() => toggleCustomization(item.id)}
                                  className="btn-outline text-sm flex items-center space-x-1"
                                >
                                  <AdjustmentsHorizontalIcon className="h-4 w-4" />
                                  <span>Customize</span>
                                </button>
                                <button
                                  onClick={() => handleGenerateEnhancedSummary(item.id, item.content, item.title, item.url)}
                                  disabled={loadingEnhanced[item.id] || !item.content}
                                  className="btn-primary text-sm flex items-center space-x-1"
                                >
                                  {loadingEnhanced[item.id] ? (
                                    <div className="loading-spinner h-4 w-4"></div>
                                  ) : (
                                    <SparklesIcon className="h-4 w-4" />
                                  )}
                                  <span>
                                    {loadingEnhanced[item.id] ? 'Generating...' : 'Generate Enhanced Summary'}
                                  </span>
                                </button>
                              </div>
                            </div>

                            {/* Customization Panel */}
                            {showCustomization[item.id] && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="mb-4"
                              >
                                <SummaryCustomization
                                  customization={summaryCustomization}
                                  onChange={handleCustomizationChange}
                                />
                              </motion.div>
                            )}

                            {/* Enhanced Summary Display */}
                            {enhancedSummaries[item.id] && (
                              <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mt-4"
                              >
                                <EnhancedAIAnalysis
                                  analysis={enhancedSummaries[item.id]}
                                  isLoading={loadingEnhanced[item.id]}
                                  showCustomizeButton={false}
                                />
                              </motion.div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'config' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    CSS Selector
                  </h4>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {currentJob.config?.css_selector || 'Not specified'}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    Data Type
                  </h4>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {currentJob.config?.data_type || 'text'}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    Use Playwright
                  </h4>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {currentJob.config?.use_playwright ? 'Yes' : 'No'}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    Wait Time
                  </h4>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {currentJob.config?.wait_time || 3} seconds
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default JobDetail;