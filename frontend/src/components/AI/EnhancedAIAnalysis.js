import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  SparklesIcon, 
  AdjustmentsHorizontalIcon,
  DocumentTextIcon,
  ListBulletIcon,
  TagIcon,
  EyeIcon,
  ClipboardDocumentIcon,
  CheckIcon
} from '@heroicons/react/24/outline';

const EnhancedAIAnalysis = ({ 
  analysis, 
  onCustomize, 
  isLoading = false,
  showCustomizeButton = true 
}) => {
  const [activeTab, setActiveTab] = useState('summary');
  const [copySuccess, setCopySuccess] = useState(false);

  if (!analysis) return null;

  const tabs = [
    { id: 'summary', label: 'Summary', icon: DocumentTextIcon },
    { id: 'highlights', label: 'Highlights', icon: EyeIcon },
    { id: 'keywords', label: 'Keywords', icon: TagIcon },
    { id: 'insights', label: 'Insights', icon: SparklesIcon }
  ];

  const renderHighlightedText = (text, highlights = []) => {
    if (!highlights || highlights.length === 0) {
      return <span>{text}</span>;
    }

    let highlightedText = text;
    highlights.forEach((highlight, index) => {
      const regex = new RegExp(`(${highlight.text})`, 'gi');
      const relevance = highlight.relevance || 'medium';
      const userQueryRelevance = highlight.user_query_relevance || '';
      
      // Enhanced highlighting with different styles based on relevance and user query connection
      let highlightClass = '';
      if (userQueryRelevance && userQueryRelevance.toLowerCase().includes('direct')) {
        highlightClass = 'bg-red-200 dark:bg-red-800 border-b-2 border-red-400 font-semibold';
      } else if (relevance === 'high') {
        highlightClass = 'bg-yellow-200 dark:bg-yellow-800 border-b border-yellow-400';
      } else if (relevance === 'medium') {
        highlightClass = 'bg-blue-200 dark:bg-blue-800';
      } else {
        highlightClass = 'bg-gray-200 dark:bg-gray-700';
      }
      
      highlightedText = highlightedText.replace(
        regex, 
        `<mark class="${highlightClass} px-1 rounded" data-relevance="${relevance}" data-user-query="${userQueryRelevance}" title="Relevance: ${relevance}${userQueryRelevance ? ' | ' + userQueryRelevance : ''}">$1</mark>`
      );
    });

    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />;
  };

  const renderSummary = () => {
    const summary = analysis.summary || analysis.enhanced_summary;
    
    if (!summary) return null;

    // Handle both string and object summaries
    if (typeof summary === 'string') {
      return (
        <div className="space-y-3">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            {renderHighlightedText(summary, analysis.highlights)}
          </p>
          {/* Show accuracy score if available */}
          {analysis.accuracy_score && (
            <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex items-center text-sm text-green-700 dark:text-green-300">
                <SparklesIcon className="h-4 w-4 mr-1" />
                Accuracy Score: {Math.round(analysis.accuracy_score * 100)}%
              </div>
            </div>
          )}
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {/* Executive Summary */}
        {summary.executive_summary && (
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
            <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center">
              <SparklesIcon className="h-4 w-4 mr-1" />
              Executive Summary
            </h4>
            <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
              {renderHighlightedText(summary.executive_summary, analysis.highlights)}
            </p>
          </div>
        )}

        {/* Main Summary */}
        {summary.text && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Professional Summary</h4>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              {renderHighlightedText(summary.text, analysis.highlights)}
            </p>
            {/* Show accuracy notes if available */}
            {summary.accuracy_notes && (
              <div className="mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-sm text-blue-700 dark:text-blue-300">
                <strong>Accuracy Notes:</strong> {summary.accuracy_notes}
              </div>
            )}
          </div>
        )}

        {/* Professional Key Points with business relevance */}
        {summary.key_points && summary.key_points.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Strategic Key Points</h4>
            <ul className="space-y-3">
              {summary.key_points.map((point, index) => {
                const pointText = typeof point === 'string' ? point : point.point;
                const importance = typeof point === 'object' ? point.importance : 'medium';
                const category = typeof point === 'object' ? point.category : 'main';
                const businessRelevance = typeof point === 'object' ? point.business_relevance : '';
                
                return (
                  <li key={index} className="flex items-start space-x-3">
                    <div className={`h-2 w-2 rounded-full mt-2 flex-shrink-0 ${
                      importance === 'high' ? 'bg-red-500' : 
                      importance === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                    }`}></div>
                    <div className="flex-1">
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {renderHighlightedText(pointText, analysis.highlights)}
                      </span>
                      <div className="flex items-center mt-1 space-x-2">
                        {category && category !== 'main' && (
                          <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                            {category}
                          </span>
                        )}
                        {businessRelevance && (
                          <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/20 px-2 py-1 rounded">
                            Business Impact
                          </span>
                        )}
                      </div>
                      {businessRelevance && (
                        <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 italic">
                          {businessRelevance}
                        </div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {/* Professional Insights */}
        {analysis.insights && analysis.insights.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Strategic Insights</h4>
            <ul className="space-y-3">
              {analysis.insights.map((insight, index) => {
                const insightText = typeof insight === 'string' ? insight : insight.insight;
                const type = typeof insight === 'object' ? insight.type : 'observation';
                const confidence = typeof insight === 'object' ? insight.confidence : 0.8;
                const businessImpact = typeof insight === 'object' ? insight.business_impact : '';
                
                return (
                  <li key={index} className="flex items-start space-x-3">
                    <SparklesIcon className="h-4 w-4 text-purple-500 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {renderHighlightedText(insightText, analysis.highlights)}
                      </span>
                      <div className="flex items-center mt-1 space-x-2">
                        <span className="text-xs text-gray-500 dark:text-gray-400 bg-purple-100 dark:bg-purple-900/20 px-2 py-1 rounded">
                          {type.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          Confidence: {Math.round(confidence * 100)}%
                        </span>
                      </div>
                      {businessImpact && (
                        <div className="mt-1 text-xs text-purple-600 dark:text-purple-400 italic">
                          <strong>Business Impact:</strong> {businessImpact}
                        </div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {/* Professional Actionable Items */}
        {analysis.actionable_items && analysis.actionable_items.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Strategic Action Items</h4>
            <ul className="space-y-3">
              {analysis.actionable_items.map((item, index) => {
                const actionText = typeof item === 'string' ? item : item.action;
                const priority = typeof item === 'object' ? item.priority : 'medium';
                const context = typeof item === 'object' ? item.context : '';
                const stakeholder = typeof item === 'object' ? item.stakeholder : '';
                const timeline = typeof item === 'object' ? item.timeline : '';
                
                return (
                  <li key={index} className="flex items-start space-x-3">
                    <div className={`h-2 w-2 rounded-full mt-2 flex-shrink-0 ${
                      priority === 'high' ? 'bg-red-500' : 
                      priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                    }`}></div>
                    <div className="flex-1">
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {renderHighlightedText(actionText, analysis.highlights)}
                      </span>
                      <div className="flex items-center mt-1 space-x-2">
                        <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                          {priority} priority
                        </span>
                        {stakeholder && (
                          <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/20 px-2 py-1 rounded">
                            {stakeholder}
                          </span>
                        )}
                        {timeline && (
                          <span className="text-xs text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/20 px-2 py-1 rounded">
                            {timeline}
                          </span>
                        )}
                      </div>
                      {context && (
                        <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 italic">
                          {context}
                        </div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {/* Quantitative Data */}
        {analysis.quantitative_data && analysis.quantitative_data.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Key Metrics & Data</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {analysis.quantitative_data.map((data, index) => {
                const metric = typeof data === 'string' ? data : data.metric;
                const context = typeof data === 'object' ? data.context : '';
                const significance = typeof data === 'object' ? data.significance : '';
                
                return (
                  <div key={index} className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {renderHighlightedText(metric, analysis.highlights)}
                    </div>
                    {context && (
                      <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        {context}
                      </div>
                    )}
                    {significance && (
                      <div className="text-xs text-blue-600 dark:text-blue-400 mt-1 italic">
                        {significance}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Stakeholders */}
        {analysis.stakeholders && analysis.stakeholders.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Key Stakeholders</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {analysis.stakeholders.map((stakeholder, index) => {
                const entity = typeof stakeholder === 'string' ? stakeholder : stakeholder.entity;
                const role = typeof stakeholder === 'object' ? stakeholder.role : '';
                const importance = typeof stakeholder === 'object' ? stakeholder.importance : '';
                const connectionToQuery = typeof stakeholder === 'object' ? stakeholder.connection_to_query : '';
                
                return (
                  <div key={index} className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                    <div className="text-sm font-medium text-blue-900 dark:text-blue-100">
                      {renderHighlightedText(entity, analysis.highlights)}
                    </div>
                    {role && (
                      <div className="text-xs text-blue-700 dark:text-blue-300 mt-1">
                        <strong>Role:</strong> {role}
                      </div>
                    )}
                    {importance && (
                      <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                        {importance}
                      </div>
                    )}
                    {connectionToQuery && (
                      <div className="text-xs text-blue-500 dark:text-blue-300 mt-1 italic">
                        <strong>Relevance:</strong> {connectionToQuery}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Business Impact */}
        {analysis.business_impact && (
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <h4 className="text-sm font-semibold text-green-900 dark:text-green-100 mb-2">Business Impact Assessment</h4>
            <p className="text-sm text-green-800 dark:text-green-200">
              {renderHighlightedText(analysis.business_impact, analysis.highlights)}
            </p>
          </div>
        )}

        {/* Standard bullets fallback */}
        {analysis.bullets && analysis.bullets.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Additional Insights</h4>
            <ul className="space-y-2">
              {analysis.bullets.map((bullet, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <ListBulletIcon className="h-4 w-4 text-primary-500 mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    {renderHighlightedText(bullet, analysis.highlights)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Enhanced metadata display */}
        {summary.metadata && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Analysis Metadata</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-gray-500 dark:text-gray-400">
              <div>
                <span className="font-medium">Type:</span> {summary.metadata.type || 'Standard'}
              </div>
              <div>
                <span className="font-medium">Length:</span> {summary.metadata.word_count || 'N/A'} words
              </div>
              <div>
                <span className="font-medium">Confidence:</span> {Math.round((summary.metadata.confidence || 0.8) * 100)}%
              </div>
              <div>
                <span className="font-medium">Accuracy:</span> {Math.round((summary.metadata.accuracy_score || 0.8) * 100)}%
              </div>
              {summary.metadata.professional_grade && (
                <div className="col-span-2">
                  <span className="font-medium text-green-600 dark:text-green-400">Professional Grade:</span> ✓ Enterprise Standard
                </div>
              )}
              {summary.metadata.business_relevance && (
                <div>
                  <span className="font-medium">Business Relevance:</span> {Math.round(summary.metadata.business_relevance * 100)}%
                </div>
              )}
              {summary.metadata.user_query_coverage && (
                <div>
                  <span className="font-medium">Query Coverage:</span> {Math.round(summary.metadata.user_query_coverage * 100)}%
                </div>
              )}
            </div>
            {summary.metadata.processing_notes && (
              <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
                <span className="font-medium">Processing Notes:</span> {summary.metadata.processing_notes}
              </div>
            )}
          </div>
        )}

        {/* Overall accuracy score */}
        {analysis.accuracy_score && (
          <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center text-sm text-green-700 dark:text-green-300">
                <SparklesIcon className="h-4 w-4 mr-1" />
                Overall Accuracy Score: {Math.round(analysis.accuracy_score * 100)}%
              </div>
              <div className="text-xs text-green-600 dark:text-green-400">
                {analysis.accuracy_score > 0.9 ? 'Excellent' : 
                 analysis.accuracy_score > 0.8 ? 'Very Good' : 
                 analysis.accuracy_score > 0.7 ? 'Good' : 'Fair'}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderHighlights = () => {
    const highlights = analysis.highlights || analysis.relevant_highlights || [];
    
    if (highlights.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          <EyeIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
          <p>No highlights available for this content.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {highlights.map((highlight, index) => {
          const userQueryRelevance = highlight.user_query_relevance || '';
          const businessImpact = highlight.business_impact || '';
          const isDirectlyRelevant = userQueryRelevance && userQueryRelevance.toLowerCase().includes('direct');
          
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-4 rounded-lg border-l-4 ${
                isDirectlyRelevant
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                  : highlight.relevance === 'high' 
                  ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20' 
                  : highlight.relevance === 'medium'
                  ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                  : 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                    "{highlight.text}"
                  </p>
                  
                  {/* User Query Relevance - Most Important */}
                  {userQueryRelevance && (
                    <div className="mt-2 p-2 bg-red-100 dark:bg-red-900/30 rounded">
                      <p className="text-xs text-red-800 dark:text-red-300">
                        <strong>🎯 Directly Relevant to Your Query:</strong> {userQueryRelevance}
                      </p>
                    </div>
                  )}
                  
                  {/* Business Impact */}
                  {businessImpact && (
                    <div className="mt-2 p-2 bg-green-100 dark:bg-green-900/30 rounded">
                      <p className="text-xs text-green-800 dark:text-green-300">
                        <strong>💼 Business Impact:</strong> {businessImpact}
                      </p>
                    </div>
                  )}
                  
                  {/* Context */}
                  {highlight.context && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                      <strong>Context:</strong> {highlight.context}
                    </p>
                  )}
                  
                  {/* Reasoning */}
                  {highlight.reasoning && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      <strong>Reasoning:</strong> {highlight.reasoning}
                    </p>
                  )}
                </div>
                
                <div className="ml-4 flex flex-col items-end space-y-1">
                  {/* User Query Relevance Badge */}
                  {isDirectlyRelevant && (
                    <span className="text-xs px-2 py-1 rounded bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300 font-semibold">
                      🎯 Direct Match
                    </span>
                  )}
                  
                  {/* Relevance Badge */}
                  <span className={`text-xs px-2 py-1 rounded ${
                    highlight.relevance === 'high' 
                      ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300'
                      : highlight.relevance === 'medium'
                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                      : 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
                  }`}>
                    {highlight.relevance || 'medium'} relevance
                  </span>
                  
                  {/* Category Badge */}
                  {highlight.category && (
                    <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400 rounded">
                      {highlight.category}
                    </span>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    );
  };

  const renderKeywords = () => {
    const keywords = analysis.keywords || analysis.enhanced_keywords || [];
    
    if (keywords.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          <TagIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
          <p>No keywords extracted from this content.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {keywords.map((keyword, index) => {
            const keywordData = typeof keyword === 'string' ? { text: keyword, weight: 1 } : keyword;
            return (
              <motion.span
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  keywordData.weight > 0.7 
                    ? 'bg-primary-100 text-primary-800 dark:bg-primary-800 dark:text-primary-100' 
                    : keywordData.weight > 0.4
                    ? 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                    : 'bg-gray-50 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                }`}
              >
                {keywordData.text}
                {keywordData.weight && (
                  <span className="ml-1 text-xs opacity-75">
                    {Math.round(keywordData.weight * 100)}%
                  </span>
                )}
              </motion.span>
            );
          })}
        </div>
      </div>
    );
  };

  const renderInsights = () => {
    const insights = analysis.ai_insights || analysis.insights || [];
    const sentiment = analysis.sentiment;
    const contentType = analysis.content_type;
    
    return (
      <div className="space-y-6">
        {/* Sentiment Analysis */}
        {sentiment && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Sentiment Analysis</h4>
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Overall Sentiment</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  sentiment.label === 'positive' 
                    ? 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100'
                    : sentiment.label === 'negative'
                    ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100'
                    : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                }`}>
                  {sentiment.label}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    sentiment.label === 'positive' ? 'bg-green-500' : 
                    sentiment.label === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                  }`}
                  style={{ width: `${Math.abs(sentiment.polarity || 0) * 100}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>Confidence: {Math.round((sentiment.confidence || 0) * 100)}%</span>
                <span>Subjectivity: {Math.round((sentiment.subjectivity || 0) * 100)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Content Type */}
        {contentType && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Content Classification</h4>
            <div className="inline-flex items-center px-3 py-2 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 rounded-lg text-sm font-medium">
              <DocumentTextIcon className="h-4 w-4 mr-2" />
              {contentType}
            </div>
          </div>
        )}

        {/* AI Insights */}
        {insights.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">AI Insights</h4>
            <div className="space-y-3">
              {insights.map((insight, index) => (
                <div key={index} className="p-3 bg-gradient-to-r from-primary-50 to-purple-50 dark:from-primary-900/20 dark:to-purple-900/20 rounded-lg border border-primary-200 dark:border-primary-800">
                  <p className="text-sm text-gray-700 dark:text-gray-300">{insight}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Readability Score */}
        {analysis.readability_score && (
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Readability</h4>
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Reading Level</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {analysis.readability_score > 80 ? 'Very Easy' :
                   analysis.readability_score > 60 ? 'Easy' :
                   analysis.readability_score > 40 ? 'Moderate' :
                   analysis.readability_score > 20 ? 'Difficult' : 'Very Difficult'}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
                <div 
                  className="h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                  style={{ width: `${analysis.readability_score}%` }}
                ></div>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Score: {analysis.readability_score}/100
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'summary':
        return renderSummary();
      case 'highlights':
        return renderHighlights();
      case 'keywords':
        return renderKeywords();
      case 'insights':
        return renderInsights();
      default:
        return renderSummary();
    }
  };

  const copyToClipboard = async () => {
    const summary = analysis.summary || analysis.enhanced_summary;
    let textToCopy = '';
    
    if (typeof summary === 'string') {
      textToCopy = summary;
    } else if (summary) {
      // Build comprehensive summary text
      const parts = [];
      
      if (summary.executive_summary) {
        parts.push('EXECUTIVE SUMMARY:\n' + summary.executive_summary + '\n');
      }
      
      if (summary.text) {
        parts.push('SUMMARY:\n' + summary.text + '\n');
      }
      
      if (summary.key_points && summary.key_points.length > 0) {
        parts.push('KEY POINTS:');
        summary.key_points.forEach((point, index) => {
          const pointText = typeof point === 'string' ? point : point.point;
          parts.push(`${index + 1}. ${pointText}`);
        });
        parts.push('');
      }
      
      if (analysis.insights && analysis.insights.length > 0) {
        parts.push('INSIGHTS:');
        analysis.insights.forEach((insight, index) => {
          const insightText = typeof insight === 'string' ? insight : insight.insight;
          parts.push(`${index + 1}. ${insightText}`);
        });
        parts.push('');
      }
      
      if (analysis.actionable_items && analysis.actionable_items.length > 0) {
        parts.push('ACTION ITEMS:');
        analysis.actionable_items.forEach((item, index) => {
          const actionText = typeof item === 'string' ? item : item.action;
          parts.push(`${index + 1}. ${actionText}`);
        });
        parts.push('');
      }
      
      textToCopy = parts.join('\n');
    }
    
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = textToCopy;
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      } catch (fallbackErr) {
        console.error('Fallback copy failed: ', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

  return (
    <div className="bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-primary-200 dark:border-primary-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <SparklesIcon className="h-5 w-5 text-primary-600" />
            <h3 className="text-lg font-medium text-primary-900 dark:text-primary-100">
              AI Analysis
            </h3>
            {analysis.model && (
              <span className="text-xs text-primary-600 dark:text-primary-400 bg-primary-100 dark:bg-primary-800 px-2 py-1 rounded">
                {analysis.model}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={copyToClipboard}
              className={`p-2 transition-colors ${
                copySuccess 
                  ? 'text-green-600 hover:text-green-700 dark:text-green-400 dark:hover:text-green-300' 
                  : 'text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300'
              }`}
              title={copySuccess ? "Copied to clipboard!" : "Copy summary to clipboard"}
            >
              {copySuccess ? (
                <CheckIcon className="h-4 w-4" />
              ) : (
                <ClipboardDocumentIcon className="h-4 w-4" />
              )}
            </button>
            {showCustomizeButton && (
              <button
                onClick={onCustomize}
                className="p-2 text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
                title="Customize summary"
              >
                <AdjustmentsHorizontalIcon className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mt-4">
          {tabs.map((tab) => {
            const IconComponent = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  activeTab === tab.id
                    ? 'bg-primary-100 dark:bg-primary-800 text-primary-700 dark:text-primary-200'
                    : 'text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300'
                }`}
              >
                <IconComponent className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            <span className="ml-3 text-primary-600 dark:text-primary-400">Analyzing content...</span>
          </div>
        ) : (
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            {renderTabContent()}
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default EnhancedAIAnalysis;