import React from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  ClockIcon,
  DocumentTextIcon,
  HeartIcon,
  StarIcon,
  TrophyIcon,
  ArrowUpIcon,
  ArrowDownIcon
} from '@heroicons/react/24/outline';

const DomainComparison = ({ comparison = {}, domains = [] }) => {
  const { metrics = {}, winner = null, insights = [] } = comparison;
  
  // Metric configurations
  const metricConfig = {
    sentiment_score: {
      icon: HeartIcon,
      label: 'Sentiment Score',
      color: 'text-pink-600 dark:text-pink-400',
      bgColor: 'bg-pink-100 dark:bg-pink-900',
      format: (value) => `${(value * 100).toFixed(1)}%`
    },
    content_quality: {
      icon: StarIcon,
      label: 'Content Quality',
      color: 'text-yellow-600 dark:text-yellow-400',
      bgColor: 'bg-yellow-100 dark:bg-yellow-900',
      format: (value) => `${(value * 100).toFixed(1)}%`
    },
    processing_time: {
      icon: ClockIcon,
      label: 'Processing Time',
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-100 dark:bg-blue-900',
      format: (value) => `${value.toFixed(2)}s`
    },
    content_length: {
      icon: DocumentTextIcon,
      label: 'Content Length',
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-100 dark:bg-green-900',
      format: (value) => `${(value / 1000).toFixed(1)}k chars`
    }
  };

  const getDomainColor = (index) => {
    const colors = [
      'bg-blue-500',
      'bg-green-500',
      'bg-purple-500',
      'bg-orange-500',
      'bg-red-500'
    ];
    return colors[index % colors.length];
  };

  const getComparisonIcon = (domain, metric) => {
    if (!winner || !winner[metric]) return null;
    
    if (winner[metric] === domain) {
      return <TrophyIcon className="h-4 w-4 text-yellow-500" />;
    }
    
    // Check if this domain is above or below average
    const values = Object.values(metrics[metric] || {});
    const average = values.reduce((sum, val) => sum + val, 0) / values.length;
    const domainValue = metrics[metric]?.[domain] || 0;
    
    if (domainValue > average) {
      return <ArrowUpIcon className="h-4 w-4 text-green-500" />;
    } else if (domainValue < average) {
      return <ArrowDownIcon className="h-4 w-4 text-red-500" />;
    }
    
    return null;
  };

  if (Object.keys(metrics).length === 0 || domains.length === 0) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Domain Comparison
        </h3>
        <div className="text-center py-8">
          <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            No comparison data available
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Domain Comparison
        </h3>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {domains.length} domains compared
        </div>
      </div>

      {/* Domain Legend */}
      <div className="flex flex-wrap gap-2 mb-6">
        {domains.map((domain, index) => (
          <div key={domain} className="flex items-center">
            <div className={`w-3 h-3 rounded-full ${getDomainColor(index)} mr-2`} />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {domain}
            </span>
            {winner && Object.values(winner).includes(domain) && (
              <TrophyIcon className="h-4 w-4 text-yellow-500 ml-1" />
            )}
          </div>
        ))}
      </div>

      {/* Metrics Comparison */}
      <div className="space-y-6">
        {Object.entries(metrics).map(([metricName, metricData], index) => {
          const config = metricConfig[metricName] || metricConfig.content_quality;
          const IconComponent = config.icon;
          const maxValue = Math.max(...Object.values(metricData));

          return (
            <motion.div
              key={metricName}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="space-y-3"
            >
              <div className="flex items-center">
                <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${config.bgColor}`}>
                  <IconComponent className={`h-4 w-4 ${config.color}`} />
                </div>
                <h4 className="ml-3 text-sm font-medium text-gray-900 dark:text-white">
                  {config.label}
                </h4>
              </div>

              <div className="space-y-2">
                {domains.map((domain, domainIndex) => {
                  const value = metricData[domain] || 0;
                  const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;

                  return (
                    <div key={domain} className="flex items-center space-x-3">
                      <div className="flex items-center min-w-0 flex-1">
                        <div className={`w-2 h-2 rounded-full ${getDomainColor(domainIndex)} mr-2 flex-shrink-0`} />
                        <span className="text-sm text-gray-600 dark:text-gray-400 truncate">
                          {domain}
                        </span>
                      </div>
                      
                      <div className="flex-1 max-w-xs">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${percentage}%` }}
                            transition={{ duration: 0.8, delay: index * 0.1 + domainIndex * 0.1 }}
                            className={`h-2 rounded-full ${getDomainColor(domainIndex)}`}
                          />
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-1 min-w-0">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {config.format(value)}
                        </span>
                        {getComparisonIcon(domain, metricName)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Winner Summary */}
      {winner && Object.keys(winner).length > 0 && (
        <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <div className="flex items-center mb-3">
            <TrophyIcon className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mr-2" />
            <h4 className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
              Performance Leaders
            </h4>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(winner).map(([metric, domain]) => {
              const config = metricConfig[metric] || metricConfig.content_quality;
              return (
                <div key={metric} className="flex items-center justify-between">
                  <span className="text-sm text-yellow-700 dark:text-yellow-400">
                    {config.label}:
                  </span>
                  <span className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
                    {domain}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Insights */}
      {insights && insights.length > 0 && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
            Key Insights
          </h4>
          <ul className="space-y-1">
            {insights.slice(0, 3).map((insight, index) => (
              <li key={index} className="text-sm text-blue-700 dark:text-blue-400">
                • {insight}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DomainComparison;