import React from 'react';
import { motion } from 'framer-motion';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChartBarIcon,
  ClockIcon,
  EyeIcon,
  HeartIcon
} from '@heroicons/react/24/outline';

const TrendsChart = ({ trends = {}, domain, days }) => {
  const trendData = trends.trends || {};
  
  // Trend type configurations
  const trendConfig = {
    sentiment: {
      icon: HeartIcon,
      label: 'Sentiment',
      color: 'text-pink-600 dark:text-pink-400',
      bgColor: 'bg-pink-100 dark:bg-pink-900'
    },
    activity: {
      icon: EyeIcon,
      label: 'Activity',
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-100 dark:bg-blue-900'
    },
    content_length: {
      icon: ChartBarIcon,
      label: 'Content Length',
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-100 dark:bg-green-900'
    },
    processing_time: {
      icon: ClockIcon,
      label: 'Processing Time',
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-100 dark:bg-purple-900'
    }
  };

  const formatTrendValue = (type, value) => {
    switch (type) {
      case 'sentiment':
        return `${(value * 100).toFixed(1)}%`;
      case 'activity':
        return `${value} items`;
      case 'content_length':
        return `${(value / 1000).toFixed(1)}k chars`;
      case 'processing_time':
        return `${value.toFixed(2)}s`;
      default:
        return value.toString();
    }
  };

  const getTrendDirection = (trend) => {
    if (!trend || !trend.change) return null;
    return trend.change > 0 ? 'up' : trend.change < 0 ? 'down' : 'stable';
  };

  const getTrendIcon = (direction) => {
    switch (direction) {
      case 'up':
        return <ArrowTrendingUpIcon className="h-4 w-4 text-green-500" />;
      case 'down':
        return <ArrowTrendingDownIcon className="h-4 w-4 text-red-500" />;
      default:
        return <div className="h-4 w-4 bg-gray-400 rounded-full" />;
    }
  };

  if (Object.keys(trendData).length === 0) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          User Trends
        </h3>
        <div className="text-center py-8">
          <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            No trend data available
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          User Trends
        </h3>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {domain && `${domain} • `}Last {days} days
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(trendData).map(([type, trend], index) => {
          const config = trendConfig[type] || trendConfig.activity;
          const IconComponent = config.icon;
          const direction = getTrendDirection(trend);
          
          return (
            <motion.div
              key={type}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${config.bgColor}`}>
                    <IconComponent className={`h-4 w-4 ${config.color}`} />
                  </div>
                  <h4 className="ml-3 text-sm font-medium text-gray-900 dark:text-white">
                    {config.label}
                  </h4>
                </div>
                {direction && getTrendIcon(direction)}
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">Current</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    {formatTrendValue(type, trend.current || 0)}
                  </span>
                </div>
                
                {trend.average !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500 dark:text-gray-400">Average</span>
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      {formatTrendValue(type, trend.average)}
                    </span>
                  </div>
                )}
                
                {trend.change !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500 dark:text-gray-400">Change</span>
                    <span className={`text-sm font-medium ${
                      trend.change > 0 ? 'text-green-600 dark:text-green-400' :
                      trend.change < 0 ? 'text-red-600 dark:text-red-400' :
                      'text-gray-600 dark:text-gray-400'
                    }`}>
                      {trend.change > 0 ? '+' : ''}{formatTrendValue(type, Math.abs(trend.change))}
                    </span>
                  </div>
                )}
              </div>
              
              {/* Simple trend visualization */}
              {trend.history && trend.history.length > 0 && (
                <div className="mt-3">
                  <div className="flex items-end space-x-1 h-8">
                    {trend.history.slice(-10).map((value, i) => {
                      const maxValue = Math.max(...trend.history);
                      const height = maxValue > 0 ? (value / maxValue) * 100 : 0;
                      
                      return (
                        <motion.div
                          key={i}
                          initial={{ height: 0 }}
                          animate={{ height: `${height}%` }}
                          transition={{ duration: 0.5, delay: index * 0.1 + i * 0.05 }}
                          className={`flex-1 ${config.bgColor} rounded-sm min-h-[2px]`}
                          title={formatTrendValue(type, value)}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
      
      {/* Summary insights */}
      {trends.insights && trends.insights.length > 0 && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
            Key Insights
          </h4>
          <ul className="space-y-1">
            {trends.insights.slice(0, 3).map((insight, index) => (
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

export default TrendsChart;