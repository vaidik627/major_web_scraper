import React from 'react';
import { motion } from 'framer-motion';
import {
  NewspaperIcon,
  AcademicCapIcon,
  ShoppingBagIcon,
  BriefcaseIcon,
  HeartIcon,
  TvIcon,
  CogIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';

const CategoriesChart = ({ categories = [] }) => {
  // Ensure categories is always an array
  const safeCategories = Array.isArray(categories) ? categories : [];
  
  // Category icons and colors
  const categoryConfig = {
    'News': {
      icon: NewspaperIcon,
      color: 'bg-blue-500',
      textColor: 'text-blue-600 dark:text-blue-400'
    },
    'Education': {
      icon: AcademicCapIcon,
      color: 'bg-green-500',
      textColor: 'text-green-600 dark:text-green-400'
    },
    'E-commerce': {
      icon: ShoppingBagIcon,
      color: 'bg-purple-500',
      textColor: 'text-purple-600 dark:text-purple-400'
    },
    'Business': {
      icon: BriefcaseIcon,
      color: 'bg-indigo-500',
      textColor: 'text-indigo-600 dark:text-indigo-400'
    },
    'Health': {
      icon: HeartIcon,
      color: 'bg-red-500',
      textColor: 'text-red-600 dark:text-red-400'
    },
    'Entertainment': {
      icon: TvIcon,
      color: 'bg-pink-500',
      textColor: 'text-pink-600 dark:text-pink-400'
    },
    'Technology': {
      icon: CogIcon,
      color: 'bg-gray-500',
      textColor: 'text-gray-600 dark:text-gray-400'
    }
  };

  // Sort categories by confidence score
  const sortedCategories = [...safeCategories].sort((a, b) => 
    (b.confidence_score || 0) - (a.confidence_score || 0)
  );

  if (safeCategories.length === 0) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Content Categories
        </h3>
        <div className="text-center py-8">
          <GlobeAltIcon className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            No categories identified yet
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Content Categories
      </h3>
      
      <div className="space-y-3">
        {sortedCategories.map((category, index) => {
          const config = categoryConfig[category.category] || categoryConfig['Technology'];
          const IconComponent = config.icon;
          const confidence = category.confidence_score || 0;
          
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
            >
              <div className="flex items-center">
                <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${config.color} bg-opacity-10`}>
                  <IconComponent className={`h-5 w-5 ${config.textColor}`} />
                </div>
                <div className="ml-4">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {category.category}
                  </h4>
                  {category.description && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {category.description}
                    </p>
                  )}
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {(confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Confidence
                  </div>
                </div>
                
                {/* Progress bar */}
                <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${confidence * 100}%` }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                    className={`h-2 rounded-full ${config.color}`}
                  />
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
      
      {/* Summary */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <div className="flex items-center justify-between text-sm">
          <span className="text-blue-600 dark:text-blue-400 font-medium">
            Primary Category:
          </span>
          <span className="text-blue-800 dark:text-blue-300 font-semibold">
            {sortedCategories[0]?.category || 'Unknown'}
          </span>
        </div>
        {sortedCategories[0]?.confidence_score && (
          <div className="flex items-center justify-between text-sm mt-1">
            <span className="text-blue-600 dark:text-blue-400">
              Confidence:
            </span>
            <span className="text-blue-800 dark:text-blue-300">
              {(sortedCategories[0].confidence_score * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default CategoriesChart;