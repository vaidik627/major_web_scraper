import React from 'react';
import { motion } from 'framer-motion';
import {
  UserIcon,
  BuildingOfficeIcon,
  MapPinIcon,
  TagIcon,
  CalendarIcon,
  CurrencyDollarIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';

const EntitiesChart = ({ entities = [] }) => {
  // Ensure entities is always an array
  const safeEntities = Array.isArray(entities) ? entities : [];
  
  // Group entities by type
  const groupedEntities = safeEntities.reduce((acc, entity) => {
    const type = entity.entity_type || 'OTHER';
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(entity);
    return acc;
  }, {});

  // Entity type icons and colors
  const entityConfig = {
    PERSON: {
      icon: UserIcon,
      color: 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20'
    },
    ORGANIZATION: {
      icon: BuildingOfficeIcon,
      color: 'bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400',
      bgColor: 'bg-green-50 dark:bg-green-900/20'
    },
    LOCATION: {
      icon: MapPinIcon,
      color: 'bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400',
      bgColor: 'bg-red-50 dark:bg-red-900/20'
    },
    DATE: {
      icon: CalendarIcon,
      color: 'bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400',
      bgColor: 'bg-purple-50 dark:bg-purple-900/20'
    },
    MONEY: {
      icon: CurrencyDollarIcon,
      color: 'bg-yellow-100 text-yellow-600 dark:bg-yellow-900 dark:text-yellow-400',
      bgColor: 'bg-yellow-50 dark:bg-yellow-900/20'
    },
    OTHER: {
      icon: TagIcon,
      color: 'bg-gray-100 text-gray-600 dark:bg-gray-900 dark:text-gray-400',
      bgColor: 'bg-gray-50 dark:bg-gray-900/20'
    }
  };

  if (safeEntities.length === 0) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Extracted Entities
        </h3>
        <div className="text-center py-8">
          <GlobeAltIcon className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            No entities extracted yet
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Extracted Entities ({safeEntities.length})
      </h3>
      
      <div className="space-y-4">
        {Object.entries(groupedEntities).map(([type, typeEntities], index) => {
          const config = entityConfig[type] || entityConfig.OTHER;
          const IconComponent = config.icon;
          
          return (
            <motion.div
              key={type}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className={`p-4 rounded-lg ${config.bgColor}`}
            >
              <div className="flex items-center mb-3">
                <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${config.color}`}>
                  <IconComponent className="h-4 w-4" />
                </div>
                <h4 className="ml-3 text-sm font-medium text-gray-900 dark:text-white">
                  {type.replace('_', ' ')} ({typeEntities.length})
                </h4>
              </div>
              
              <div className="flex flex-wrap gap-2">
                {typeEntities.slice(0, 10).map((entity, entityIndex) => (
                  <motion.span
                    key={entityIndex}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.2, delay: (index * 0.1) + (entityIndex * 0.05) }}
                    className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                    title={typeof entity.confidence_score === 'number' ? `Confidence: ${(entity.confidence_score * 100).toFixed(1)}%` : undefined}
                  >
                    {entity.entity_text}
                    {entity.confidence_score && (
                      <span className="ml-1 text-xs text-gray-500">
                        {(entity.confidence_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </motion.span>
                ))}
                {typeEntities.length > 10 && (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                    +{typeEntities.length - 10} more
                  </span>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default EntitiesChart;