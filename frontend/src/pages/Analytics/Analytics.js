import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  EyeIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline';
import { useScraperStore } from '../../store/scraperStore';
import EnhancedAnalyticsDashboard from '../../components/Analytics/EnhancedAnalyticsDashboard';

const Analytics = () => {
  const { fetchDashboardData } = useScraperStore();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchDashboardData();
        setDashboardData(data);
      } catch (error) {
        console.error('Failed to load analytics data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [fetchDashboardData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner h-8 w-8"></div>
      </div>
    );
  }

  const stats = dashboardData?.overview || {};

  const tabs = [
    {
      id: 'overview',
      name: 'Overview',
      icon: ChartBarIcon,
      description: 'General scraping analytics and metrics'
    },
    {
      id: 'ai-insights',
      name: 'AI Insights',
      icon: CpuChipIcon,
      description: 'Advanced AI-powered content analysis'
    }
  ];

  const renderTabContent = () => {
    if (activeTab === 'ai-insights') {
      return <EnhancedAnalyticsDashboard />;
    }

    // Original overview content
    return (
      <div className="space-y-8">

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="card p-6"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100 dark:bg-primary-900">
              <ChartBarIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Jobs
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.total_jobs || 0}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="card p-6"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-success-100 dark:bg-success-900">
              <ArrowTrendingUpIcon className="h-6 w-6 text-success-600 dark:text-success-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Success Rate
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.total_jobs > 0 
                  ? Math.round((stats.completed_jobs / stats.total_jobs) * 100)
                  : 0}%
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="card p-6"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900">
              <EyeIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Items Scraped
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {(stats.total_scraped_items || 0).toLocaleString()}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="card p-6"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-warning-100 dark:bg-warning-900">
              <ChartBarIcon className="h-6 w-6 text-warning-600 dark:text-warning-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Running Jobs
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.running_jobs || 0}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts Placeholder */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.4 }}
        className="card p-6"
      >
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Job Performance Over Time
        </h2>
        <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="text-center">
            <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
              Charts Coming Soon
            </h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Interactive charts and visualizations will be available here.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.5 }}
        className="card p-6"
      >
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Recent Activity
        </h2>
        {dashboardData?.recent_jobs?.length > 0 ? (
          <div className="space-y-3">
            {dashboardData.recent_jobs.slice(0, 5).map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
              >
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {job.name}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(job.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-900 dark:text-white">
                    {job.processed_urls}/{job.total_urls} URLs
                  </p>
                  <span className={`badge ${
                    job.status === 'completed' ? 'badge-success' :
                    job.status === 'running' ? 'badge-warning' : 'badge-error'
                  }`}>
                    {job.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-500 dark:text-gray-400">No recent activity</p>
          </div>
        )}
      </motion.div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Analytics
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Insights and trends from your web scraping activities
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const IconComponent = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group inline-flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <IconComponent className="h-5 w-5 mr-2" />
                {tab.name}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {renderTabContent()}
      </motion.div>
    </div>
  );
};

export default Analytics;