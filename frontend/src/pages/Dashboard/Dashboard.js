import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  PlusIcon,
  GlobeAltIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { useScraperStore } from '../../store/scraperStore';

const Dashboard = () => {
  const { fetchJobs, fetchDashboardData } = useScraperStore();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let intervalId;

    const loadDashboardData = async () => {
      try {
        const [, dashData] = await Promise.all([
          fetchJobs({ silent: true }),
          fetchDashboardData({ silent: true }),
        ]);
        setDashboardData(dashData);
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
    intervalId = setInterval(loadDashboardData, 5000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchJobs, fetchDashboardData]);

  const stats = dashboardData?.overview || {
    total_jobs: 0,
    completed_jobs: 0,
    running_jobs: 0,
    failed_jobs: 0,
    total_scraped_items: 0,
  };

  const recentJobs = dashboardData?.recent_jobs || [];

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

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return 'badge-success';
      case 'running':
        return 'badge-warning';
      case 'failed':
        return 'badge-error';
      default:
        return 'badge-info';
    }
  };

  if (loading) {
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
        <h1 className="text-2xl font-extrabold tracking-tight text-gradient">Dashboard</h1>
        <div className="mt-4 sm:mt-0 flex items-center space-x-3">
          <Link
            to="/scraper"
            className="btn-primary inline-flex items-center"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            New Scraping Job
          </Link>
          <button
            type="button"
            onClick={async () => {
              setLoading(true);
              try {
                const [, dashData] = await Promise.all([
                  fetchJobs({ silent: true }),
                  fetchDashboardData({ silent: true }),
                ]);
                setDashboardData(dashData);
              } finally {
                setLoading(false);
              }
            }}
            className="btn-outline"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-brand-600/10 to-primary-600/10">
              <GlobeAltIcon className="h-6 w-6 text-brand-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Jobs
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.total_jobs}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-success-100 dark:bg-success-900">
              <CheckCircleIcon className="h-6 w-6 text-success-600 dark:text-success-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Completed
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.completed_jobs}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-warning-100 dark:bg-warning-900">
              <ClockIcon className="h-6 w-6 text-warning-600 dark:text-warning-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Running
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.running_jobs}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-error-100 dark:bg-error-900">
              <ExclamationTriangleIcon className="h-6 w-6 text-error-600 dark:text-error-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Failed
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.failed_jobs}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
          className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-brand-100 dark:bg-brand-900">
              <ChartBarIcon className="h-6 w-6 text-brand-600 dark:text-brand-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Items Scraped
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {stats.total_scraped_items.toLocaleString()}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Recent Jobs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.5 }}
        className="card"
      >
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Recent Jobs
            </h2>
            <Link
              to="/jobs"
              className="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
            >
              View all
            </Link>
          </div>
        </div>
        <div className="card-body">
          {recentJobs.length === 0 ? (
            <div className="text-center py-8">
              <GlobeAltIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                No scraping jobs yet
              </h3>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Get started by creating your first scraping job.
              </p>
              <div className="mt-6">
                <Link to="/scraper" className="btn-primary">
                  <PlusIcon className="h-4 w-4 mr-2" />
                  Create Job
                </Link>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {recentJobs.map((job) => (
                <div
                  key={job.id}
                  className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200"
                >
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(job.status)}
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        {job.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {job.processed_urls}/{job.total_urls} URLs processed
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className={`badge ${getStatusBadge(job.status)}`}>
                      {job.status}
                    </span>
                    <Link
                      to={`/jobs/${job.id}`}
                      className="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400"
                    >
                      View
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.6 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        <Link
          to="/scraper"
          className="card p-6 hover:shadow-lg transition-shadow duration-300 group"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100 dark:bg-primary-900 group-hover:bg-primary-200 dark:group-hover:bg-primary-800 transition-colors duration-200">
              <PlusIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                New Scraping Job
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Start scraping websites with AI
              </p>
            </div>
          </div>
        </Link>

        <Link
          to="/jobs"
          className="card p-6 hover:shadow-lg transition-shadow duration-300 group"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-success-100 dark:bg-success-900 group-hover:bg-success-200 dark:group-hover:bg-success-800 transition-colors duration-200">
              <GlobeAltIcon className="h-6 w-6 text-success-600 dark:text-success-400" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Manage Jobs
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                View and manage your jobs
              </p>
            </div>
          </div>
        </Link>

        <Link
          to="/analytics"
          className="card p-6 hover:shadow-lg transition-shadow duration-300 group"
        >
          <div className="flex items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900 group-hover:bg-purple-200 dark:group-hover:bg-purple-800 transition-colors duration-200">
              <ChartBarIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                View Analytics
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Analyze your scraping data
              </p>
            </div>
          </div>
        </Link>
      </motion.div>
    </div>
  );
};

export default Dashboard;