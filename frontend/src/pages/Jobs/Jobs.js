import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  EyeIcon,
  TrashIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { useScraperStore } from '../../store/scraperStore';

const Jobs = () => {
  const { jobs, jobsLoading, fetchJobs, deleteJob } = useScraperStore();

  useEffect(() => {
    let intervalId;
    const load = async () => {
      try {
        await fetchJobs();
      } catch (e) {}
    };
    load();
    intervalId = setInterval(load, 5000);
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchJobs]);

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

  const handleDelete = async (jobId) => {
    if (window.confirm('Are you sure you want to delete this job?')) {
      try {
        await deleteJob(jobId);
      } catch (error) {
        console.error('Failed to delete job:', error);
      }
    }
  };

  if (jobsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner h-8 w-8"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Scraping Jobs
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage and monitor your web scraping jobs
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <Link to="/scraper" className="btn-primary">
            New Job
          </Link>
        </div>
      </div>

      {jobs.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-12"
        >
          <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No scraping jobs yet
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Get started by creating your first scraping job.
          </p>
          <div className="mt-6">
            <Link to="/scraper" className="btn-primary">
              Create Job
            </Link>
          </div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card overflow-hidden"
        >
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Job
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {jobs.map((job) => (
                  <motion.tr
                    key={job.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {getStatusIcon(job.status)}
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {job.name}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {job.total_urls} URLs
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`badge ${getStatusBadge(job.status)}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-sm text-gray-900 dark:text-white">
                          {job.processed_urls}/{job.total_urls}
                        </div>
                        <div className="ml-2 w-16 bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                          <div
                            className="bg-primary-600 h-2 rounded-full"
                            style={{
                              width: `${(job.processed_urls / job.total_urls) * 100}%`,
                            }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {new Date(job.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end space-x-2">
                        <Link
                          to={`/jobs/${job.id}`}
                          className="text-primary-600 hover:text-primary-900 dark:text-primary-400"
                        >
                          <EyeIcon className="h-4 w-4" />
                        </Link>
                        <button
                          onClick={() => handleDelete(job.id)}
                          className="text-red-600 hover:text-red-900"
                        >
                          <TrashIcon className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Jobs;