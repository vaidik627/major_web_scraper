import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  GlobeAltIcon,
  SparklesIcon,
  ChartBarIcon,
  CloudArrowDownIcon,
  CpuChipIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

const features = [
  {
    name: 'AI-Powered Extraction',
    description: 'Intelligent content analysis and automatic data extraction using advanced AI models.',
    icon: SparklesIcon,
  },
  {
    name: 'Dynamic & Static Sites',
    description: 'Handle both JavaScript-heavy dynamic sites and traditional static websites seamlessly.',
    icon: GlobeAltIcon,
  },
  {
    name: 'Advanced Analytics',
    description: 'Comprehensive data visualization, sentiment analysis, and trend tracking.',
    icon: ChartBarIcon,
  },
  {
    name: 'Multiple Export Formats',
    description: 'Export your data in CSV, Excel, JSON formats with one-click convenience.',
    icon: CloudArrowDownIcon,
  },
  {
    name: 'Smart Automation',
    description: 'Automated scheduling, bulk processing, and intelligent content detection.',
    icon: CpuChipIcon,
  },
  {
    name: 'Enterprise Security',
    description: 'Secure authentication, data encryption, and privacy-focused architecture.',
    icon: ShieldCheckIcon,
  },
];

const Landing = () => {
  return (
    <div className="min-h-screen bg-animated">
      {/* Header */}
      <header className="relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-600">
                <GlobeAltIcon className="h-6 w-6 text-white" />
              </div>
              <span className="ml-3 text-xl font-bold text-gray-900 dark:text-white">
                AI Web Scraper
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <Link
                to="/login"
                className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors duration-200"
              >
                Sign In
              </Link>
              <Link
                to="/register"
                className="btn-primary"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-20 sm:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-4xl sm:text-6xl font-extrabold tracking-tight text-gray-900 dark:text-white"
            >
              <span className="text-gradient">AI-Powered</span>
              <br />
              Web Scraping
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="mt-6 text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
            >
              Extract, analyze, and visualize web data with intelligent automation. 
              Our AI understands content context and delivers structured insights from any website.
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="mt-10 flex flex-col sm:flex-row gap-4 justify-center"
            >
              <Link
                to="/register"
                className="btn-primary text-lg px-8 py-3 shadow-soft"
              >
                Start Scraping Free
              </Link>
              <Link
                to="/login"
                className="btn-outline text-lg px-8 py-3"
              >
                View Demo
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white/80 backdrop-blur dark:bg-gray-800/70">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
              Everything you need for web scraping
            </h2>
            <p className="mt-4 text-lg text-gray-600 dark:text-gray-300">
              Powerful features designed for modern data extraction needs
            </p>
          </div>
          
          <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className="flex items-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-brand-600/10 to-primary-600/10">
                    <feature.icon className="h-6 w-6 text-brand-600" />
                  </div>
                  <h3 className="ml-4 text-lg font-semibold text-gray-900 dark:text-white">
                    {feature.name}
                  </h3>
                </div>
                <p className="mt-4 text-gray-600 dark:text-gray-300">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-brand-600 to-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white">
            Ready to transform your data extraction?
          </h2>
          <p className="mt-4 text-xl text-primary-100">
            Join thousands of developers and businesses using AI Web Scraper
          </p>
          <div className="mt-8">
            <Link
              to="/register"
              className="inline-flex items-center px-8 py-3 border border-transparent text-lg font-medium rounded-md text-primary-600 bg-white hover:bg-gray-50 transition-colors duration-200"
            >
              Get Started Today
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 dark:bg-gray-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="flex items-center justify-center">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
                <GlobeAltIcon className="h-5 w-5 text-white" />
              </div>
              <span className="ml-3 text-lg font-semibold text-white">
                AI Web Scraper
              </span>
            </div>
            <p className="mt-4 text-gray-400">
              Intelligent web scraping for the modern web
            </p>
            <div className="mt-8 text-sm text-gray-500">
              © 2024 AI Web Scraper. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;