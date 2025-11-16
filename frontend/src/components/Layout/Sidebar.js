import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HomeIcon,
  GlobeAltIcon,
  BriefcaseIcon,
  ChartBarIcon,
  CogIcon,
  XMarkIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Scraper', href: '/scraper', icon: GlobeAltIcon },
  { name: 'Quick Scrape', href: '/quick-scrape', icon: BoltIcon },
  { name: 'Jobs', href: '/jobs', icon: BriefcaseIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
];

const Sidebar = ({ isOpen, onClose }) => {
  const location = useLocation();

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 30,
      },
    },
    closed: {
      x: '-100%',
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 30,
      },
    },
  };

  const overlayVariants = {
    open: { opacity: 1 },
    closed: { opacity: 0 },
  };

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial="closed"
            animate="open"
            exit="closed"
            variants={overlayVariants}
            className="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar - always off-canvas and overlays content */}
      <motion.div
        initial="closed"
        animate={isOpen ? 'open' : 'closed'}
        variants={sidebarVariants}
        className="fixed inset-y-0 left-0 z-50 w-64 bg-white/90 dark:bg-gray-900/80 backdrop-blur shadow-soft border-r border-gray-200/60 dark:border-gray-800/60"
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-gray-200/60 dark:border-gray-800/60">
            <Link to="/dashboard" className="flex items-center">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-600 to-primary-600 shadow-glow">
                <GlobeAltIcon className="h-5 w-5 text-white drop-shadow" />
              </div>
              <span className="ml-3 text-lg font-semibold text-gray-900 dark:text-white">
                AI Scraper
              </span>
            </Link>
            
            {/* Close button (mobile only) */}
            <button
              onClick={onClose}
              className="lg:hidden rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 dark:hover:bg-gray-700"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={onClose}
                  className={`
                    group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200
                    ${
                      isActive
                        ? 'bg-brand-50 text-brand-700 dark:bg-brand-900/40 dark:text-brand-200 shadow-sm'
                        : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-white'
                    }
                  `}
                >
                  <item.icon
                    className={`
                      mr-3 h-5 w-5 flex-shrink-0
                      ${
                        isActive
                          ? 'text-brand-600 dark:text-brand-400'
                          : 'text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300'
                      }
                    `}
                  />
                  {item.name}
                  
                  {/* Active indicator */}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute right-0 w-1 h-8 bg-brand-600 rounded-l-full"
                      initial={false}
                      transition={{
                        type: 'spring',
                        stiffness: 500,
                        damping: 30,
                      }}
                    />
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-4">
            <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
              AI Web Scraper v1.0.0
            </div>
          </div>
        </div>
      </motion.div>
    </>
  );
};

export default Sidebar;