import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

import { useThemeStore } from './store/themeStore';

// Components
import Layout from './components/Layout/Layout';

// Pages
import Dashboard from './pages/Dashboard/Dashboard';
import Scraper from './pages/Scraper/Scraper';
import Jobs from './pages/Jobs/Jobs';
import JobDetail from './pages/Jobs/JobDetail';
import Analytics from './pages/Analytics/Analytics';
import Settings from './pages/Settings/Settings';
import Landing from './pages/Landing/Landing';
import QuickScrape from './pages/QuickScrape/QuickScrape';

function App() {
  const { isDark, initializeTheme } = useThemeStore();

  useEffect(() => {
    initializeTheme();
  }, [initializeTheme]);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <AnimatePresence mode="wait">
        <Routes>
          {/* Public Routes */}
          <Route 
            path="/" 
            element={<Navigate to="/dashboard" replace />} 
          />
          <Route 
            path="/landing" 
            element={
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Landing />
              </motion.div>
            } 
          />

          {/* Protected Routes */}
          <Route
            path="/dashboard"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Dashboard />
                  </motion.div>
                </Layout>
            }
          />
          <Route
            path="/scraper"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Scraper />
                  </motion.div>
                </Layout>
            }
          />
          <Route
            path="/quick-scrape"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <QuickScrape />
                  </motion.div>
                </Layout>
            }
          />
          <Route
            path="/jobs"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Jobs />
                  </motion.div>
                </Layout>
            }
          />
          <Route
            path="/jobs/:id"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <JobDetail />
                  </motion.div>
                </Layout>
            }
          />
          <Route
            path="/analytics"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Analytics />
                  </motion.div>
                </Layout>
            }
          />
          <Route
            path="/settings"
            element={
                <Layout>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Settings />
                  </motion.div>
                </Layout>
            }
          />

          {/* Catch all route */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AnimatePresence>
    </div>
  );
}

export default App;