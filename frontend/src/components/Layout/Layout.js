import React, { useState, createContext, useMemo } from 'react';
import { motion } from 'framer-motion';
import Sidebar from './Sidebar';
import AnimatedBackground from '../UI/AnimatedBackground';
import StickyHeader from './StickyHeader';

export const SidebarContext = createContext({
  sidebarOpen: false,
  openSidebar: () => {},
  closeSidebar: () => {},
  toggleSidebar: () => {},
});

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      <AnimatedBackground />
      {/* Sidebar */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      {/* Main content */}
      <motion.div
        className="flex-1 flex flex-col overflow-hidden"
        animate={{ x: sidebarOpen ? 256 : 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        style={{ willChange: 'transform' }}
      >
        <SidebarContext.Provider
          value={useMemo(
            () => ({
              sidebarOpen,
              openSidebar: () => setSidebarOpen(true),
              closeSidebar: () => setSidebarOpen(false),
              toggleSidebar: () => setSidebarOpen((prev) => !prev),
            }),
            [sidebarOpen]
          )}
        >
          {/* Header (sticky enhanced) */}
          <StickyHeader />
          {/* Page content */}
          <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50 dark:bg-gray-900">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="w-full max-w-none px-4 sm:px-6 lg:px-8 py-8"
            >
              {children}
            </motion.div>
          </main>
        </SidebarContext.Provider>
      </motion.div>
    </div>
  );
};

export default Layout;