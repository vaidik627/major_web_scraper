import React, { useContext } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { GlobeAltIcon, MoonIcon, SunIcon, Bars3Icon } from '@heroicons/react/24/outline';
import { useThemeStore } from '../../store/themeStore';
import { SidebarContext } from './Layout';

const nav = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Scraper', to: '/scraper' },
  { label: 'Quick Scrape', to: '/quick-scrape' },
  { label: 'Analytics', to: '/analytics' },
];

const StickyHeader = () => {
  const { isDark, toggleTheme } = useThemeStore();
  const { toggleSidebar } = useContext(SidebarContext);
          
  return (
    <header className="sticky top-0 z-40 bg-white/70 dark:bg-gray-900/70 backdrop-blur border-b border-gray-200/60 dark:border-gray-800/60 shadow-soft">
      <div className="mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center">
          <button
            type="button"
            onClick={toggleSidebar}
            className="mr-2 rounded-md p-2 text-gray-500 hover:text-gray-800 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800"
            aria-label="Open menu"
            title="Toggle menu"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          <Link to="/" className="flex items-center">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-600 to-primary-600 shadow-glow">
            <GlobeAltIcon className="h-5 w-5 text-white" />
          </span>
          <span className="ml-3 text-lg font-extrabold tracking-tight text-gradient">AI Web Scraper</span>
          </Link>
        </div>

        <nav className="hidden md:flex items-center gap-6">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                `text-sm transition-colors ${isActive
                  ? 'text-brand-700 dark:text-brand-300'
                  : 'text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white'}`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <button
            onClick={toggleTheme}
            className="rounded-lg p-2 text-gray-500 hover:text-gray-800 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800 transition"
            title={isDark ? 'Light mode' : 'Dark mode'}
          >
            {isDark ? <SunIcon className="h-5 w-5" /> : <MoonIcon className="h-5 w-5" />}
          </button>
          
          
        </div>
      </div>
    </header>
  );
};

export default StickyHeader;


