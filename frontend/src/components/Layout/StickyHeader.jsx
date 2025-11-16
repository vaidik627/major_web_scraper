import React, { useContext } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Menu, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { GlobeAltIcon, MoonIcon, SunIcon, UserCircleIcon, ArrowRightOnRectangleIcon, Bars3Icon } from '@heroicons/react/24/outline';
import { useThemeStore } from '../../store/themeStore';
import { useAuthStore } from '../../store/authStore';
import { SidebarContext } from './Layout';

const nav = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Scraper', to: '/scraper' },
  { label: 'Quick Scrape', to: '/quick-scrape' },
  { label: 'Analytics', to: '/analytics' },
];

const StickyHeader = () => {
  const { isDark, toggleTheme } = useThemeStore();
  const { user, logout } = useAuthStore();
  const { toggleSidebar } = useContext(SidebarContext);

  const handleLogout = () => {
    logout();
  };

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
          
          {/* User menu - only show if user is logged in */}
          {user ? (
            <Menu as="div" className="relative">
              <Menu.Button className="flex items-center rounded-full bg-white/80 dark:bg-gray-800/80 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900">
                <span className="sr-only">Open user menu</span>
                <div className="flex items-center space-x-3">
                  <UserCircleIcon className="h-8 w-8 text-brand-600" />
                  <div className="hidden md:block text-left">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {user?.username}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {user?.email}
                    </div>
                  </div>
                </div>
              </Menu.Button>
              
              <Transition
                as={Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-xl bg-white/95 dark:bg-gray-800/95 py-2 shadow-soft ring-1 ring-black/5 focus:outline-none">
                  <Menu.Item>
                    {({ active }) => (
                      <Link
                        to="/settings"
                        className={`${active ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white' : 'text-gray-700 dark:text-gray-300'} block px-4 py-2 text-sm rounded-md transition-colors duration-200`}
                      >
                        Settings
                      </Link>
                    )}
                  </Menu.Item>
                  <Menu.Item>
                    {({ active }) => (
                      <button
                        onClick={handleLogout}
                        className={`${active ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white' : 'text-gray-700 dark:text-gray-300'} w-full text-left block px-4 py-2 text-sm rounded-md transition-colors duration-200 flex items-center`}
                      >
                        <ArrowRightOnRectangleIcon className="mr-2 h-4 w-4" />
                        Sign out
                      </button>
                    )}
                  </Menu.Item>
                </Menu.Items>
              </Transition>
            </Menu>
          ) : (
            // Show login/register buttons only if user is not logged in
            <>
              <Link to="/login" className="hidden sm:inline-flex btn-outline">Sign in</Link>
              <Link to="/register" className="hidden sm:inline-flex btn-primary">Get Started</Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default StickyHeader;


