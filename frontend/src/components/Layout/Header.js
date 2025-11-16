import React from 'react';
import { Menu, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { Link } from 'react-router-dom';
import {
  Bars3Icon,
  UserCircleIcon,
  ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../store/authStore';

const Header = ({ onMenuClick }) => {
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
  };

  return (
    <header className="bg-white/70 dark:bg-gray-900/60 backdrop-blur">
      <div className="flex h-16 items-center justify-between px-6">
        {/* Left side */}
        <div className="flex items-center">
          {/* Hamburger menu button */}
          <button
            onClick={onMenuClick}
            className="rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 dark:hover:bg-gray-700 dark:hover:text-gray-300"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          
          {/* Logo/Brand */}
          <div className="ml-4 lg:ml-0">
            <h1 className="text-lg font-semibold text-gradient">
              AI Web Scraper
            </h1>
          </div>
        </div>

        {/* Right side - Profile menu only */}
        <div className="flex items-center">
          {/* User menu */}
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
        </div>
      </div>
    </header>
  );
};

export default Header;