import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ExclamationTriangleIcon, 
  CheckCircleIcon,
  XCircleIcon 
} from '@heroicons/react/24/outline';
import Modal from './Modal';

const ConfirmationModal = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title = "Confirm Action",
  message = "Are you sure you want to proceed?",
  confirmText = "Confirm",
  cancelText = "Cancel",
  type = "warning", // warning, danger, success, info
  requireTextConfirmation = false,
  confirmationText = "DELETE",
  isLoading = false,
  loadingText = "Processing..."
}) => {
  const [step, setStep] = useState(1);
  const [textInput, setTextInput] = useState('');

  const typeConfig = {
    warning: {
      icon: ExclamationTriangleIcon,
      iconColor: 'text-yellow-600 dark:text-yellow-400',
      iconBg: 'bg-yellow-100 dark:bg-yellow-900',
      buttonColor: 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500'
    },
    danger: {
      icon: XCircleIcon,
      iconColor: 'text-red-600 dark:text-red-400',
      iconBg: 'bg-red-100 dark:bg-red-900',
      buttonColor: 'bg-red-600 hover:bg-red-700 focus:ring-red-500'
    },
    success: {
      icon: CheckCircleIcon,
      iconColor: 'text-green-600 dark:text-green-400',
      iconBg: 'bg-green-100 dark:bg-green-900',
      buttonColor: 'bg-green-600 hover:bg-green-700 focus:ring-green-500'
    },
    info: {
      icon: ExclamationTriangleIcon,
      iconColor: 'text-blue-600 dark:text-blue-400',
      iconBg: 'bg-blue-100 dark:bg-blue-900',
      buttonColor: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'
    }
  };

  const config = typeConfig[type];
  const IconComponent = config.icon;

  const handleClose = () => {
    setStep(1);
    setTextInput('');
    onClose();
  };

  const handleFirstConfirm = () => {
    if (requireTextConfirmation) {
      setStep(2);
    } else {
      onConfirm();
    }
  };

  const handleFinalConfirm = () => {
    if (textInput === confirmationText) {
      onConfirm();
    }
  };

  const isTextConfirmationValid = textInput === confirmationText;

  return (
    <Modal 
      isOpen={isOpen} 
      onClose={handleClose}
      closeOnOverlayClick={!isLoading}
      showCloseButton={!isLoading}
    >
      <div className="sm:flex sm:items-start">
        {/* Icon */}
        <div className={`mx-auto flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full ${config.iconBg} sm:mx-0 sm:h-10 sm:w-10`}>
          <IconComponent className={`h-6 w-6 ${config.iconColor}`} />
        </div>

        {/* Content */}
        <div className="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left flex-1">
          <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white">
            {title}
          </h3>
          
          <div className="mt-2">
            {step === 1 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-3"
              >
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {message}
                </p>
                {requireTextConfirmation && (
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3">
                    <p className="text-sm text-yellow-800 dark:text-yellow-200">
                      ⚠️ This action cannot be undone. You will need to confirm by typing "{confirmationText}" in the next step.
                    </p>
                  </div>
                )}
              </motion.div>
            )}

            {step === 2 && requireTextConfirmation && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-4"
              >
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  To confirm this action, please type <span className="font-mono font-bold text-red-600 dark:text-red-400">"{confirmationText}"</span> below:
                </p>
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder={`Type "${confirmationText}" to confirm`}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-red-500 focus:border-red-500 dark:bg-gray-700 dark:text-white"
                  autoFocus
                  disabled={isLoading}
                />
                {textInput && !isTextConfirmationValid && (
                  <p className="text-sm text-red-600 dark:text-red-400">
                    Text doesn't match. Please type "{confirmationText}" exactly.
                  </p>
                )}
              </motion.div>
            )}

            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center justify-center py-4"
              >
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                  {loadingText}
                </span>
              </motion.div>
            )}
          </div>
        </div>
      </div>

      {/* Actions */}
      {!isLoading && (
        <div className="mt-5 sm:mt-4 sm:flex sm:flex-row-reverse">
          {step === 1 && (
            <>
              <button
                type="button"
                className={`inline-flex w-full justify-center rounded-md border border-transparent px-4 py-2 text-base font-medium text-white shadow-sm ${config.buttonColor} focus:outline-none focus:ring-2 focus:ring-offset-2 sm:ml-3 sm:w-auto sm:text-sm transition-colors duration-200`}
                onClick={handleFirstConfirm}
              >
                {requireTextConfirmation ? 'Continue' : confirmText}
              </button>
              <button
                type="button"
                className="mt-3 inline-flex w-full justify-center rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-2 text-base font-medium text-gray-700 dark:text-gray-300 shadow-sm hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 sm:mt-0 sm:w-auto sm:text-sm transition-colors duration-200"
                onClick={handleClose}
              >
                {cancelText}
              </button>
            </>
          )}

          {step === 2 && requireTextConfirmation && (
            <>
              <button
                type="button"
                className={`inline-flex w-full justify-center rounded-md border border-transparent px-4 py-2 text-base font-medium text-white shadow-sm ${config.buttonColor} focus:outline-none focus:ring-2 focus:ring-offset-2 sm:ml-3 sm:w-auto sm:text-sm transition-colors duration-200 ${!isTextConfirmationValid ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={handleFinalConfirm}
                disabled={!isTextConfirmationValid}
              >
                {confirmText}
              </button>
              <button
                type="button"
                className="mt-3 inline-flex w-full justify-center rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-2 text-base font-medium text-gray-700 dark:text-gray-300 shadow-sm hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 sm:mt-0 sm:w-auto sm:text-sm transition-colors duration-200"
                onClick={() => setStep(1)}
              >
                Back
              </button>
            </>
          )}
        </div>
      )}
    </Modal>
  );
};

export default ConfirmationModal;