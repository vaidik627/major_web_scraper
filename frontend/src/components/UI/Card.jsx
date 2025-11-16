import React from 'react';
import { motion } from 'framer-motion';

const Card = ({ children, className = '' }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className={`card p-5 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 ${className}`}
    >
      {children}
    </motion.div>
  );
};

export default Card;
