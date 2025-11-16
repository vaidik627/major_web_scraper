import React from 'react';
import Card from './Card';

const ResultsGrid = ({ items = [], renderItem }) => {
  if (!items.length) {
    return (
      <Card className="text-center">
        <p className="text-sm text-gray-600 dark:text-gray-300">No results yet. Start a scrape to see results here.</p>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((item, idx) => (
        <Card key={idx}>{renderItem(item, idx)}</Card>
      ))}
    </div>
  );
};

export default ResultsGrid;
