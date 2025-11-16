import React from 'react';

const AnimatedBackground = () => {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-brand-50 via-white to-primary-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800" />
      <div className="absolute inset-0 opacity-70">
        <div className="absolute -inset-1 blur-2xl">
          <div className="bg-[radial-gradient(700px_700px_at_0%_0%,rgba(139,92,246,0.25),transparent_60%)] animate-[bgFloat_18s_ease-in-out_infinite_alternate]" />
          <div className="bg-[radial-gradient(700px_700px_at_100%_0%,rgba(59,130,246,0.20),transparent_60%)] animate-[bgFloat_20s_ease-in-out_infinite_alternate]" />
          <div className="bg-[radial-gradient(700px_700px_at_100%_100%,rgba(16,185,129,0.18),transparent_60%)] animate-[bgFloat_22s_ease-in-out_infinite_alternate]" />
          <div className="bg-[radial-gradient(700px_700px_at_0%_100%,rgba(236,72,153,0.16),transparent_60%)] animate-[bgFloat_24s_ease-in-out_infinite_alternate]" />
        </div>
      </div>
      <svg className="absolute bottom-0 left-0 right-0 h-48 text-brand-600/10 dark:text-brand-400/10" viewBox="0 0 1440 320" preserveAspectRatio="none" aria-hidden="true">
        <path fill="currentColor" d="M0,256L48,256C96,256,192,256,288,224C384,192,480,128,576,122.7C672,117,768,171,864,186.7C960,203,1056,181,1152,170.7C1248,160,1344,160,1392,160L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z" />
      </svg>
    </div>
  );
};

export default AnimatedBackground;


