// engine.js
const { Marp } = require('@marp-team/marp-core');

// Configuration object for Marp
const marpConfig = {
  // Enable math typesetting with enhanced settings
  math: true,
  katex: {
    // Use more robust settings for KaTeX
    trust: true,
    strict: false,
    macros: {
      // Add any macro definitions here if needed
      '\\R': '\\mathbb{R}',
    },
    // Ensure formulas are properly displayed
    output: 'html',
    // Better error handling
    throwOnError: false,
    // Allow display mode
    displayMode: true,
    // Improved alignment
    fleqn: false,
  },
  // Other Marp configuration options
  markdown: {
    breaks: true,
    html: true,
  },
  // Enable HTML directly
  html: true,
};

// Create and export Marp instance with custom config
module.exports = opts => {
  // Create Marp instance with our config
  const marp = new Marp({
    ...opts,
    ...marpConfig,
  });

  // Use our custom mermaid plugin
  return marp.use(require('./mermaid-plugin'));
};
