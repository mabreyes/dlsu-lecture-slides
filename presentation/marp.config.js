module.exports = {
  // Enable KaTeX math rendering
  math: true,
  
  // Use kroki plugin for diagrams
  engine: ({ marp }) => marp.use(require('./kroki-plugin'))
} 