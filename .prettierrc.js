module.exports = {
  semi: true,
  singleQuote: true,
  tabWidth: 2,
  trailingComma: 'es5',
  printWidth: 100,
  bracketSpacing: true,
  arrowParens: 'avoid',
  endOfLine: 'lf',
  overrides: [
    {
      files: '*.md',
      options: {
        proseWrap: 'always',
        tabWidth: 4,
      },
    },
    {
      files: '*.json',
      options: {
        tabWidth: 2,
      },
    },
  ],
}; 