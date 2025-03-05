module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  extends: [
    'standard'
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  rules: {
    'semi': ['error', 'always'],
    'quotes': ['error', 'single'],
    'no-console': 'warn',
    'no-unused-vars': 'warn',
    'space-before-function-paren': ['error', {
      'anonymous': 'always',
      'named': 'never',
      'asyncArrow': 'always'
    }],
    'camelcase': 'warn',
    'indent': ['error', 2],
    'arrow-spacing': ['error', { 'before': true, 'after': true }],
    'no-var': 'error',
    'prefer-const': 'error',
    'no-multiple-empty-lines': ['error', { 'max': 2, 'maxEOF': 1 }]
  },
  ignorePatterns: [
    'node_modules/',
    'dist/',
    'build/'
  ]
}; 