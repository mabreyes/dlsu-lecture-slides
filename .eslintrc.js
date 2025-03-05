module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  extends: ['eslint:recommended', 'plugin:import/recommended', 'prettier'],
  parserOptions: {
    ecmaVersion: 2021,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: ['import'],
  rules: {
    // Error prevention
    'no-console': ['warn', { allow: ['warn', 'error', 'info'] }],
    'no-debugger': 'warn',
    'no-alert': 'warn',
    'no-var': 'error',
    'prefer-const': 'error',

    // Style consistency
    indent: ['error', 2],
    'linebreak-style': ['error', 'unix'],
    quotes: ['error', 'single', { avoidEscape: true }],
    semi: ['error', 'always'],
    'comma-dangle': ['error', 'always-multiline'],

    // Best practices
    curly: ['error', 'multi-line'],
    'brace-style': ['error', '1tbs', { allowSingleLine: true }],
    'block-spacing': ['error', 'always'],
    'comma-spacing': ['error', { before: false, after: true }],
    'array-bracket-spacing': ['error', 'never'],
    'object-curly-spacing': ['error', 'always'],
    'arrow-spacing': ['error', { before: true, after: true }],
    'key-spacing': ['error', { beforeColon: false, afterColon: true }],
    'keyword-spacing': ['error', { before: true, after: true }],

    // ES6+
    'arrow-parens': ['error', 'as-needed'],
    'no-confusing-arrow': ['error', { allowParens: true }],
    'no-duplicate-imports': 'error',
    'no-useless-constructor': 'error',
    'no-useless-computed-key': 'error',
    'prefer-template': 'error',
    'template-curly-spacing': ['error', 'never'],

    // Import plugin rules
    'import/no-unresolved': 'error',
    'import/named': 'error',
    'import/default': 'error',
    'import/namespace': 'error',
    'import/order': [
      'error',
      {
        groups: ['builtin', 'external', 'internal', 'parent', 'sibling', 'index'],
        'newlines-between': 'always',
      },
    ],
  },
  settings: {
    'import/resolver': {
      node: {
        extensions: ['.js', '.jsx', '.ts', '.tsx'],
      },
    },
  },
  overrides: [
    {
      files: ['**/*.ts', '**/*.tsx'],
      parser: '@typescript-eslint/parser',
      extends: ['plugin:@typescript-eslint/recommended'],
      plugins: ['@typescript-eslint'],
      rules: {
        // TypeScript specific rules
        '@typescript-eslint/explicit-function-return-type': 'off',
        '@typescript-eslint/explicit-module-boundary-types': 'off',
        '@typescript-eslint/no-explicit-any': 'warn',
        '@typescript-eslint/no-unused-vars': [
          'error',
          {
            argsIgnorePattern: '^_',
            varsIgnorePattern: '^_',
          },
        ],
      },
    },
    {
      files: [
        '**/*.test.js',
        '**/*.test.ts',
        '**/*.test.jsx',
        '**/*.test.tsx',
        '**/__tests__/**/*',
      ],
      env: {
        jest: true,
      },
      rules: {
        // Relaxed rules for test files
        'no-console': 'off',
      },
    },
  ],
};
