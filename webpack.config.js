const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  mode: 'development',
  entry: './src/js/index.js',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
    clean: true,
    publicPath: '/',
  },
  devtool: 'inline-source-map',
  devServer: {
    static: [
      './dist',
      { directory: path.resolve(__dirname, 'presentation'), publicPath: '/presentation' },
    ],
    hot: true,
    historyApiFallback: {
      rewrites: [
        { from: /./, to: '/404.html' }
      ]
    },
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'MLP Visualizer',
      template: './src/index.html',
    }),
    new HtmlWebpackPlugin({
      template: './src/404.html',
      filename: '404.html',
      inject: false
    }),
    new CopyPlugin({
      patterns: [
        {
          from: 'presentation/mlp.html',
          to: 'mlp.html',
        },
        {
          from: 'presentation/*.css',
          to: '[name][ext]',
        },
        {
          from: 'presentation/*.js',
          to: '[name][ext]',
        },
        {
          from: 'src/css',
          to: 'css',
        }
      ],
    }),
  ],
};
