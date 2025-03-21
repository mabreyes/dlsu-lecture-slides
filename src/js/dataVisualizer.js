import * as d3 from 'd3';

/**
 * Class for visualizing datasets and model predictions
 */
class DataVisualizer {
  constructor(containerId, width = null, height = null) {
    this.containerId = containerId;
    console.log(`DataVisualizer initializing with container: #${containerId}`);
    this.container = d3.select(`#${containerId}`);

    if (this.container.empty()) {
      console.error(`Container #${containerId} not found in the DOM`);
      return;
    }

    console.log(`Container found, dimensions:`, this.container.node().getBoundingClientRect());

    this.width = width || this.container.node().getBoundingClientRect().width;
    this.height = height || this.container.node().getBoundingClientRect().height;

    // SVG setup
    this.svg = null;
    this.margin = { top: 20, right: 20, bottom: 40, left: 40 };

    // Scales
    this.xScale = null;
    this.yScale = null;

    // Data
    this.data = null;
    this.predictionData = null;

    // Model for predictions
    this.model = null;

    // Type of visualization
    this.visualizationType = '2d-scatter';

    // Initialize SVG
    this.initSVG();
  }

  /**
   * Initialize the SVG element
   */
  initSVG() {
    this.container.selectAll('svg').remove();

    this.svg = this.container
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('viewBox', [0, 0, this.width, this.height])
      .style('font', "10px -apple-system, BlinkMacSystemFont, 'Inter', sans-serif");
  }

  /**
   * Set data to be visualized
   */
  setData(data, type = '2d-scatter') {
    this.data = data;
    this.visualizationType = type;
  }

  /**
   * Set model for making predictions
   */
  setModel(model) {
    this.model = model;
    console.log('Model set in DataVisualizer');
  }

  /**
   * Render visualization based on type
   */
  render() {
    if (!this.data) {
      console.error('No data available');
      return;
    }

    // Clear previous visualization
    this.svg.selectAll('*').remove();

    // Determine which visualization to use
    switch (this.visualizationType) {
      case '2d-scatter':
        this.render2DScatter();
        break;
      case 'line':
        this.renderLine();
        break;
      default:
        console.error('Unknown visualization type:', this.visualizationType);
    }
  }

  /**
   * Render 2D scatter plot
   */
  render2DScatter() {
    // Create scales
    const plotWidth = this.width - this.margin.left - this.margin.right;
    const plotHeight = this.height - this.margin.top - this.margin.bottom;

    // Check if we have 2D input data
    if (this.data[0].input.length !== 2) {
      console.error('2D scatter plot requires 2D input data');
      return;
    }

    // Extract x and y values from data
    const xValues = this.data.map(d => d.input[0]);
    const yValues = this.data.map(d => d.input[1]);

    // Create scales with some padding
    const xExtent = d3.extent(xValues);
    const yExtent = d3.extent(yValues);

    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1;

    this.xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([this.margin.left, this.width - this.margin.right]);

    this.yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([this.height - this.margin.bottom, this.margin.top]);

    // Create plot container
    const plot = this.svg.append('g').attr('class', 'plot');

    // Add axes
    plot
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.height - this.margin.bottom})`)
      .call(d3.axisBottom(this.xScale).tickSize(6).tickPadding(8));

    plot
      .append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${this.margin.left},0)`)
      .call(d3.axisLeft(this.yScale).tickSize(6).tickPadding(8));

    // If we have a model, render decision boundaries
    if (this.model) {
      this.renderDecisionBoundary(plot);
    }

    // Add data points
    plot
      .selectAll('.data-point')
      .data(this.data)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => this.xScale(d.input[0]))
      .attr('cy', d => this.yScale(d.input[1]))
      .attr('r', 9)
      .style('fill', d => {
        // Color based on output class (assuming binary classification)
        if (d.output.length === 1) {
          return d.output[0] > 0.5 ? '#3498db' : '#e74c3c';
        } else {
          // For multi-class, use index of max value
          const maxIndex = d.output.indexOf(Math.max(...d.output));
          const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'];
          return colors[maxIndex % colors.length];
        }
      })
      .style('stroke', '#fff')
      .style('stroke-width', 1.5)
      .style('opacity', 0.8);
  }

  /**
   * Render decision boundary for 2D data
   */
  renderDecisionBoundary(plot) {
    // Create a grid of points
    const gridSize = 50;
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    const xStep = (xDomain[1] - xDomain[0]) / gridSize;
    const yStep = (yDomain[1] - yDomain[0]) / gridSize;

    // Generate grid points
    const gridData = [];
    const inputs = [];

    for (let x = xDomain[0]; x <= xDomain[1]; x += xStep) {
      for (let y = yDomain[0]; y <= yDomain[1]; y += yStep) {
        inputs.push([x, y]);
        gridData.push({
          x,
          y,
          prediction: null, // Will be filled later
        });
      }
    }

    // Make predictions in batch for better performance
    try {
      console.log('Making grid predictions...');
      const predictions = this.model.predict(inputs);

      // Update gridData with predictions
      for (let i = 0; i < gridData.length; i++) {
        gridData[i].prediction = Array.isArray(predictions[i]) ? predictions[i] : [predictions[i]];
      }

      // Create color scale with more intense colors
      const colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]);

      // More vivid colors for multi-class predictions
      const vividColors = ['#1a73e8', '#e53935', '#00c853', '#ff8f00', '#8e24aa'];

      // Draw boundary
      plot
        .selectAll('.grid-point')
        .data(gridData)
        .enter()
        .append('rect')
        .attr('class', 'grid-point')
        .attr('x', d => this.xScale(d.x) - xStep / 2)
        .attr('y', d => this.yScale(d.y) - yStep / 2)
        .attr('width', Math.ceil(this.xScale(xDomain[0] + xStep) - this.xScale(xDomain[0])))
        .attr('height', Math.ceil(this.yScale(yDomain[0]) - this.yScale(yDomain[0] + yStep)))
        .style('fill', d => {
          // Determine color based on prediction
          if (!d.prediction || d.prediction.length === 0) {
            return '#CCCCCC'; // Default gray for missing predictions
          }

          if (d.prediction.length === 1) {
            // Apply a more saturated color for binary classification
            const value = d.prediction[0];
            // Apply a gamma correction to increase contrast
            const gamma = 0.7; // Values less than 1 increase contrast
            const adjustedValue = Math.pow(value, gamma);
            return colorScale(adjustedValue);
          } else {
            // For multi-class, use index of max value for color with more vivid colors
            const maxIndex = d.prediction.indexOf(Math.max(...d.prediction));
            return vividColors[maxIndex % vividColors.length];
          }
        })
        .style('opacity', 0.8); // Increased from 0.6 for more visible predictions
    } catch (error) {
      console.error('Error rendering decision boundary:', error);
    }
  }

  /**
   * Render line chart for 1D function approximation
   */
  renderLine() {
    // Create scales
    const plotWidth = this.width - this.margin.left - this.margin.right;
    const plotHeight = this.height - this.margin.top - this.margin.bottom;

    // Check if we have 1D input data
    if (this.data[0].input.length !== 1) {
      console.error('Line chart requires 1D input data');
      return;
    }

    // Extract x and y values from data
    const xValues = this.data.map(d => d.input[0]);
    const yValues = this.data.map(d => d.output[0]);

    // Create scales with some padding
    const xExtent = d3.extent(xValues);
    const yExtent = d3.extent(yValues);

    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 0.1;

    this.xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([this.margin.left, this.width - this.margin.right]);

    this.yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([this.height - this.margin.bottom, this.margin.top]);

    // Create plot container
    const plot = this.svg.append('g').attr('class', 'plot');

    // Add axes
    plot
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.height - this.margin.bottom})`)
      .call(d3.axisBottom(this.xScale).tickSize(6).tickPadding(8));

    plot
      .append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${this.margin.left},0)`)
      .call(d3.axisLeft(this.yScale).tickSize(6).tickPadding(8));

    // Add data points
    plot
      .selectAll('.data-point')
      .data(this.data)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => this.xScale(d.input[0]))
      .attr('cy', d => this.yScale(d.output[0]))
      .attr('r', 7)
      .style('fill', '#3498db')
      .style('stroke', '#fff')
      .style('stroke-width', 1);

    // If we have a model, render the predictions
    if (this.model) {
      this.renderCurve(plot);
    }
  }

  /**
   * Render predicted curve for 1D function approximation
   */
  renderCurve(plot) {
    try {
      // Create a set of points along the x-axis for predictions
      const numPoints = 100;
      const xDomain = this.xScale.domain();
      const xStep = (xDomain[1] - xDomain[0]) / numPoints;

      // Generate points and get predictions
      const curvePoints = [];
      for (let x = xDomain[0]; x <= xDomain[1]; x += xStep) {
        try {
          // Get prediction for this x value
          const prediction = this.model.predict([x]);
          const predictedY = prediction[0];

          curvePoints.push({
            x: x,
            y: predictedY,
          });
        } catch (error) {
          console.error('Error making prediction:', error);
        }
      }

      // Define line generator
      const lineGenerator = d3
        .line()
        .x(d => this.xScale(d.x))
        .y(d => this.yScale(d.y))
        .curve(d3.curveBasis);

      // Add the curve
      plot
        .append('path')
        .datum(curvePoints)
        .attr('class', 'prediction-curve')
        .attr('fill', 'none')
        .attr('stroke', '#d32f2f') // More vivid red
        .attr('stroke-width', 5)
        .attr('stroke-dasharray', '8,4')
        .attr('d', lineGenerator)
        .style('filter', 'drop-shadow(0px 2px 3px rgba(0, 0, 0, 0.3))'); // Add shadow for better visibility

      // Add a subtle glow effect
      plot
        .append('path')
        .datum(curvePoints)
        .attr('class', 'prediction-curve-glow')
        .attr('fill', 'none')
        .attr('stroke', '#ff5252')
        .attr('stroke-width', 9)
        .attr('stroke-opacity', 0.3)
        .attr('stroke-linecap', 'round')
        .attr('d', lineGenerator)
        .lower(); // Put it behind the main curve
    } catch (error) {
      console.error('Error rendering prediction curve:', error);
    }
  }

  /**
   * Resize visualization
   */
  resize(width, height) {
    this.width = width || this.container.node().getBoundingClientRect().width;
    this.height = height || this.container.node().getBoundingClientRect().height;

    this.svg
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('viewBox', [0, 0, this.width, this.height]);

    if (this.data) {
      this.render();
    }
  }

  /**
   * Update model for making predictions
   */
  updateModel(model) {
    this.model = model;
    // Re-render with updated model
    this.render();
    console.log('Model updated in DataVisualizer');
  }
}

export default DataVisualizer;
