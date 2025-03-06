/**
 * Custom script to render mathematical formulas inside Mermaid diagrams
 * This solves the issue of formulas not rendering properly in Mermaid
 */
document.addEventListener('DOMContentLoaded', function () {
  // Wait for both Mermaid and KaTeX to be ready
  setTimeout(function () {
    renderFormulasInMermaid();
  }, 2000); // Adjust timeout as needed

  // Add a mutation observer to handle dynamically loaded diagrams
  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      if (mutation.addedNodes.length) {
        renderFormulasInMermaid();
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
});

/**
 * Main function to process and render formulas in Mermaid diagrams
 */
function renderFormulasInMermaid() {
  // Find all rendered Mermaid SVGs
  const mermaidSvgs = document.querySelectorAll('.mermaid svg');

  mermaidSvgs.forEach(svg => {
    // Process all text elements in the SVG
    const textElements = svg.querySelectorAll('text');

    textElements.forEach(textEl => {
      const text = textEl.textContent;

      // Identify potential formulas using common patterns
      if (isLikelyFormula(text)) {
        replaceWithRenderedFormula(textEl, text);
      }
    });
  });
}

/**
 * Check if text is likely to contain a mathematical formula
 */
function isLikelyFormula(text) {
  return (
    text.includes('sigma') ||
    text.includes('max(') ||
    text.includes('partial') ||
    text.includes('tanh') ||
    (text.includes('=') && (text.includes('^') || text.includes('/')))
  );
}

/**
 * Replace a text element with a properly rendered formula
 */
function replaceWithRenderedFormula(element, formulaText) {
  // Create a temporary container for rendering
  const tempContainer = document.createElement('div');
  tempContainer.style.position = 'absolute';
  tempContainer.style.visibility = 'hidden';
  tempContainer.style.zIndex = '-1000';
  document.body.appendChild(tempContainer);

  // Convert the text to LaTeX
  const latexCode = convertToLatex(formulaText);

  try {
    // If KaTeX is available, use it to render
    if (typeof katex !== 'undefined') {
      tempContainer.innerHTML = katex.renderToString(latexCode, {
        throwOnError: false,
        displayMode: false,
        output: 'html',
      });

      // Create a foreignObject to host the HTML content
      const foreignObject = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
      foreignObject.setAttribute('width', Math.max(tempContainer.offsetWidth, 50) + 10); // Add padding
      foreignObject.setAttribute('height', Math.max(tempContainer.offsetHeight, 20) + 5);
      foreignObject.setAttribute('x', element.getAttribute('x'));
      foreignObject.setAttribute(
        'y',
        parseFloat(element.getAttribute('y')) - tempContainer.offsetHeight / 2
      );
      foreignObject.setAttribute('class', 'formula-container');

      // Create HTML div inside the foreignObject
      const div = document.createElement('div');
      div.innerHTML = tempContainer.innerHTML;
      div.style.color = window.getComputedStyle(element).color;
      foreignObject.appendChild(div);

      // Add the rendered formula
      element.parentNode.insertBefore(foreignObject, element);
      element.style.display = 'none';
    }
  } catch (e) {
    console.error('Error rendering formula:', e);
  }

  // Clean up the temporary container
  document.body.removeChild(tempContainer);
}

/**
 * Convert text with mathematical notation to LaTeX
 */
function convertToLatex(text) {
  return (
    text
      // Handle sigma notation
      .replace(/sigma\(x\)/g, '\\sigma(x)')
      .replace(/sigma\(x_i\)/g, '\\sigma(x_i)')

      // Handle partial derivatives
      .replace(/partial L\/partial (\w+)/g, '\\frac{\\partial L}{\\partial $1}')
      .replace(/partial J\/partial (\w+)/g, '\\frac{\\partial J}{\\partial $1}')

      // Handle fractions
      .replace(/1\/\(1\+e\^-x\)/g, '\\frac{1}{1+e^{-x}}')
      .replace(/\(e\^x-e\^-x\)\/\(e\^x\+e\^-x\)/g, '\\frac{e^x-e^{-x}}{e^x+e^{-x}}')
      .replace(/e\^x_i\/sum\(e\^x_j\)/g, '\\frac{e^{x_i}}{\\sum e^{x_j}}')

      // Handle max functions
      .replace(/max\(0,x\)/g, '\\max(0,x)')
      .replace(/max\(0.01x, x\)/g, '\\max(0.01x, x)')
      .replace(/max\(αx, x\)/g, '\\max(\\alpha x, x)')

      // Handle other symbols
      .replace(/sum\(/g, '\\sum(')
      .replace(/Φ\(x\)/g, '\\Phi(x)')
  );
}
