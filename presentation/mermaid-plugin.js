// mermaid-plugin.js - Custom plugin for mermaid diagrams and math rendering
module.exports = function mermaidPlugin(md) {
  const originalFence = md.renderer.rules.fence;

  // Add mermaid script and debugging to head
  const originalRender = md.renderer.render;
  md.renderer.render = function(tokens, options, env) {
    const result = originalRender.call(this, tokens, options, env);
    return result.replace(
      '</head>',
      `
      <!-- Custom styles for math and mermaid -->
      <style>
        /* Mermaid diagram styling */
        .mermaid-wrapper {
          display: flex;
          justify-content: center;
          margin: 20px 0;
          width: 100%;
        }
        .mermaid {
          background-color: transparent !important;
          font-family: 'Fira Code', monospace;
          text-align: center;
          max-width: 100%;
        }
        
        /* Math formula styling */
        .katex-display {
          margin: 1em 0 !important;
          overflow-x: auto;
          overflow-y: hidden;
          padding: 0.5em 0;
        }
        .katex { 
          font-size: 1.1em !important; 
        }
        
        /* Fix math colors on dark backgrounds */
        [data-theme-color="dark"] .katex,
        [data-theme-color="black"] .katex {
          color: #f8f8f2;
        }
        
        /* Debug info */
        .debug-info {
          background: #f8f9fa;
          border: 1px solid #ddd;
          border-radius: 4px;
          color: #333;
          font-family: monospace;
          font-size: 12px;
          margin: 10px 0;
          padding: 10px;
          white-space: pre-wrap;
          word-break: break-all;
        }
        .debug-info.error {
          background: #fff0f0;
          border-color: #ffcccc;
          color: #cc0000;
        }
      </style>
      
      <!-- Load necessary libraries -->
      <script src="https://cdn.jsdelivr.net/npm/mermaid@9.4.0/dist/mermaid.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/katex@0.16.7/dist/katex.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/katex@0.16.7/dist/contrib/auto-render.min.js"></script>
      
      <script>
        // Detailed debugging for better troubleshooting
        console.log('Document loaded, initializing plugins...');
        
        function debugInPage(message, isError = false) {
          const debugDiv = document.createElement('div');
          debugDiv.className = isError ? 'debug-info error' : 'debug-info';
          debugDiv.textContent = message;
          document.body.appendChild(debugDiv);
          console.log(message);
        }
        
        document.addEventListener('DOMContentLoaded', () => {
          console.log('DOM fully loaded');
          
          // 1. Initialize KaTeX for math rendering
          try {
            console.log('Initializing KaTeX...');
            renderMathInElement(document.body, {
              delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\\\(', right: '\\\\)', display: false},
                {left: '\\\\[', right: '\\\\]', display: true},
                {left: '\\\\begin{equation}', right: '\\\\end{equation}', display: true},
                {left: '\\\\begin{align}', right: '\\\\end{align}', display: true},
                {left: '\\\\begin{alignat}', right: '\\\\end{alignat}', display: true},
                {left: '\\\\begin{gather}', right: '\\\\end{gather}', display: true},
                {left: '\\\\begin{CD}', right: '\\\\end{CD}', display: true}
              ],
              throwOnError: false,
              errorColor: '#cc0000',
              strict: false,
              trust: true
            });
            
            // Check if KaTeX rendered anything
            const mathElements = document.querySelectorAll('.katex');
            console.log(\`KaTeX rendered \${mathElements.length} elements\`);
            
            // Look for potential math elements that weren't rendered
            document.querySelectorAll('p, li, td').forEach(el => {
              const text = el.textContent;
              if (
                (text.includes('$') && (text.match(/\\$/g) || []).length >= 2) || 
                text.includes('\\\\(') || 
                text.includes('\\\\[')
              ) {
                if (!el.querySelector('.katex')) {
                  console.warn('Potential unrendered math:', text);
                }
              }
            });
          } catch (error) {
            console.error('KaTeX initialization error:', error);
            debugInPage('KaTeX Error: ' + error.message, true);
          }
          
          // 2. Initialize Mermaid for diagrams
          try {
            console.log('Initializing Mermaid...');
            
            // Configure Mermaid
            mermaid.initialize({
              startOnLoad: false,
              theme: 'default',
              securityLevel: 'loose',
              flowchart: { 
                useMaxWidth: false, 
                htmlLabels: true,
                curve: 'basis'
              },
              sequence: { 
                useMaxWidth: false,
                wrap: true,
                width: 150
              },
              gantt: { 
                useMaxWidth: false,
                fontSize: 14 
              },
              pie: {
                useWidth: 500
              }
            });
            
            // Find all mermaid diagram containers
            const mermaidElements = document.querySelectorAll('pre.mermaid, div.mermaid, pre.language-mermaid');
            console.log(\`Found \${mermaidElements.length} mermaid elements\`);
            
            // Process each mermaid diagram
            if (mermaidElements.length > 0) {
              mermaidElements.forEach((el, index) => {
                try {
                  const id = \`mermaid-diagram-\${index}\`;
                  el.id = id;
                  
                  const content = el.textContent.trim();
                  console.log(\`Processing diagram #\${index}:\`, content.substring(0, 50) + '...');
                  
                  // Only render if we have content
                  if (content.length > 0) {
                    mermaid.render(id, content, (svgCode) => {
                      el.innerHTML = svgCode;
                      el.setAttribute('data-processed', 'true');
                    }, el);
                  } else {
                    console.warn(\`Empty mermaid diagram #\${index}\`);
                  }
                } catch (error) {
                  console.error(\`Error rendering mermaid diagram #\${index}:\`, error);
                  debugInPage(\`Mermaid Error in diagram #\${index}: \${error.message}\`, true);
                }
              });
            } else {
              console.warn('No mermaid diagrams found with the expected selectors');
              
              // Try to find potential mermaid blocks that might be mis-tagged
              const codeBlocks = document.querySelectorAll('pre code');
              codeBlocks.forEach((block, i) => {
                const content = block.textContent;
                if (
                  content.includes('graph ') || 
                  content.includes('sequenceDiagram') || 
                  content.includes('gantt') || 
                  content.includes('classDiagram')
                ) {
                  console.warn('Potential mermaid diagram found in non-mermaid code block:', content.substring(0, 50) + '...');
                }
              });
            }
          } catch (error) {
            console.error('Mermaid initialization error:', error);
            debugInPage('Mermaid Error: ' + error.message, true);
          }
        });
      </script>
      </head>`
    );
  };

  // Enhanced handling of mermaid code blocks
  md.renderer.rules.fence = function(tokens, idx, options, env, self) {
    const token = tokens[idx];
    const code = token.content.trim();
    const info = token.info ? token.info.trim().toLowerCase() : '';
    
    if (info === 'mermaid') {
      // Ensure valid unique ID
      const id = `mermaid-diagram-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
      
      // Create a structured mermaid container with failsafe rendering
      return `
        <div class="mermaid-wrapper">
          <pre class="mermaid" id="${id}" data-processed="false">${code}</pre>
        </div>
      `;
    }
    
    // Use original renderer for other languages
    return originalFence(tokens, idx, options, env, self);
  };
  
  return md;
}; 