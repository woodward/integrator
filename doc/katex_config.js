
// console.log("inside of katex_config");

(function() {
  // console.log("after page loaded");

  renderMathInElement(document.body, {
    // customised options
    // • auto-render specific keys, e.g.:
    delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        // {left: '\\(', right: '\\)', display: false},
        // {left: '\\[', right: '\\]', display: true}
    ],
    // • rendering keys, e.g.:
    throwOnError : false
  } );

})();
