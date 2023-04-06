// Adapted from the contents of: https://katex.org/js/index.js
// and
// https://github.com/elixir-nx/livebook/blob/main/assets/js/cell/markdown.js

// load fonts
window.WebFontConfig = {
  custom: {
      families: ['KaTeX_AMS', 'KaTeX_Caligraphic:n4,n7', 'KaTeX_Fraktur:n4,n7',
          'KaTeX_Main:n4,n7,i4,i7', 'KaTeX_Math:i4,i7', 'KaTeX_Script',
          'KaTeX_SansSerif:n4,n7,i4', 'KaTeX_Size1', 'KaTeX_Size2', 'KaTeX_Size3',
          'KaTeX_Size4', 'KaTeX_Typewriter'],
  },
};

(function() {
  // From: livebook/assets/js/cell/markdown.js:87
  // https://github.com/elixir-nx/livebook/blob/467d724bfb51faa71d343090cc0d3132541eb9e7/assets/js/cell/markdown.js#L87
  function __renderMathInString(string) {
      return string.replace(
        /(\${1,2})([\s\S]*?)\1/g,
        (match, delimiter, math) => {
          const displayMode = delimiter === "$$";

          return katex.renderToString(math.trim(), {
            displayMode,
            throwOnError: false,
            errorColor: "inherit",
          });
        }
      );
    }

  var bodyContent = document.getElementsByTagName("body");

  for (bodyItem of bodyContent) {
      // There's probably a better way to do this:
      bodyItem.innerHTML = __renderMathInString(bodyItem.innerHTML);
  }
})();
