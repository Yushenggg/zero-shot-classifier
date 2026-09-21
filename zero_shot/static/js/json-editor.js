// Reusable editable JSON viewer: syntax-highlighted <pre> behind a transparent
// <textarea>, with synced scrolling and Tab-to-indent.

function highlightJson(text) {
  const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return escaped.replace(
    /("(?:\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = "j-num";
      if (/^"/.test(match)) cls = /:\s*$/.test(match) ? "j-key" : "j-str";
      else if (/^(true|false)$/.test(match)) cls = "j-bool";
      else if (match === "null") cls = "j-null";
      return '<span class="' + cls + '">' + match + "</span>";
    }
  );
}

export function createJsonEditor(root, initial = "") {
  root.classList.add("json-editor");
  const gutter = document.createElement("div");
  gutter.className = "json-gutter";
  const pre = document.createElement("pre");
  pre.className = "json-highlight";
  const code = document.createElement("code");
  pre.appendChild(code);
  const input = document.createElement("textarea");
  input.className = "json-input";
  input.spellcheck = false;
  input.setAttribute("autocapitalize", "off");
  input.setAttribute("autocomplete", "off");
  root.append(gutter, pre, input);

  const render = () => {
    code.innerHTML = highlightJson(input.value) + "\n";
    const lines = input.value.split("\n").length;
    let numbers = "";
    for (let i = 1; i <= lines; i++) numbers += i + "\n";
    gutter.textContent = numbers;
  };
  const syncScroll = () => {
    pre.scrollTop = input.scrollTop;
    pre.scrollLeft = input.scrollLeft;
    gutter.scrollTop = input.scrollTop;
  };
  input.addEventListener("input", render);
  input.addEventListener("scroll", syncScroll);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const start = input.selectionStart;
      const end = input.selectionEnd;
      input.value = input.value.slice(0, start) + "  " + input.value.slice(end);
      input.selectionStart = input.selectionEnd = start + 2;
      render();
    }
  });

  const editor = {
    get value() {
      return input.value;
    },
    set value(v) {
      input.value = v;
      render();
      syncScroll();
    },
    getValue() {
      return input.value;
    },
    setValue(v) {
      this.value = v;
    },
    render,
  };
  editor.value = initial;
  return editor;
}
