import { createJsonEditor } from "./json-editor.js";
import { templates } from "./templates.js";
import { renderResults, initModal } from "./results.js";
import { initTheme } from "./theme.js";

initTheme();

const questionEditor = createJsonEditor(document.getElementById("question"));
const stateEditor = createJsonEditor(document.getElementById("state"));

function loadTemplate(name) {
  const t = templates[name] || templates.choice;
  questionEditor.value = JSON.stringify(t.question, null, 2);
  stateEditor.value = JSON.stringify(t.state, null, 2);
}

document.getElementById("type").addEventListener("change", (e) => loadTemplate(e.target.value));
loadTemplate("choice");

const dot = document.getElementById("dot");
const statusText = document.getElementById("statusText");
const output = document.getElementById("output");
const runBtn = document.getElementById("run");

async function checkHealth() {
  dot.className = "dot";
  statusText.textContent = "checking model…";
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      document.getElementById("modelId").textContent = data.model_id;
      dot.className = "dot ok";
      const dev = data.device + (data.device !== "cpu" ? " / " + data.gpu : "");
      statusText.textContent = (data.model_loaded ? "model loaded" : "loading on first run") + " · " + dev;
    } else {
      dot.className = "dot err";
      statusText.textContent = "unavailable";
    }
  } catch (e) {
    dot.className = "dot err";
    statusText.textContent = "health check failed";
  }
}

function parseEditor(editor, label) {
  try {
    return JSON.parse(editor.value);
  } catch (e) {
    throw new Error(label + " is not valid JSON: " + e.message);
  }
}

initModal();

runBtn.addEventListener("click", async () => {
  output.innerHTML = '<span class="spinner">Classifying… (the first run loads the model)</span>';
  runBtn.disabled = true;
  try {
    const question = parseEditor(questionEditor, "Questions");
    const state = parseEditor(stateEditor, "State");
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ questions: question, state }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "HTTP " + res.status);
    output.innerHTML = "";
    output.appendChild(renderResults(data, question));
    checkHealth();
  } catch (e) {
    output.innerHTML = '<div class="error"></div>';
    output.firstChild.textContent = String(e.message || e);
  } finally {
    runBtn.disabled = false;
  }
});

checkHealth();
