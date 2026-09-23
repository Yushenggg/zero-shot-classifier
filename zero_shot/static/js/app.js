import { createJsonEditor } from "./json-editor.js";
import { templates } from "./templates.js";
import { renderResults, initModal, openLightbox } from "./results.js";
import { initTheme } from "./theme.js";

initTheme();

const questionEditor = createJsonEditor(document.getElementById("question"));
const stateEditor = createJsonEditor(document.getElementById("state"));
const exampleSelect = document.getElementById("type");

function loadTemplate(name) {
  const t = templates[name] || templates.choice;
  questionEditor.value = JSON.stringify(t.question, null, 2);
  stateEditor.value = JSON.stringify(t.state ?? {}, null, 2);
  setMode(t.mode === "image" ? "image" : "text");
}

exampleSelect.addEventListener("change", (e) => loadTemplate(e.target.value));

const dot = document.getElementById("dot");
const statusText = document.getElementById("statusText");
const output = document.getElementById("output");
const runBtn = document.getElementById("run");
const modeToggle = document.getElementById("modeToggle");
const modeToggleLabel = document.getElementById("modeToggleLabel");
const imagePanel = document.getElementById("imagePanel");
const statePanel = document.getElementById("statePanel");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const imageNote = document.getElementById("imageNote");
const imageClear = document.getElementById("imageClear");

let mode = "text";
let imageFile = null;

function setMode(next) {
  if (next === "image" && modeToggle.disabled) return;
  mode = next;
  modeToggle.checked = next === "image";
  modeToggleLabel.dataset.mode = next;
  imagePanel.hidden = next !== "image";
  statePanel.hidden = next === "image";
}

function setVisionAvailable(available) {
  const disabled = available === false;
  modeToggle.disabled = disabled;
  modeToggleLabel.classList.toggle("disabled", disabled);
  modeToggleLabel.title = disabled
    ? "The configured model is text-only. Serve a vision model (e.g. config.toml or config.cpu.toml) for image input."
    : "";
  const imageOption = exampleSelect.querySelector('option[value="image"]');
  if (imageOption) imageOption.disabled = disabled;
  if (disabled && mode === "image") setMode("text");
  if (disabled && exampleSelect.value === "image") {
    exampleSelect.value = "choice";
    loadTemplate("choice");
  }
}

modeToggle.addEventListener("change", () => {
  setMode(modeToggle.checked ? "image" : "text");
});
loadTemplate("choice");

function clearImage() {
  if (imagePreview.src) URL.revokeObjectURL(imagePreview.src);
  imageFile = null;
  imageInput.value = "";
  imagePreview.hidden = true;
  imagePreview.removeAttribute("src");
  imageNote.textContent = "Attach an image to classify it with a vision-language model.";
}

imageInput.addEventListener("change", () => {
  imageFile = imageInput.files && imageInput.files[0] ? imageInput.files[0] : null;
  if (!imageFile) return clearImage();
  imagePreview.src = URL.createObjectURL(imageFile);
  imagePreview.hidden = false;
  imageNote.textContent = imageFile.name + " · sent to /v1/classify/image";
});

imageClear.addEventListener("click", clearImage);

imagePreview.addEventListener("click", () => {
  if (imageFile) openLightbox(imagePreview.src, imageFile.name);
});

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
      setVisionAvailable(data.multimodal);
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
    let res;
    if (mode === "image") {
      if (!imageFile) throw new Error("Attach an image first (or switch to Text mode).");
      const form = new FormData();
      form.append("file", imageFile);
      form.append("questions", JSON.stringify(question));
      res = await fetch("/v1/classify/image", { method: "POST", body: form });
    } else {
      const state = parseEditor(stateEditor, "State");
      res = await fetch("/v1/classify/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ questions: question, state }),
      });
    }
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
