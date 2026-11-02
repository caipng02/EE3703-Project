"use strict";

const $ = (id) => document.getElementById(id);
const labels = { drill: "Drill", scissor: "Scissors", screwdriver: "Screwdriver" };
const colors = { drill: "#277d61", scissor: "#b97b20", screwdriver: "#647bb9" };
const modelNames = { yolo: "YOLO", faster_rcnn: "Faster R-CNN" };
const state = { image: null, data: null, fileName: "", detections: null, busy: false,
  view: "original", resultModel: null, models: null, requestId: 0 };
const canvas = $("imageCanvas");
const ctx = canvas.getContext("2d");

function selectedModel() {
  return document.querySelector('input[name="model"]:checked').value;
}

function announce(message) { $("announcement").textContent = message; }
function showError(message = "") {
  $("errorMessage").textContent = message;
  $("errorMessage").hidden = !message;
}

function syncControls() {
  $("detectButton").disabled = !state.image || state.busy;
  $("detectLabel").textContent = state.busy ? "Detecting…" : "Run detection";
  $("uploadButton").disabled = state.busy;
  $("clearButton").disabled = state.busy;
  $("fileInput").disabled = state.busy;
  document.querySelectorAll('input[name="model"]').forEach((radio) => { radio.disabled = state.busy; });
  $("detectionsTab").disabled = state.detections === null || state.busy;
  $("originalTab").disabled = state.busy;
  $("loadingOverlay").hidden = !state.busy;
  $("imageStage").setAttribute("aria-busy", String(state.busy));
  $("originalTab").classList.toggle("active", state.view === "original");
  $("detectionsTab").classList.toggle("active", state.view === "detections");
  $("originalTab").setAttribute("aria-pressed", String(state.view === "original"));
  $("detectionsTab").setAttribute("aria-pressed", String(state.view === "detections"));
}

function resetResults() {
  state.detections = null;
  state.resultModel = null;
  state.view = "original";
  $("resultCount").textContent = "—";
  $("resultsBody").replaceChildren();
  const hint = document.createElement("p");
  hint.className = "results-placeholder";
  hint.textContent = "Run a detector to see the tools, confidence scores and inference time.";
  $("resultsBody").append(hint);
  $("viewMeta").textContent = state.image ? "Original image" : "Ready when you are";
  syncControls();
}

function drawImage(annotated = state.view === "detections") {
  if (!state.image) return;
  canvas.width = state.image.naturalWidth;
  canvas.height = state.image.naturalHeight;
  ctx.drawImage(state.image, 0, 0);
  if (!annotated || state.detections === null) return;
  const fontSize = Math.max(12, Math.round(Math.max(canvas.width, canvas.height) / 65));
  const lineWidth = Math.max(2, Math.round(fontSize / 6));
  const pad = Math.round(fontSize / 3);
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
  ctx.textBaseline = "top";
  for (const detection of state.detections) {
    const [x1, y1, x2, y2] = detection.box;
    const text = `${labels[detection.label]}  ${(detection.confidence * 100).toFixed(1)}%`;
    const color = colors[detection.label];
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    // Keep the stroke inside the image at its edges.
    ctx.strokeRect(Math.max(lineWidth / 2, x1), Math.max(lineWidth / 2, y1),
      Math.max(0, Math.min(canvas.width - lineWidth / 2, x2) - Math.max(lineWidth / 2, x1)),
      Math.max(0, Math.min(canvas.height - lineWidth / 2, y2) - Math.max(lineWidth / 2, y1)));
    const tagWidth = Math.min(canvas.width, ctx.measureText(text).width + pad * 2);
    const tagHeight = fontSize + pad * 2;
    const tagX = Math.max(0, Math.min(x1, canvas.width - tagWidth));
    const tagY = Math.max(0, Math.min(y1 >= tagHeight ? y1 - tagHeight : y1, canvas.height - tagHeight));
    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, tagX + pad, tagY + pad, Math.max(1, tagWidth - pad * 2));
  }
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("This file could not be read. Try another image."));
    reader.readAsDataURL(file);
  });
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("This image could not be opened. Try a JPG, PNG or WebP file."));
    image.src = url;
  });
}

async function chooseFile(file) {
  if (!file || state.busy) return;
  // Invalidate the old image while reading a replacement, including its results.
  clearImage();
  const requestId = state.requestId;
  try {
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) throw new Error("Choose a JPG, PNG or WebP image.");
    if (file.size > 12 * 1024 * 1024) throw new Error("Please choose an image smaller than 12 MB.");
    const data = await readFile(file);
    const image = await loadImage(data);
    if (requestId !== state.requestId) return;
    if (image.naturalWidth * image.naturalHeight > 24_000_000) throw new Error("Please choose an image with at most 24 megapixels.");
    state.image = image;
    state.data = data.split(",", 2)[1];
    state.fileName = file.name;
    $("fileName").textContent = file.name;
    $("fileDetails").textContent = `${image.naturalWidth} × ${image.naturalHeight} px · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    $("fileInfo").hidden = false;
    $("emptyState").hidden = true;
    canvas.hidden = false;
    $("imageMeta").textContent = `${image.naturalWidth} × ${image.naturalHeight} px`;
    $("uploadTitle").textContent = "Try another perspective";
    $("uploadButton").firstChild.textContent = "Change image ";
    $("viewMeta").textContent = "Original image";
    drawImage(false);
    announce(`${file.name} loaded. Select a detector and run detection.`);
  } catch (error) {
    if (requestId !== state.requestId) return;
    clearImage(false);
    showError(error.message);
  }
  syncControls();
}

function clearImage(clearError = true) {
  ++state.requestId;
  state.image = null;
  state.data = null;
  state.fileName = "";
  $("fileInput").value = "";
  $("fileInfo").hidden = true;
  $("emptyState").hidden = false;
  canvas.hidden = true;
  canvas.width = 0;
  canvas.height = 0;
  $("imageMeta").textContent = "No image selected";
  $("uploadTitle").textContent = "Drop your image here";
  $("uploadButton").firstChild.textContent = "Upload image ";
  resetResults();
  if (clearError) showError();
}

async function refreshModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("Connection failed");
    const data = await response.json();
    state.models = data.models;
    for (const model of data.models) {
      const element = $(`status-${model.id}`);
      if (!element) continue;
      element.textContent = model.ready ? "Connected" : "Not connected yet";
      element.title = model.message;
      element.classList.toggle("ready", model.ready);
    }
  } catch {
    state.models = null;
    for (const key of Object.keys(modelNames)) {
      $(`status-${key}`).textContent = "Connection unavailable";
      $(`status-${key}`).classList.remove("ready");
    }
  }
}

function renderResults(result) {
  $("resultCount").textContent = result.detections.length;
  const body = $("resultsBody");
  body.replaceChildren();
  const summary = document.createElement("div");
  summary.className = "results-summary";
  const model = document.createElement("strong");
  model.textContent = modelNames[result.model];
  const time = document.createElement("span");
  time.textContent = `· ${result.inference_ms.toLocaleString()} ms inference`;
  const count = document.createElement("span");
  count.textContent = result.detections.length ? `· ${result.detections.length} tool${result.detections.length === 1 ? "" : "s"} detected` : "· No tools detected in this image";
  summary.append(model, time, count);
  body.append(summary);
  const chips = document.createElement("div");
  chips.className = "result-chips";
  for (const detection of result.detections) {
    const chip = document.createElement("span");
    chip.className = "detection-chip";
    const dot = document.createElement("i");
    dot.className = `${detection.label}-dot`;
    const name = document.createElement("span");
    name.textContent = labels[detection.label];
    const confidence = document.createElement("strong");
    confidence.textContent = `${(detection.confidence * 100).toFixed(1)}%`;
    chip.append(dot, name, confidence);
    chips.append(chip);
  }
  body.append(chips);
  $("viewMeta").textContent = `${modelNames[result.model]} · ${result.detections.length} detected`;
}

async function runDetection() {
  if (!state.image || state.busy) return;
  showError();
  const model = selectedModel();
  resetResults();
  drawImage(false);
  state.busy = true;
  syncControls();
  $("loadingText").textContent = `${modelNames[model]} is finding your tools…`;
  announce("Detection started.");
  try {
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: state.data, model }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Detection failed. Please try again.");
    state.image = await loadImage(`data:image/png;base64,${result.image}`);
    state.detections = result.detections;
    state.resultModel = result.model;
    state.view = "detections";
    renderResults(result);
    drawImage();
    announce(`Detection complete. ${result.detections.length} tools detected with ${modelNames[model]}.`);
  } catch (error) {
    const message = error instanceof TypeError ? "Cannot reach the local app. Check that it is still running." : error.message;
    showError(message);
    announce(message);
  } finally {
    state.busy = false;
    syncControls();
    refreshModels();
  }
}

$("uploadButton").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (event) => {
  chooseFile(event.target.files[0]);
  event.target.value = "";
});
$("clearButton").addEventListener("click", () => clearImage());
$("detectButton").addEventListener("click", runDetection);
document.querySelectorAll('input[name="model"]').forEach((radio) => radio.addEventListener("change", () => {
  resetResults(); showError(); drawImage(false);
}));
for (const view of ["original", "detections"]) {
  $(`${view}Tab`).addEventListener("click", () => {
    state.view = view;
    drawImage();
    syncControls();
    $("viewMeta").textContent = view === "original" ? "Original image"
      : `${modelNames[state.resultModel]} · ${state.detections.length} detected`;
  });
}
for (const eventName of ["dragenter", "dragover"]) {
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!state.busy) $("dropzone").classList.add("dragover");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("dropzone").classList.remove("dragover");
    if (eventName === "drop") chooseFile(event.dataTransfer.files[0]);
  });
}
// Prevent an accidental drop outside the upload target from replacing the page.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
refreshModels();
syncControls();
