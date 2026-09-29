const ANCHOR_LABELS = { "0": "Team A", "1": "Team B", "referee": "Referee" };

let state = {
  jobId: null,
  frameCount: 0,
  keypoints: [],       // [{id, name, x, y}]
  usedKeypointIds: new Set(),
  correspondences: [],  // [{pixel:[x,y], pitch:[x,y], name}]
  imageNaturalWidth: 0,
  imageNaturalHeight: 0,
  anchorPickMode: null,   // null | "0" | "1" | "referee"
  currentFrameIndex: 0,
  teamAnchors: {},        // {"0": {pixel:[x,y], frame_index}, "1": {...}, "referee": {...}}
};

function showScreen(id) {
  document.querySelectorAll(".screen").forEach(el => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

async function loadKeypoints() {
  const res = await fetch("/api/keypoints");
  state.keypoints = await res.json();
  refreshKeypointDropdown();
}

function refreshKeypointDropdown() {
  const select = document.getElementById("keypoint-select");
  select.innerHTML = "";
  state.keypoints
    .filter(kp => !state.usedKeypointIds.has(kp.id))
    .forEach(kp => {
      const opt = document.createElement("option");
      opt.value = kp.id;
      opt.textContent = `${kp.name} (${kp.x}, ${kp.y})`;
      select.appendChild(opt);
    });
}

// --- Upload ---
const dropzone = document.getElementById("dropzone");
const dropzoneFilename = document.getElementById("dropzone-filename");
const fileInput = document.getElementById("file-input");

function showSelectedFile(file) {
  if (!file) {
    dropzoneFilename.textContent = "No file selected";
    dropzoneFilename.classList.remove("has-file");
    return;
  }
  dropzoneFilename.textContent = file.name;
  dropzoneFilename.classList.add("has-file");
}

fileInput.addEventListener("change", () => showSelectedFile(fileInput.files[0]));

["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "dragend"].forEach(evt => {
  dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"));
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showSelectedFile(fileInput.files[0]);
  }
});

document.getElementById("upload-btn").addEventListener("click", async () => {
  if (!fileInput.files.length) { alert("Choose a video file first"); return; }
  document.getElementById("upload-status").textContent = "Uploading…";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  if (!res.ok) {
    document.getElementById("upload-status").textContent = "Upload failed: " + (await res.text());
    return;
  }
  const data = await res.json();
  state.jobId = data.job_id;
  state.frameCount = data.frame_count;

  await loadKeypoints();
  const slider = document.getElementById("frame-slider");
  slider.max = state.frameCount - 1;
  slider.value = 0;
  loadFrame(0);
  showScreen("screen-calibrate");
});

// --- Calibration ---
function loadFrame(index) {
  const canvas = document.getElementById("calibrate-canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    state.imageNaturalWidth = img.naturalWidth;
    state.imageNaturalHeight = img.naturalHeight;
    redrawCanvas(img);
  };
  img.src = `/api/jobs/${state.jobId}/frame/${index}?t=${Date.now()}`;
  document.getElementById("frame-label").textContent = `frame ${index} / ${state.frameCount - 1}`;
  state.currentFrameIndex = index;
}

let currentImage = null;
function redrawCanvas(img) {
  if (img) currentImage = img;
  const canvas = document.getElementById("calibrate-canvas");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(currentImage, 0, 0);
  ctx.font = "16px sans-serif";
  state.correspondences.forEach(c => {
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(c.pixel[0], c.pixel[1], 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText(c.name, c.pixel[0] + 8, c.pixel[1] - 8);
  });
  Object.entries(state.teamAnchors).forEach(([label, a]) => {
    if (a.frame_index !== state.currentFrameIndex) return;  // only draw anchors picked on this frame
    ctx.fillStyle = "#f59e0b";
    const [x, y] = a.pixel;
    ctx.fillRect(x - 7, y - 7, 14, 14);
    ctx.fillText(ANCHOR_LABELS[label], x + 10, y + 4);
  });
}

document.getElementById("frame-slider").addEventListener("input", (e) => {
  loadFrame(parseInt(e.target.value, 10));
});

document.getElementById("calibrate-canvas").addEventListener("click", (e) => {
  const canvas = document.getElementById("calibrate-canvas");
  const rect = canvas.getBoundingClientRect();
  const scaleX = state.imageNaturalWidth / rect.width;
  const scaleY = state.imageNaturalHeight / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  if (state.anchorPickMode) {
    state.teamAnchors[state.anchorPickMode] = { pixel: [x, y], frame_index: state.currentFrameIndex };
    state.anchorPickMode = null;
    refreshAnchorList();
    redrawCanvas();
    return;
  }

  const select = document.getElementById("keypoint-select");
  if (!select.value) { alert("Pick a keypoint from the dropdown first"); return; }
  const kp = state.keypoints.find(k => k.id === parseInt(select.value, 10));

  state.correspondences.push({ pixel: [x, y], pitch: [kp.x, kp.y], name: kp.name });
  state.usedKeypointIds.add(kp.id);
  refreshKeypointDropdown();
  refreshPointList();
  redrawCanvas();
});

// --- Team anchors ---
function armAnchorPick(label) {
  state.anchorPickMode = label;
}
document.getElementById("pick-team0-btn").addEventListener("click", () => armAnchorPick("0"));
document.getElementById("pick-team1-btn").addEventListener("click", () => armAnchorPick("1"));
document.getElementById("pick-referee-btn").addEventListener("click", () => armAnchorPick("referee"));

function refreshAnchorList() {
  const list = document.getElementById("anchor-list");
  list.innerHTML = "";
  Object.entries(state.teamAnchors).forEach(([label, a]) => {
    const row = document.createElement("div");
    row.innerHTML = `<span>${ANCHOR_LABELS[label]} — frame ${a.frame_index}, (${a.pixel[0].toFixed(0)}, ${a.pixel[1].toFixed(0)})</span>`;
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "remove";
    removeBtn.className = "secondary";
    removeBtn.onclick = () => {
      delete state.teamAnchors[label];
      refreshAnchorList();
      redrawCanvas();
    };
    row.appendChild(removeBtn);
    list.appendChild(row);
  });
  updateSaveButtonState();
}

function updateSaveButtonState() {
  const hasTeams = "0" in state.teamAnchors && "1" in state.teamAnchors;
  document.getElementById("save-calibration-btn").disabled = state.correspondences.length < 4 || !hasTeams;
}

function refreshPointList() {
  document.getElementById("point-count").textContent = state.correspondences.length;
  const list = document.getElementById("point-list");
  list.innerHTML = "";
  state.correspondences.forEach((c, i) => {
    const row = document.createElement("div");
    row.innerHTML = `<span>${c.name} — (${c.pixel[0].toFixed(0)}, ${c.pixel[1].toFixed(0)})</span>`;
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "remove";
    removeBtn.className = "secondary";
    removeBtn.onclick = () => {
      state.usedKeypointIds.delete(state.keypoints.find(k => k.name === c.name).id);
      state.correspondences.splice(i, 1);
      refreshKeypointDropdown();
      refreshPointList();
      redrawCanvas();
    };
    row.appendChild(removeBtn);
    list.appendChild(row);
  });
  updateSaveButtonState();
}

document.getElementById("save-calibration-btn").addEventListener("click", async () => {
  const feedback = document.getElementById("calibration-feedback");
  feedback.textContent = "Saving calibration…";
  const res = await fetch(`/api/jobs/${state.jobId}/calibrate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.correspondences),
  });
  if (!res.ok) {
    feedback.textContent = "Failed: " + (await res.text());
    return;
  }
  const quality = await res.json();

  feedback.textContent = "Saving team examples…";
  const anchorPoints = Object.entries(state.teamAnchors).map(([label, a]) => ({
    label, pixel: a.pixel, frame_index: a.frame_index,
  }));
  const anchorRes = await fetch(`/api/jobs/${state.jobId}/team_anchors`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(anchorPoints),
  });
  if (!anchorRes.ok) {
    feedback.textContent = "Failed: " + (await anchorRes.text());
    return;
  }

  feedback.textContent = `Calibrated — mean error ${quality.mean_error_m.toFixed(2)}m, max ${quality.max_error_m.toFixed(2)}m`;
  await fetch(`/api/jobs/${state.jobId}/run`, { method: "POST" });
  showScreen("screen-processing");
  pollStatus();
});

// --- Processing / polling ---
const STAGE_LABELS = {
  running_pipeline: "Detecting players, tracking, classifying teams, tracking ball…",
  running_analytics: "Generating heatmaps and passing network…",
  rendering_topdown: "Rendering 2D top-down view…",
  done: "Done",
  error: "Error",
};

async function pollStatus() {
  const res = await fetch(`/api/jobs/${state.jobId}/status`);
  const data = await res.json();

  document.getElementById("stage-label").textContent = STAGE_LABELS[data.stage] || data.stage;
  document.getElementById("progress-fill").style.width = `${Math.round(data.progress * 100)}%`;

  if (data.stage === "error") {
    document.getElementById("stage-label").textContent = "Error: " + data.error;
    return;
  }
  if (data.stage === "done") {
    showResults(data);
    return;
  }
  setTimeout(pollStatus, 2000);
}

function showResults(data) {
  const jobId = state.jobId;
  document.getElementById("video-annotated").src = `/api/jobs/${jobId}/files/${data.results.annotated_video}`;
  document.getElementById("video-topdown").src = `/api/jobs/${jobId}/files/${data.results.topdown_video}`;
  document.getElementById("img-heatmap-0").src = `/api/jobs/${jobId}/files/${data.results.heatmap_team_0}`;
  document.getElementById("img-heatmap-1").src = `/api/jobs/${jobId}/files/${data.results.heatmap_team_1}`;
  document.getElementById("img-passing").src = `/api/jobs/${jobId}/files/${data.results.passing_network}`;
  document.getElementById("img-formation-0").src = `/api/jobs/${jobId}/files/${data.results.formation_team_0}`;
  document.getElementById("img-formation-1").src = `/api/jobs/${jobId}/files/${data.results.formation_team_1}`;

  (data.formations || []).forEach(f => {
    const el = document.getElementById(`formation-label-${f.team}`);
    if (el) el.textContent = `${f.label} (confidence: ${f.confidence})`;
  });

  const warningsContainer = document.getElementById("warnings-container");
  warningsContainer.innerHTML = "";
  if (data.warnings && data.warnings.length) {
    data.warnings.forEach(w => {
      const div = document.createElement("div");
      div.className = "warning";
      div.textContent = "⚠ " + w;
      warningsContainer.appendChild(div);
    });
    document.getElementById("quality-ok").textContent = "";
  } else {
    document.getElementById("quality-ok").textContent = "✓ No quality issues detected.";
  }

  showScreen("screen-results");
}

document.getElementById("new-video-btn").addEventListener("click", () => {
  state = {
    jobId: null, frameCount: 0, keypoints: [], usedKeypointIds: new Set(), correspondences: [],
    anchorPickMode: null, currentFrameIndex: 0, teamAnchors: {},
  };
  document.getElementById("file-input").value = "";
  showSelectedFile(null);
  document.getElementById("point-list").innerHTML = "";
  document.getElementById("point-count").textContent = "0";
  document.getElementById("anchor-list").innerHTML = "";
  document.getElementById("save-calibration-btn").disabled = true;
  document.getElementById("calibration-feedback").textContent = "";
  showScreen("screen-upload");
});
