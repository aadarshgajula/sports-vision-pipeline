let state = {
  jobId: null,
  frameCount: 0,
  keypoints: [],       // [{id, name, x, y}]
  usedKeypointIds: new Set(),
  correspondences: [],  // [{pixel:[x,y], pitch:[x,y], name}]
  imageNaturalWidth: 0,
  imageNaturalHeight: 0,
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
document.getElementById("upload-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("file-input");
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

  const select = document.getElementById("keypoint-select");
  if (!select.value) { alert("Pick a keypoint from the dropdown first"); return; }
  const kp = state.keypoints.find(k => k.id === parseInt(select.value, 10));

  state.correspondences.push({ pixel: [x, y], pitch: [kp.x, kp.y], name: kp.name });
  state.usedKeypointIds.add(kp.id);
  refreshKeypointDropdown();
  refreshPointList();
  redrawCanvas();
});

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
  document.getElementById("save-calibration-btn").disabled = state.correspondences.length < 4;
}

document.getElementById("save-calibration-btn").addEventListener("click", async () => {
  const feedback = document.getElementById("calibration-feedback");
  feedback.textContent = "Saving…";
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
  state = { jobId: null, frameCount: 0, keypoints: [], usedKeypointIds: new Set(), correspondences: [] };
  document.getElementById("file-input").value = "";
  document.getElementById("point-list").innerHTML = "";
  document.getElementById("point-count").textContent = "0";
  document.getElementById("save-calibration-btn").disabled = true;
  document.getElementById("calibration-feedback").textContent = "";
  showScreen("screen-upload");
});
