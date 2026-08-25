const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];

const ui = {
  state: null,
  trail: [],
  camera: { yaw: -0.7, pitch: 0.55, zoom: 900, target: [0, 0, 0] },
  dragging: false,
  lastPointer: null,
};

const elements = {
  robotStatus: document.querySelector("#robotStatus"),
  phoneStatus: document.querySelector("#phoneStatus"),
  loopMetric: document.querySelector("#loopMetric"),
  errorMetric: document.querySelector("#errorMetric"),
  voltageMetric: document.querySelector("#voltageMetric"),
  currentMetric: document.querySelector("#currentMetric"),
  robotName: document.querySelector("#robotName"),
  orientationMode: document.querySelector("#orientationMode"),
  mappingNote: document.querySelector("#mappingNote"),
  controlledPose: document.querySelector("#controlledPose"),
  translationGain: document.querySelector("#translationGain"),
  cartesianStep: document.querySelector("#cartesianStep"),
  gripperSpeed: document.querySelector("#gripperSpeed"),
  canvas: document.querySelector("#robotCanvas"),
  baseButton: document.querySelector("#baseButton"),
  controlMessage: document.querySelector("#controlMessage"),
  phaseLabel: document.querySelector("#phaseLabel"),
  motorRows: document.querySelector("#motorRows"),
};

function setStatus(element, text, className) {
  element.className = `status ${className}`;
  element.innerHTML = `<span></span>${text}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function liveMetrics(state) {
  const electrical = Object.values(state.electrical || {});
  const minimumVoltage = electrical.length ? Math.min(...electrical.map((item) => Number(item.voltage_v))) : null;
  const totalCurrent = electrical.length ? electrical.reduce((sum, item) => sum + Number(item.current_ma), 0) : null;
  elements.loopMetric.textContent = state.loop_ms == null ? "—" : `${Number(state.loop_ms).toFixed(1)} ms`;
  elements.errorMetric.textContent = state.cartesian?.error_m == null ? "—" : `${(state.cartesian.error_m * 1000).toFixed(1)} mm`;
  elements.voltageMetric.textContent = minimumVoltage == null ? "—" : `${minimumVoltage.toFixed(1)} V`;
  elements.currentMetric.textContent = totalCurrent == null ? "—" : `${totalCurrent.toFixed(0)} mA`;
}

function renderTable(state) {
  elements.motorRows.innerHTML = JOINTS.map((joint) => {
    const electrical = state.electrical?.[joint] || {};
    const actual = state.positions?.[joint];
    const command = state.commands?.[joint];
    const unit = joint === "gripper" ? "%" : "°";
    return `<tr>
      <td>${joint.replaceAll("_", " ")}</td>
      <td>${actual == null ? "—" : `${Number(actual).toFixed(1)}${unit}`}</td>
      <td>${command == null ? "—" : `${Number(command).toFixed(1)}${unit}`}</td>
      <td>${electrical.voltage_v == null ? "—" : `${Number(electrical.voltage_v).toFixed(1)} V`}</td>
      <td>${electrical.current_ma == null ? "—" : `${Number(electrical.current_ma).toFixed(0)} mA`}</td>
      <td>${electrical.load_percent == null ? "—" : `${Number(electrical.load_percent).toFixed(1)}%`}</td>
      <td>${electrical.temperature_c == null ? "—" : `${electrical.temperature_c}°C`}</td>
    </tr>`;
  }).join("");
}

function rotateAndProject(point, width, height) {
  const x = point[0] - ui.camera.target[0];
  const y = point[1] - ui.camera.target[1];
  const z = point[2] - ui.camera.target[2];
  const cy = Math.cos(ui.camera.yaw), sy = Math.sin(ui.camera.yaw);
  const cp = Math.cos(ui.camera.pitch), sp = Math.sin(ui.camera.pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const y2 = cp * y1 - sp * z;
  const depth = sp * y1 + cp * z;
  const perspective = 1 / Math.max(0.55, 1 + depth * 0.7);
  return [width / 2 + x1 * ui.camera.zoom * perspective, height / 2 - y2 * ui.camera.zoom * perspective, depth];
}

function centerCameraOnRobot(links) {
  const points = Object.values(links);
  if (!points.length) return;
  ui.camera.target = [0, 1, 2].map((axis) => {
    const values = points.map((point) => Number(point[axis]));
    return (Math.min(...values) + Math.max(...values)) / 2;
  });
}

function drawGrid(ctx, width, height) {
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(120,150,130,.14)";
  for (let index = -5; index <= 5; index += 1) {
    const offset = index / 10;
    for (const pair of [[[offset, -.5, 0], [offset, .5, 0]], [[-.5, offset, 0], [.5, offset, 0]]]) {
      const a = rotateAndProject(pair[0], width, height);
      const b = rotateAndProject(pair[1], width, height);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }
  ctx.restore();
}

function drawRobot(ctx, width, height, links, edges, color, lineWidth, dashed = false) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.setLineDash(dashed ? [8, 7] : []);
  const segments = edges.map((edge) => {
    const parent = links[edge.parent], child = links[edge.child];
    if (!parent || !child) return null;
    const a = rotateAndProject(parent, width, height), b = rotateAndProject(child, width, height);
    return { a, b, depth: (a[2] + b[2]) / 2 };
  }).filter(Boolean).sort((a, b) => a.depth - b.depth);
  segments.forEach(({ a, b }) => { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); });
  if (!dashed) {
    Object.values(links).forEach((point) => {
      const projected = rotateAndProject(point, width, height);
      ctx.beginPath(); ctx.arc(projected[0], projected[1], 4.2, 0, Math.PI * 2); ctx.fill();
    });
  }
  ctx.restore();
}

function drawTrail(ctx, width, height) {
  if (ui.trail.length < 2) return;
  ctx.save();
  ctx.strokeStyle = "rgba(91,215,231,.7)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ui.trail.forEach((point, index) => {
    const p = rotateAndProject(point, width, height);
    if (index === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
  });
  ctx.stroke();
  ctx.restore();
}

function renderRobot(state) {
  const canvas = elements.canvas;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const robot = state.robot || {};
  centerCameraOnRobot(robot.actual_links_m || {});
  drawGrid(ctx, width, height);
  drawTrail(ctx, width, height);
  drawRobot(ctx, width, height, robot.target_links_m || {}, robot.edges || [], "rgba(255,174,50,.72)", 4, true);
  drawRobot(ctx, width, height, robot.actual_links_m || {}, robot.edges || [], "rgba(57,220,133,.96)", 7, false);
}

function updateState(state) {
  ui.state = state;
  const mapping = state.control_mapping || {};
  const orientationEnabled = Boolean(mapping.orientation_enabled);
  setStatus(elements.robotStatus, state.connected ? "Robot connected" : "Offline", state.connected ? "online" : "offline");
  setStatus(elements.phoneStatus, state.phone_enabled ? "Phone commanding" : "Phone released", state.phone_enabled ? "active" : "neutral");
  elements.robotName.textContent = state.robot?.name || "SO-101";
  elements.orientationMode.textContent = orientationEnabled ? "6-DOF" : "XYZ ONLY";
  elements.controlledPose.textContent = orientationEnabled ? "XYZ + roll, pitch, yaw" : "XYZ (orientation locked)";
  elements.mappingNote.textContent = orientationEnabled
    ? "Phone translation and rotation are active. Set ENABLE_PHONE_ORIENTATION to False for XYZ-only control."
    : "Phone roll, pitch, and yaw are ignored. Set ENABLE_PHONE_ORIENTATION to True to restore six-DoF control.";
  elements.translationGain.textContent = mapping.translation_gain == null ? "—" : Number(mapping.translation_gain).toFixed(2);
  elements.cartesianStep.textContent = mapping.max_ee_step_m == null ? "—" : `${Number(mapping.max_ee_step_m).toFixed(2)} m`;
  elements.gripperSpeed.textContent = mapping.gripper_speed_factor == null ? "—" : Number(mapping.gripper_speed_factor).toFixed(0);
  elements.phaseLabel.textContent = state.phase || "unknown";
  elements.baseButton.disabled = !state.connected || state.return_base_pending;
  const actualEE = state.cartesian?.actual_position_m;
  if (Array.isArray(actualEE) && state.connected) {
    const last = ui.trail.at(-1);
    if (!last || actualEE.some((value, index) => Math.abs(value - last[index]) > 0.0002)) {
      ui.trail.push(actualEE);
      if (ui.trail.length > 400) ui.trail.shift();
    }
  } else if (!state.connected) ui.trail = [];
  liveMetrics(state);
  renderTable(state);
  renderRobot(state);
}

elements.baseButton.addEventListener("click", async () => {
  elements.controlMessage.textContent = "Base return queued. Release Hold to move.";
  elements.controlMessage.className = "control-message";
  elements.controlMessage.dataset.kind = "base";
  try { updateState(await api("/api/return-to-base", { method: "POST" })); }
  catch (error) { elements.controlMessage.textContent = error.message; elements.controlMessage.className = "control-message error"; }
});

elements.canvas.addEventListener("pointerdown", (event) => { ui.dragging = true; ui.lastPointer = [event.clientX, event.clientY]; elements.canvas.setPointerCapture(event.pointerId); });
elements.canvas.addEventListener("pointermove", (event) => {
  if (!ui.dragging) return;
  const dx = event.clientX - ui.lastPointer[0], dy = event.clientY - ui.lastPointer[1];
  ui.camera.yaw += dx * 0.008;
  ui.camera.pitch = Math.max(-1.35, Math.min(1.35, ui.camera.pitch + dy * 0.008));
  ui.lastPointer = [event.clientX, event.clientY];
  if (ui.state) renderRobot(ui.state);
});
elements.canvas.addEventListener("pointerup", () => { ui.dragging = false; });
elements.canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  ui.camera.zoom = Math.max(350, Math.min(1800, ui.camera.zoom * Math.exp(-event.deltaY * 0.001)));
  if (ui.state) renderRobot(ui.state);
}, { passive: false });

async function refresh() {
  try { updateState(await api("/api/state")); }
  catch (error) {
    elements.controlMessage.textContent = error.message;
    elements.controlMessage.className = "control-message error";
  }
}

refresh();
window.setInterval(refresh, 200);
window.addEventListener("resize", () => ui.state && renderRobot(ui.state));
