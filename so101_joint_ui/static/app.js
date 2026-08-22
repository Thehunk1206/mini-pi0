const JOINT_LABELS = {
  shoulder_pan: "Shoulder pan",
  shoulder_lift: "Shoulder lift",
  elbow_flex: "Elbow flex",
  wrist_flex: "Wrist flex",
  wrist_roll: "Wrist roll",
  gripper: "Gripper",
};

const state = {
  snapshot: null,
  dragging: new Set(),
  sendTimers: new Map(),
  locallyClearedEvents: 0,
  shownError: null,
};

const elements = {
  serialPort: document.querySelector("#serialPort"),
  connectButton: document.querySelector("#connectButton"),
  disconnectButton: document.querySelector("#disconnectButton"),
  torqueButton: document.querySelector("#torqueButton"),
  stopButton: document.querySelector("#stopButton"),
  statusBadge: document.querySelector("#statusBadge"),
  statusText: document.querySelector("#statusText"),
  robotId: document.querySelector("#robotId"),
  controlHint: document.querySelector("#controlHint"),
  jointGrid: document.querySelector("#jointGrid"),
  eventLog: document.querySelector("#eventLog"),
  clearLogButton: document.querySelector("#clearLogButton"),
  toast: document.querySelector("#toast"),
};

function formatValue(value, unit) {
  return `${Number(value ?? 0).toFixed(1)}${unit}`;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("visible"), 3500);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function createJointCards(snapshot) {
  elements.jointGrid.innerHTML = "";
  Object.entries(JOINT_LABELS).forEach(([joint, label], index) => {
    const limit = snapshot.limits[joint];
    const card = document.createElement("article");
    card.className = "joint-card";
    card.style.setProperty("--index", index);
    card.innerHTML = `
      <div class="joint-title-row">
        <div>
          <span class="joint-id">M${index + 1}</span>
          <h3>${label}</h3>
        </div>
        <div class="actual-readout">
          <span>Actual</span>
          <strong data-actual="${joint}">—</strong>
        </div>
      </div>
      <input
        class="joint-slider"
        data-slider="${joint}"
        type="range"
        min="${limit.minimum}"
        max="${limit.maximum}"
        step="0.1"
        disabled
        aria-label="${label} target"
      />
      <div class="slider-meta">
        <span>${formatValue(limit.minimum, limit.unit)}</span>
        <strong data-target="${joint}">Target —</strong>
        <span>${formatValue(limit.maximum, limit.unit)}</span>
      </div>
      <div class="telemetry-row">
        <span>Command <b data-command="${joint}">—</b></span>
        <span>Temp <b data-temp="${joint}">—</b></span>
        <span>Voltage <b data-voltage="${joint}">—</b></span>
      </div>
    `;
    elements.jointGrid.appendChild(card);

    const slider = card.querySelector(`[data-slider="${joint}"]`);
    slider.addEventListener("pointerdown", () => state.dragging.add(joint));
    slider.addEventListener("pointerup", () => state.dragging.delete(joint));
    slider.addEventListener("pointercancel", () => state.dragging.delete(joint));
    slider.addEventListener("input", () => {
      card.querySelector(`[data-target="${joint}"]`).textContent =
        `Target ${formatValue(slider.value, limit.unit)}`;
      scheduleTarget(joint, Number(slider.value));
    });
  });
}

function scheduleTarget(joint, value) {
  window.clearTimeout(state.sendTimers.get(joint));
  state.sendTimers.set(
    joint,
    window.setTimeout(async () => {
      try {
        state.snapshot = await api("/api/target", {
          method: "POST",
          body: JSON.stringify({ joint, value }),
        });
      } catch (error) {
        showToast(error.message, true);
      }
    }, 70),
  );
}

function render(snapshot) {
  if (!state.snapshot) createJointCards(snapshot);
  state.snapshot = snapshot;

  const connected = snapshot.connected;
  const holding = connected && snapshot.torque_enabled;
  elements.serialPort.value ||= snapshot.serial_port || "";
  elements.robotId.textContent = snapshot.robot_id;
  elements.connectButton.disabled = connected;
  elements.disconnectButton.disabled = !connected;
  elements.torqueButton.disabled = !connected;
  elements.stopButton.disabled = !connected;
  elements.torqueButton.textContent = holding ? "Release torque" : "Enable hold";

  elements.statusBadge.className = `status-badge ${holding ? "holding" : connected ? "released" : "offline"}`;
  elements.statusText.textContent = holding ? "Holding" : connected ? "Torque released" : "Offline";
  elements.controlHint.textContent = holding
    ? "Drag a slider; motion is rate-limited and bounded by calibration."
    : connected
      ? "Torque is released. Enable hold to unlock sliders."
      : "Connect the arm to unlock the controls.";

  Object.keys(JOINT_LABELS).forEach((joint) => {
    const limit = snapshot.limits[joint];
    const slider = document.querySelector(`[data-slider="${joint}"]`);
    const actual = document.querySelector(`[data-actual="${joint}"]`);
    const target = document.querySelector(`[data-target="${joint}"]`);
    const command = document.querySelector(`[data-command="${joint}"]`);
    const temp = document.querySelector(`[data-temp="${joint}"]`);
    const voltage = document.querySelector(`[data-voltage="${joint}"]`);
    if (!slider) return;

    slider.disabled = !holding;
    if (!state.dragging.has(joint)) slider.value = snapshot.targets[joint];
    actual.textContent = formatValue(snapshot.positions[joint], limit.unit);
    target.textContent = `Target ${formatValue(snapshot.targets[joint], limit.unit)}`;
    command.textContent = formatValue(snapshot.commanded_positions[joint], limit.unit);
    temp.textContent = snapshot.temperatures[joint] == null ? "—" : `${snapshot.temperatures[joint]}°C`;
    voltage.textContent = snapshot.voltages[joint] == null ? "—" : `${(snapshot.voltages[joint] / 10).toFixed(1)}V`;
  });

  const events = snapshot.events.slice(state.locallyClearedEvents).slice().reverse();
  elements.eventLog.innerHTML = events.length
    ? events
        .map(
          (event) => `
            <div class="event ${event.level}">
              <time>${event.time}</time>
              <span>${event.message}</span>
            </div>`,
        )
        .join("")
    : '<p class="empty-log">No events in this view.</p>';

  if (snapshot.last_error && snapshot.last_error !== state.shownError) {
    state.shownError = snapshot.last_error;
    showToast(snapshot.last_error, true);
  }
  if (!snapshot.last_error) state.shownError = null;
}

async function refresh() {
  try {
    const snapshot = await api("/api/state");
    render(snapshot);
  } catch (error) {
    showToast(error.message, true);
  }
}

elements.connectButton.addEventListener("click", async () => {
  elements.connectButton.disabled = true;
  elements.connectButton.textContent = "Connecting…";
  try {
    const snapshot = await api("/api/connect", {
      method: "POST",
      body: JSON.stringify({ serial_port: elements.serialPort.value }),
    });
    render(snapshot);
    showToast("Connected and holding the measured pose.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.connectButton.textContent = "Connect";
    await refresh();
  }
});

elements.disconnectButton.addEventListener("click", async () => {
  try {
    render(await api("/api/disconnect", { method: "POST" }));
    showToast("Disconnected; torque released.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.torqueButton.addEventListener("click", async () => {
  const enabled = !state.snapshot?.torque_enabled;
  try {
    render(
      await api("/api/torque", {
        method: "POST",
        body: JSON.stringify({ enabled }),
      }),
    );
    showToast(enabled ? "Holding torque enabled." : "Torque released.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.stopButton.addEventListener("click", async () => {
  try {
    render(await api("/api/emergency-stop", { method: "POST" }));
    showToast("Emergency stop applied. Torque is released.");
  } catch (error) {
    showToast(error.message, true);
  }
});

elements.clearLogButton.addEventListener("click", () => {
  state.locallyClearedEvents = state.snapshot?.events.length || 0;
  render(state.snapshot);
});

window.addEventListener("keydown", async (event) => {
  if (event.key === "Escape" && state.snapshot?.connected) {
    event.preventDefault();
    try {
      render(await api("/api/emergency-stop", { method: "POST" }));
      showToast("Emergency stop applied. Torque is released.");
    } catch (error) {
      showToast(error.message, true);
    }
  }
});

refresh();
window.setInterval(refresh, 250);
