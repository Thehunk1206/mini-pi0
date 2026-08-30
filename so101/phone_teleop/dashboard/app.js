import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader from "/static/vendor/urdf-loader/URDFLoader.js";

const STREAMS = ["raw_ik", "one_euro", "kalman", "quintic", "ruckig", "measured"];
const STREAM_LABELS = { raw_ik: "Raw IK", one_euro: "One-Euro", kalman: "Kalman", quintic: "Custom quintic", ruckig: "Ruckig", measured: "Measured" };
const COLORS = { raw_ik: "#ff6b62", one_euro: "#61c9e8", kalman: "#aa8dff", quintic: "#f1b84b", ruckig: "#5cdf8b", measured: "#dfe8e3" };
const PROFILE_LABEL = { Smooth: "tuned", Safe: "×1.0", Balanced: "×1.5", Responsive: "×2.5" };
const GRIPPER_RANGE = [-0.174533, 1.74533];

const $ = (id) => document.getElementById(id);
const elements = Object.fromEntries([
  "modeBadge", "phaseBadge", "profileLabel", "viewer", "modelStatus", "ghostToggle", "fitButton", "resetCameraButton",
  "liveMetrics", "simControls", "liveControls", "timelinePanel", "scenarioSelect", "recordingField", "recordingPath", "targetEditor", "loadScenarioButton",
  "selectedStream", "streamChecks", "profileSelect", "advancedSettings", "minCutoff", "beta", "derivativeCutoff", "deadband", "applySettingsButton",
  "orientationField", "orientationEnabled",
  "settingsMessage", "baseButton", "restartButton", "playButton", "stepButton", "timeLabel", "timeline", "speedSelect",
  "plotJoint", "positionPlot", "velocityPlot", "accelerationPlot", "jerkPlot", "phonePlot", "telemetryPlot", "jointRows", "otgResult", "toast"
].map((id) => [id, $(id)]));

const app = { meta: null, state: null, history: null, historySequence: null, historyRefreshInFlight: false, draggingTimeline: false, settingsDirty: false, target: [], robot: null, ghost: null };

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* no JSON */ }
    const error = new Error(message); error.status = response.status; throw error;
  }
  return response.json();
}

let toastTimer;
function toast(message, error = false) {
  clearTimeout(toastTimer); elements.toast.textContent = message; elements.toast.className = `show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { elements.toast.className = ""; }, 3500);
}

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.domElement.tabIndex = 0;
elements.viewer.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(34, 1, 0.005, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.62, -0.62, 0.42);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.screenSpacePanning = true;
controls.target.set(-0.08, 0, 0.16);
scene.add(new THREE.HemisphereLight(0xdfffea, 0x18201c, 2.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.4); keyLight.position.set(1.2, -1.4, 2.0); keyLight.castShadow = true; scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x7ce7a0, 1.8); rimLight.position.set(-1.0, 1.2, .8); scene.add(rimLight);
const grid = new THREE.GridHelper(1.2, 24, 0x365348, 0x1e2c27); grid.rotation.x = Math.PI / 2; grid.position.z = -0.003; scene.add(grid);
scene.add(new THREE.AxesHelper(0.08));

function resizeRenderer() {
  const { width, height } = elements.viewer.getBoundingClientRect();
  if (!width || !height) return;
  renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
}
function animate() { requestAnimationFrame(animate); resizeRenderer(); controls.update(); renderer.render(scene, camera); }
animate();

function styleModel(model, ghost = false) {
  model.traverse((object) => {
    if (!object.isMesh) return;
    object.castShadow = !ghost; object.receiveShadow = !ghost;
    if (ghost) {
      object.material = new THREE.MeshStandardMaterial({ color: 0xf1b84b, transparent: true, opacity: .23, roughness: .7, depthWrite: false });
      object.renderOrder = 2;
    } else {
      const sourceColor = object.material?.color || new THREE.Color(0xe4be25);
      object.material = new THREE.MeshStandardMaterial({ color: sourceColor, roughness: .68, metalness: .03 });
    }
  });
}

async function loadModels(url) {
  try {
    const loader = new URDFLoader(); loader.parseCollision = false;
    const [robot, ghost] = await Promise.all([loader.loadAsync(url), loader.loadAsync(url)]);
    styleModel(robot, false); styleModel(ghost, true); ghost.visible = elements.ghostToggle.checked;
    scene.add(ghost); scene.add(robot); app.robot = robot; app.ghost = ghost;
    elements.modelStatus.innerHTML = "<i></i>Official model · 13 STL assets · 6 joints";
    elements.modelStatus.className = "model-status ready"; autoFit(); updateRobotPose();
  } catch (error) {
    elements.modelStatus.innerHTML = `<i></i>Model load failed · ${error.message}`; elements.modelStatus.className = "model-status error";
  }
}

function jointRadians(name, value) {
  if (name === "gripper") return GRIPPER_RANGE[0] + Math.max(0, Math.min(100, Number(value))) / 100 * (GRIPPER_RANGE[1] - GRIPPER_RANGE[0]);
  return THREE.MathUtils.degToRad(Number(value));
}
function poseModel(model, positions = {}) {
  if (!model) return;
  Object.entries(positions).forEach(([name, value]) => { if (Number.isFinite(Number(value)) && model.joints?.[name]) model.setJointValue(name, jointRadians(name, value)); });
}
function updateRobotPose() {
  if (!app.state) return;
  poseModel(app.robot, app.state.positions || {}); poseModel(app.ghost, app.state.commands || {});
}
function autoFit() {
  if (!app.robot) return;
  const box = new THREE.Box3().setFromObject(app.robot); const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, .12); controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).add(new THREE.Vector3(radius * 2.8, -radius * 2.8, radius * 1.8));
  camera.near = Math.max(radius / 100, .001); camera.far = radius * 30; camera.updateProjectionMatrix(); controls.update();
}
function resetCamera() { camera.position.set(.62, -.62, .42); controls.target.set(-.08, 0, .16); camera.near = .005; camera.far = 20; camera.updateProjectionMatrix(); controls.update(); }

function makeTargetEditor(joints, values) {
  elements.targetEditor.innerHTML = joints.map((joint, index) => `<label>${joint.replaceAll("_", " ")}<input data-joint-index="${index}" type="number" step="1" value="${Number(values[index]).toFixed(1)}"></label>`).join("");
}
function makeStreamChecks() {
  elements.streamChecks.innerHTML = STREAMS.map((name) => `<label><input type="checkbox" value="${name}" ${["raw_ik", "quintic", "ruckig"].includes(name) ? "checked" : ""}>${STREAM_LABELS[name]}</label>`).join("");
}

function format(value, digits = 1) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function renderMetrics(state) {
  const control = state.control || {}; const filter = control.phone_filter || {};
  const cutoff = filter.cutoff_hz ? `${format(Math.max(...filter.cutoff_hz), 2)} Hz` : "1.00 Hz";
  const errorValues = Object.values(state.tracking_error || control.tracking?.errors || {});
  const peakError = errorValues.length ? Math.max(...errorValues.map(Number)) : 0;
  const metrics = [
    ["TIME", state.mode === "simulation" ? `${format(state.playback_time_s, 2)} s` : `${format(state.loop_ms, 1)} ms loop`],
    ["CLUTCH", state.phone_enabled ? "HOLD ACTIVE" : "RELEASED"],
    ["FILTER CUTOFF", cutoff],
    ["PEAK ERROR", `${format(peakError, 2)}${peakError ? "° / %" : ""}`],
    ["STREAM", STREAM_LABELS[state.selected_stream] || "Ruckig command"],
  ];
  elements.liveMetrics.innerHTML = metrics.map(([label, value]) => `<div class="metric"><small>${label}</small><strong>${value}</strong></div>`).join("");
}

function renderTable(state) {
  const velocity = state.streams?.[state.selected_stream]?.velocity || state.control?.ruckig?.velocity || {};
  const errors = state.tracking_error || state.control?.tracking?.errors || {};
  const joints = app.meta.joint_names;
  elements.jointRows.innerHTML = joints.map((joint) => {
    const electrical = state.electrical?.[joint] || {}; const unit = joint === "gripper" ? "%" : "°";
    return `<tr><td>${joint.replaceAll("_", " ")}</td><td>${format(state.positions?.[joint])}${unit}</td><td>${format(state.commands?.[joint])}${unit}</td><td>${format(errors[joint], 2)}</td><td>${format(velocity[joint], 2)}</td><td>${format(electrical.current_ma, 0)} mA</td><td>${format(electrical.voltage_v, 2)} V</td></tr>`;
  }).join("");
  elements.otgResult.textContent = `OTG ${state.otg_result || state.control?.ruckig?.result || "—"}`;
}

function canvasFrame(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2); const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); return { ctx, width: rect.width, height: rect.height };
}
function drawPlot(canvas, series, markerTime, bounds = []) {
  const { ctx, width, height } = canvasFrame(canvas); ctx.clearRect(0, 0, width, height);
  const margin = { l: 43, r: 12, t: 22, b: 22 }; const w = width - margin.l - margin.r, h = height - margin.t - margin.b;
  const points = series.flatMap((item) => item.values.map((value, index) => [item.times[index], value])).filter(([, value]) => Number.isFinite(value));
  bounds.forEach((value) => { if (Number.isFinite(value)) points.push([0, value]); });
  if (!points.length) { ctx.fillStyle = "#718078"; ctx.font = "11px ui-monospace"; ctx.fillText("No samples yet", margin.l, margin.t + 16); return; }
  const tMax = Math.max(...points.map(([time]) => time), 1e-3); let yMin = Math.min(...points.map(([, value]) => value)); let yMax = Math.max(...points.map(([, value]) => value));
  const padding = Math.max((yMax - yMin) * .12, .01); yMin -= padding; yMax += padding;
  const x = (time) => margin.l + time / tMax * w; const y = (value) => margin.t + (yMax - value) / (yMax - yMin) * h;
  ctx.strokeStyle = "#26342e"; ctx.lineWidth = 1; ctx.font = "9px ui-monospace"; ctx.fillStyle = "#708079";
  for (let i = 0; i <= 4; i++) { const py = margin.t + h * i / 4; ctx.beginPath(); ctx.moveTo(margin.l, py); ctx.lineTo(width - margin.r, py); ctx.stroke(); ctx.fillText((yMax - (yMax-yMin)*i/4).toFixed(1), 2, py + 3); }
  bounds.forEach((value) => { ctx.strokeStyle = "rgba(255,107,98,.55)"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(margin.l, y(value)); ctx.lineTo(width-margin.r,y(value));ctx.stroke();ctx.setLineDash([]); });
  series.forEach((item, seriesIndex) => { ctx.strokeStyle = item.color; ctx.lineWidth = item.width || 1.6; ctx.globalAlpha = item.alpha || 1; ctx.beginPath(); item.values.forEach((value,index)=>{const px=x(item.times[index]),py=y(value);if(index===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);});ctx.stroke();ctx.globalAlpha=1;ctx.fillStyle=item.color;ctx.font="9px ui-monospace";ctx.fillText(item.label,margin.l+seriesIndex*74,10); });
  if (Number.isFinite(markerTime)) { ctx.strokeStyle = "rgba(255,255,255,.42)";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(markerTime),margin.t);ctx.lineTo(x(markerTime),margin.t+h);ctx.stroke(); }
  ctx.fillStyle="#708079";ctx.fillText("0",margin.l, height-5);ctx.fillText(`${tMax.toFixed(1)}s`,width-margin.r-28,height-5);
}

function historySamples() { return app.history?.samples || []; }
function plotTrajectoryField(canvas, field) {
  const samples = historySamples(); if (!samples.length) return drawPlot(canvas, [], 0);
  const joint = elements.plotJoint.value; const visible = app.state?.visible_streams || checkedStreams();
  const series = visible.filter((name) => samples[0].streams?.[name]).map((name) => ({ label: STREAM_LABELS[name], color: COLORS[name], times: samples.map((s,i)=>s.time_s ?? i/30), values: samples.map((s)=>Number(s.streams[name][field][joint])) }));
  let limit = null; const constraints = app.history?.constraints || app.state?.constraints;
  if (constraints && field !== "position") { const index = app.meta.joint_names.indexOf(joint); const key = joint === "gripper" ? `gripper_${field}` : `arm_${field}`; limit = constraints[key]?.[joint === "gripper" ? 0 : index]; }
  drawPlot(canvas, series, app.state?.playback_time_s, limit ? [limit, -limit] : []);
}
function plotPhone() {
  const samples = historySamples(); const axes = ["x","y","z"]; const axisColors=["#ff6b62","#5cdf8b","#61c9e8"];
  const series=[]; axes.forEach((axis,index)=>{series.push({label:`raw ${axis}`,color:axisColors[index],alpha:.45,width:1,times:samples.map((s,i)=>s.time_s??i/30),values:samples.map(s=>Number(s.phone?.raw_xyz_m?.[index]??0))});series.push({label:`filtered ${axis}`,color:axisColors[index],width:1.8,times:samples.map((s,i)=>s.time_s??i/30),values:samples.map(s=>Number(s.phone?.filtered_xyz_m?.[index]??0))});});
  drawPlot(elements.phonePlot, series, app.state?.playback_time_s);
}
function plotTelemetry() {
  const samples=historySamples(); const joint=elements.plotJoint.value;
  const times=samples.map((s,i)=>s.time_s??i/30); const series=[
    {label:"current /100",color:"#f1b84b",times,values:samples.map(s=>Number(s.electrical?.[joint]?.current_ma??0)/100)},
    {label:"voltage",color:"#61c9e8",times,values:samples.map(s=>Number(s.electrical?.[joint]?.voltage_v??0))},
    {label:"tracking error",color:"#ff6b62",times,values:samples.map(s=>Number(s.tracking_error?.[joint]??0))},
  ]; drawPlot(elements.telemetryPlot,series,app.state?.playback_time_s);
}
function redrawPlots(){plotTrajectoryField(elements.positionPlot,"position");plotTrajectoryField(elements.velocityPlot,"velocity");plotTrajectoryField(elements.accelerationPlot,"acceleration");plotTrajectoryField(elements.jerkPlot,"jerk");plotPhone();plotTelemetry();}

function checkedStreams(){return [...elements.streamChecks.querySelectorAll("input:checked")].map(input=>input.value);}
function updateState(state){
  app.state=state; elements.modeBadge.textContent=state.mode==="simulation"?"HARDWARE FREE":"LIVE HARDWARE";elements.phaseBadge.textContent=(state.phase||"unknown").toUpperCase();
  const profile=state.active_profile||"Safe";elements.profileLabel.textContent=`${profile} ${PROFILE_LABEL[profile]||""}`;
  const filter=state.filter_settings||state.control?.filter_settings||{};
  if(!app.settingsDirty){elements.profileSelect.value=profile;elements.orientationEnabled.checked=Boolean(state.control_mapping?.orientation_enabled);[[elements.minCutoff,"min_cutoff_hz",1],[elements.beta,"beta",1],[elements.derivativeCutoff,"derivative_cutoff_hz",1],[elements.deadband,"deadband_m",1000]].forEach(([input,key,scale])=>{if(Number.isFinite(Number(filter[key])))input.value=Number(filter[key])*scale;});}
  elements.applySettingsButton.disabled=Boolean(state.phone_enabled);
  if(app.settingsDirty)elements.settingsMessage.textContent=state.phone_enabled?"Release Hold to enable Apply.":"Ready to apply while Hold is released.";
  if (state.mode === "simulation" && [...elements.scenarioSelect.options].some(option => option.value === state.scenario)) elements.scenarioSelect.value = state.scenario;
  elements.playButton.textContent=state.playing?"Ⅱ":"▶";
  if(state.mode==="simulation"&&!app.draggingTimeline){elements.timeline.max=state.duration_s;elements.timeline.value=state.playback_time_s;elements.timeLabel.textContent=`${format(state.playback_time_s,2)} / ${format(state.duration_s,2)} s`;}
  renderMetrics(state);renderTable(state);updateRobotPose();
}

async function refreshHistory(){
  if(app.historyRefreshInFlight)return;
  app.historyRefreshInFlight=true;
  try{
    const incremental=app.meta.mode==="live"&&Number.isFinite(app.historySequence);
    const suffix=incremental?`?after_sequence=${app.historySequence}`:"";
    const history=await api(`/api/history${suffix}`);
    if(app.meta.mode==="live"){
      const incoming=(history.samples||[]).map(sample=>({sequence:Number(sample.sequence),streams:{ruckig:{position:sample.commands||{},velocity:sample.control?.ruckig?.velocity||{},acceleration:sample.control?.ruckig?.acceleration||{},jerk:sample.control?.ruckig?.jerk||{}},measured:{position:sample.positions||{},velocity:{},acceleration:{},jerk:{}}},phone:{raw_xyz_m:sample.control?.phone_filter?.raw_position_m||[0,0,0],filtered_xyz_m:sample.control?.phone_filter?.filtered_position_m||[0,0,0]},electrical:sample.electrical||{},tracking_error:sample.control?.tracking?.errors||{}}));
      const existing=incremental&&!history.reset?(app.history?.samples||[]):[];
      const samples=[...existing,...incoming].slice(-1800);
      const firstSequence=samples[0]?.sequence??0;
      samples.forEach((sample,index)=>{sample.time_s=Number.isFinite(sample.sequence)?(sample.sequence-firstSequence)/30:index/30;});
      app.history={...history,samples,constraints:app.state?.control?.constraints};
      app.historySequence=Number(history.latest_sequence);
    }else app.history=history;
    redrawPlots();
  }finally{app.historyRefreshInFlight=false;}
}

async function playback(action, extra={}){try{updateState(await api("/api/playback",{method:"POST",body:JSON.stringify({action,...extra})}));}catch(error){toast(error.message,true);}}
elements.fitButton.addEventListener("click",autoFit);elements.resetCameraButton.addEventListener("click",resetCamera);elements.ghostToggle.addEventListener("change",()=>{if(app.ghost)app.ghost.visible=elements.ghostToggle.checked;});
elements.playButton.addEventListener("click",()=>playback(app.state?.playing?"pause":"play",{speed:Number(elements.speedSelect.value)}));elements.restartButton.addEventListener("click",()=>playback("restart"));elements.stepButton.addEventListener("click",()=>playback("step"));
elements.timeline.addEventListener("pointerdown",()=>{app.draggingTimeline=true;});elements.timeline.addEventListener("change",async()=>{app.draggingTimeline=false;await playback("scrub",{time_s:Number(elements.timeline.value)});});elements.timeline.addEventListener("input",()=>{elements.timeLabel.textContent=`${format(elements.timeline.value,2)} / ${format(elements.timeline.max,2)} s`;});
elements.speedSelect.addEventListener("change",()=>{if(app.state?.playing)playback("play",{speed:Number(elements.speedSelect.value)});});
elements.selectedStream.addEventListener("change",async()=>{if(app.meta.mode==="simulation"){try{updateState(await api("/api/streams",{method:"PUT",body:JSON.stringify({selected:elements.selectedStream.value,visible:checkedStreams()})}));}catch(error){toast(error.message,true);}}});
elements.streamChecks.addEventListener("change",async(event)=>{const visible=checkedStreams();if(!visible.length){event.target.checked=true;return;}if(app.meta.mode==="simulation"){try{updateState(await api("/api/streams",{method:"PUT",body:JSON.stringify({selected:elements.selectedStream.value,visible})}));app.state.visible_streams=visible;redrawPlots();}catch(error){toast(error.message,true);}}});
elements.plotJoint.addEventListener("change",redrawPlots);
[elements.profileSelect,elements.orientationEnabled,elements.minCutoff,elements.beta,elements.derivativeCutoff,elements.deadband].forEach((input) => {
  ["input", "change"].forEach((eventName) => input.addEventListener(eventName, () => {
    app.settingsDirty = true;
    if (app.state) updateState(app.state);
  }));
});
function updateScenarioFields(){const recorded=elements.scenarioSelect.value==="recorded_session";elements.recordingField.classList.toggle("hidden",!recorded);elements.targetEditor.classList.toggle("hidden",recorded);}
elements.scenarioSelect.addEventListener("change",updateScenarioFields);
elements.loadScenarioButton.addEventListener("click",async()=>{const target=[...elements.targetEditor.querySelectorAll("input")].map(input=>Number(input.value));const payload={name:elements.scenarioSelect.value,target};if(payload.name==="recorded_session")payload.recording=elements.recordingPath.value;elements.loadScenarioButton.disabled=true;elements.loadScenarioButton.textContent="Generating…";elements.settingsMessage.textContent="";try{updateState(await api("/api/scenario",{method:"POST",body:JSON.stringify(payload)}));await refreshHistory();toast("Experiment regenerated from identical target conditions.");}catch(error){toast(error.message,true);}finally{elements.loadScenarioButton.disabled=false;elements.loadScenarioButton.textContent="Load experiment";}});
elements.applySettingsButton.addEventListener("click",async()=>{const payload={profile:elements.profileSelect.value,min_cutoff_hz:Number(elements.minCutoff.value),beta:Number(elements.beta.value),derivative_cutoff_hz:Number(elements.derivativeCutoff.value),deadband_m:Number(elements.deadband.value)/1000};if(app.meta.mode==="live")payload.orientation_enabled=elements.orientationEnabled.checked;try{await api("/api/settings",{method:"PUT",body:JSON.stringify(payload)});app.settingsDirty=false;if(app.meta.mode==="live"){await new Promise(resolve=>setTimeout(resolve,100));updateState(await api("/api/state"));}elements.settingsMessage.textContent=app.meta.mode==="simulation"?"Applied. Scenario reset to t = 0.":"Applied to live control.";await refreshHistory();}catch(error){elements.settingsMessage.textContent=error.message;toast(error.message,error.status!==409);}});
elements.baseButton.addEventListener("click",async()=>{try{updateState(await api("/api/return-to-base",{method:"POST"}));toast("Base return queued. Release Hold.");}catch(error){toast(error.message,true);}});

async function init(){
  try{
    app.meta=await api("/api/meta");elements.plotJoint.innerHTML=app.meta.joint_names.map(name=>`<option value="${name}">${name.replaceAll("_"," ")}</option>`).join("");makeStreamChecks();
    if(app.meta.mode==="simulation"){
      const scenarios=await api("/api/scenarios");elements.scenarioSelect.innerHTML=scenarios.scenarios.map(name=>`<option value="${name}">${name.replaceAll("_"," ")}</option>`).join("");app.target=scenarios.default_target;makeTargetEditor(app.meta.joint_names,app.target);updateScenarioFields();
    }else{elements.simControls.classList.add("hidden");elements.liveControls.classList.remove("hidden");elements.orientationField.classList.remove("hidden");elements.timelinePanel.classList.add("hidden");}
    const state=await api("/api/state");updateState(state);elements.selectedStream.value=state.selected_stream||"ruckig";await Promise.all([loadModels(app.meta.model_url),refreshHistory()]);
    window.setInterval(async()=>{try{updateState(await api("/api/state"));}catch(error){elements.phaseBadge.textContent="OFFLINE";}},app.meta.mode==="simulation"?80:180);
    if(app.meta.mode==="live")window.setInterval(()=>refreshHistory().catch(()=>{}),1000);
  }catch(error){toast(error.message,true);elements.modelStatus.textContent=`Initialization failed · ${error.message}`;elements.modelStatus.className="model-status error";}
}
window.addEventListener("resize",redrawPlots);init();
