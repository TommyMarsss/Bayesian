package occupancy

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// ScenarioData bundles everything the replay page needs for one run.
type ScenarioData struct {
	Name   string
	POcc   float64
	PFree  float64
	Frames []Frame
}

// PageData is the full payload embedded in the generated HTML file.
type PageData struct {
	Width, Height int
	Truth         []bool // ground-truth occupancy, for the overlay toggle
	Scenarios     []ScenarioData
}

type jsonFrame struct {
	Probs string     `json:"p"` // base64 of 0..255 quantized probabilities
	Stats FrameStats `json:"s"`
}

type jsonScenario struct {
	Name   string      `json:"name"`
	POcc   float64     `json:"pOcc"`
	PFree  float64     `json:"pFree"`
	Frames []jsonFrame `json:"frames"`
}

type jsonPage struct {
	Width     int            `json:"width"`
	Height    int            `json:"height"`
	Truth     string         `json:"truth"` // base64 of 0/1 bytes
	MinObs    int            `json:"minObs"`
	Extreme   float64        `json:"extreme"`
	Scenarios []jsonScenario `json:"scenarios"`
}

// WriteHTML renders the data as one self-contained static HTML file:
// all frames and logic are embedded, no server and no third-party
// libraries are involved.
func WriteHTML(w io.Writer, data PageData) error {
	jp := jsonPage{
		Width:   data.Width,
		Height:  data.Height,
		MinObs:  DefaultMinObs,
		Extreme: DefaultExtreme,
	}
	truth := make([]byte, len(data.Truth))
	for i, t := range data.Truth {
		if t {
			truth[i] = 1
		}
	}
	jp.Truth = base64.StdEncoding.EncodeToString(truth)
	for _, sc := range data.Scenarios {
		js := jsonScenario{Name: sc.Name, POcc: sc.POcc, PFree: sc.PFree}
		for _, f := range sc.Frames {
			js.Frames = append(js.Frames, jsonFrame{
				Probs: base64.StdEncoding.EncodeToString(f.Probs),
				Stats: f.Stats,
			})
		}
		jp.Scenarios = append(jp.Scenarios, js)
	}
	payload, err := json.Marshal(jp)
	if err != nil {
		return fmt.Errorf("occupancy: marshal page data: %w", err)
	}
	_, err = io.WriteString(w, strings.Replace(pageTemplate, "__DATA__", string(payload), 1))
	return err
}

const pageTemplate = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Log-Odds 占据栅格地图回放</title>
<style>
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; margin: 24px; background:#fafafa; color:#222; }
  h1 { font-size: 20px; }
  .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  canvas { border: 1px solid #999; image-rendering: pixelated; background:#777; }
  .panel { min-width: 260px; }
  button, select { font-size: 14px; padding: 4px 10px; margin: 2px; cursor: pointer; }
  button.active { background:#1565c0; color:#fff; border-color:#1565c0; }
  .stats { font-family: Menlo, Consolas, monospace; font-size: 13px; line-height: 1.7; margin-top: 12px; }
  #warn { display:none; margin-top:10px; padding:8px 10px; background:#ffebee; border:1px solid #c62828; color:#b71c1c; font-size:13px; border-radius:4px; }
  .legend { font-size:12px; margin-top:8px; }
  .sw { display:inline-block; width:12px; height:12px; vertical-align:-2px; margin:0 4px 0 10px; border:1px solid #666; }
  input[type=range] { width: 420px; }
  label { font-size: 13px; }
</style>
</head>
<body>
<h1>Log-Odds 占据栅格地图 — 逐帧回放</h1>
<div class="row">
  <div>
    <canvas id="cv"></canvas>
    <div>
      <input type="range" id="slider" min="0" value="0">
    </div>
    <div>
      <button id="prev">&#9664; 上一帧</button>
      <button id="play">&#9654; 播放</button>
      <button id="next">下一帧 &#9654;</button>
      <select id="speed">
        <option value="400">0.5&times;</option>
        <option value="200" selected>1&times;</option>
        <option value="80">2.5&times;</option>
        <option value="30">6&times;</option>
      </select>
      <label><input type="checkbox" id="truth"> 叠加真实地图</label>
    </div>
    <div class="legend">
      颜色 = 占据概率:
      <span class="sw" style="background:#2e7d32"></span>空闲
      <span class="sw" style="background:#8a8a8a"></span>未知
      <span class="sw" style="background:#c62828"></span>占据
      <span class="sw" style="background:repeating-linear-gradient(45deg,#000 0 2px,transparent 2px 4px)"></span>真实障碍(叠加)
    </div>
  </div>
  <div class="panel">
    <div id="scenarios"></div>
    <div class="stats" id="stats"></div>
    <div id="warn"></div>
  </div>
</div>
<script>
"use strict";
const DATA = __DATA__;

const W = DATA.width, H = DATA.height;
const truth = Uint8Array.from(atob(DATA.truth), c => c.charCodeAt(0));
const frames = DATA.scenarios.map(sc => sc.frames.map(f =>
  Uint8Array.from(atob(f.p), c => c.charCodeAt(0))));

let scenario = 0, frame = 0, playing = false, timer = null;

const cv = document.getElementById("cv");
const scale = Math.max(4, Math.floor(720 / Math.max(W, H)));
cv.width = W * scale; cv.height = H * scale;
const ctx = cv.getContext("2d");
const img = ctx.createImageData(W, H);

const slider = document.getElementById("slider");
const statsEl = document.getElementById("stats");
const warnEl = document.getElementById("warn");

function lerp(a, b, t) { return a + (b - a) * t; }
// 空闲绿(46,125,50) -> 未知灰(138,138,138) -> 占据红(198,40,40)
function colorFor(p) {
  if (p >= 0.5) {
    const t = (p - 0.5) * 2;
    return [lerp(138,198,t), lerp(138,40,t), lerp(138,40,t)];
  }
  const t = p * 2;
  return [lerp(46,138,t), lerp(125,138,t), lerp(50,138,t)];
}

function render() {
  const probs = frames[scenario][frame];
  for (let i = 0; i < W * H; i++) {
    const [r, g, b] = colorFor(probs[i] / 255);
    img.data[i*4] = r; img.data[i*4+1] = g; img.data[i*4+2] = b; img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(cv, 0, 0, W, H, 0, 0, cv.width, cv.height);
  if (document.getElementById("truth").checked) {
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (truth[y*W+x]) ctx.fillRect(x*scale, y*scale, scale, scale);
    }
  }
  slider.max = frames[scenario].length - 1;
  slider.value = frame;
  const s = DATA.scenarios[scenario].frames[frame].s;
  const sc = DATA.scenarios[scenario];
  statsEl.innerHTML =
    "场景: <b>" + sc.name + "</b> (p_occ=" + sc.pOcc + ", p_free=" + sc.pFree + ")<br>" +
    "帧: " + (frame+1) + " / " + frames[scenario].length + "<br>" +
    "已观测栅格: " + s.Observed + "<br>" +
    "极端概率栅格 (p&gt;" + DATA.extreme + " 或 &lt;" + (1-DATA.extreme).toFixed(2) + "): " + s.Extreme + "<br>" +
    "饱和栅格: " + s.Saturated + "<br>" +
    "观测不足 " + DATA.minObs + " 次即极端: <b>" + s.Premature + "</b>";
  if (s.Premature > 0) {
    warnEl.style.display = "block";
    warnEl.textContent = "⚠ 检测到过度自信异常收敛: " + s.Premature +
      " 个栅格在观测不足 " + DATA.minObs + " 次时已达到极端概率, 传感器逆模型置信度可能过高。";
  } else {
    warnEl.style.display = "none";
  }
}

function setFrame(f) {
  const n = frames[scenario].length;
  frame = Math.max(0, Math.min(n - 1, f));
  render();
}
function stop() { playing = false; clearInterval(timer); document.getElementById("play").innerHTML = "&#9654; 播放"; }

document.getElementById("play").onclick = () => {
  if (playing) { stop(); return; }
  playing = true;
  document.getElementById("play").innerHTML = "&#10074;&#10074; 暂停";
  timer = setInterval(() => {
    if (frame + 1 >= frames[scenario].length) { stop(); return; }
    setFrame(frame + 1);
  }, +document.getElementById("speed").value);
};
document.getElementById("prev").onclick = () => { stop(); setFrame(frame - 1); };
document.getElementById("next").onclick = () => { stop(); setFrame(frame + 1); };
slider.oninput = () => { stop(); setFrame(+slider.value); };
document.getElementById("truth").onchange = render;
document.getElementById("speed").onchange = () => { if (playing) { stop(); document.getElementById("play").click(); } };

const scDiv = document.getElementById("scenarios");
DATA.scenarios.forEach((sc, i) => {
  const b = document.createElement("button");
  b.textContent = sc.name;
  b.onclick = () => { stop(); scenario = i; frame = 0;
    scDiv.querySelectorAll("button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); render(); };
  if (i === 0) b.classList.add("active");
  scDiv.appendChild(b);
});
render();
</script>
</body>
</html>
`
