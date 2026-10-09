import base64
import json
import tempfile
from pathlib import Path

import cv2
from google.colab import output
from IPython.display import display, Javascript

from model4_tracker import (
    extract_rgb_frames,
    first_frame,
    track_with_semantic_reacquisition,
)

REPO = Path("/content/Target_Locker")
SAM_CKPT = Path("/content/sam2.1_hiera_tiny.pt")

MAX_SIDE = 768

display(Javascript(r"""
(() => {
  const old = document.getElementById("tl-app-model4");
  if (old) old.remove();

  const root = document.createElement("div");
  root.id = "tl-app-model4";
  root.innerHTML = `
  <style>
    #tl-app-model4{max-width:1050px;margin:14px auto;color:#e8f7ff;font-family:Inter,system-ui,Arial,sans-serif}
    #tl-app-model4 *{box-sizing:border-box}
    .shell{overflow:hidden;border-radius:26px;background:radial-gradient(circle at 0 0,rgba(249,115,22,.16),transparent 30%),radial-gradient(circle at 100% 0,rgba(239,68,68,.13),transparent 28%),#030712;border:1px solid rgba(251,146,60,.25);box-shadow:0 30px 80px rgba(0,0,0,.35)}
    .head{padding:22px 26px;border-bottom:1px solid rgba(251,146,60,.14);display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap}
    .kicker{font-size:10px;letter-spacing:4px;color:#fb923c;font-weight:800}
    .title{font-size:31px;font-weight:950;letter-spacing:-1px;color:white;margin-top:4px}
    .sub{font-size:13px;color:#64748b;margin-top:4px}
    .chip{border:1px solid #7c2d12;background:#140b08;color:#fdba74;border-radius:999px;padding:8px 12px;font-size:11px;font-weight:800;letter-spacing:1px}
    .body{padding:18px}
    .preview{position:relative;overflow:hidden;border-radius:18px;background:#000;border:1px solid rgba(251,146,60,.20);min-height:480px;display:flex;align-items:center;justify-content:center}
    .preview video,.preview canvas{width:100%;height:auto;max-height:650px;display:block;background:#000}
    .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:#64748b;text-align:center;padding:70px 20px}
    .upload{margin-top:8px;border:1px solid #fb923c;background:linear-gradient(135deg,#ea580c,#dc2626);color:white;border-radius:12px;padding:11px 20px;font-weight:900;cursor:pointer}
    .overlay{display:none;position:absolute;inset:0;z-index:10;background:rgba(2,6,23,.86);backdrop-filter:blur(7px);align-items:center;justify-content:center;flex-direction:column}
    .ring{width:76px;height:76px;border-radius:50%;border:2px solid rgba(251,146,60,.15);border-top-color:#fb923c;border-right-color:#ef4444;animation:spin 1s linear infinite}
    .otitle{margin-top:18px;font-weight:950;letter-spacing:2px;font-size:18px}
    .osub{margin-top:5px;color:#94a3b8;font-size:12px;text-align:center;padding:0 18px}
    .scan{display:none;pointer-events:none;position:absolute;z-index:4;left:0;right:0;height:8%;background:linear-gradient(to bottom,transparent,rgba(251,146,60,.05),rgba(251,146,60,.75),rgba(251,146,60,.05),transparent);animation:scan 2.4s linear infinite}
    .reticle{position:absolute;z-index:5;pointer-events:none;display:none;width:62px;height:62px;transform:translate(-50%,-50%)}
    .reticle:before,.reticle:after{content:"";position:absolute;background:#fb923c;box-shadow:0 0 12px #fb923c}
    .reticle:before{left:30px;top:0;width:2px;height:62px}.reticle:after{top:30px;left:0;height:2px;width:62px}
    .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:13px}
    .btn{border-radius:11px;padding:10px 15px;font-weight:900;cursor:pointer;border:1px solid #334155;background:#111827;color:#cbd5e1}
    .btn.primary{border-color:#fb923c;background:linear-gradient(135deg,#ea580c,#dc2626);color:white}
    .btn:disabled{opacity:.35;cursor:not-allowed}
    .selectbox{border-radius:11px;padding:10px 12px;font-weight:800;border:1px solid #334155;background:#111827;color:#e2e8f0}
    .info{margin-left:auto;font-family:monospace;color:#64748b;font-size:11px}
    .telemetry{margin-top:13px;display:grid;grid-template-columns:repeat(4,1fr);gap:9px}
    .card{background:#07111f;border:1px solid rgba(251,146,60,.10);border-radius:12px;padding:10px 12px}
    .card span{display:block;color:#475569;font-size:9px;letter-spacing:1.6px}.card b{display:block;margin-top:4px;color:#fff7ed;font-size:13px}
    .progress{display:none;margin-top:13px;background:#07111f;border-radius:999px;height:8px;overflow:hidden}
    .progress>div{width:0;height:100%;background:linear-gradient(90deg,#fb923c,#ef4444);transition:width .2s ease}
    .metrics3{display:none;margin-top:14px;grid-template-columns:repeat(4,1fr);gap:9px}
    .metric3{background:#07111f;border:1px solid rgba(103,232,249,.14);border-radius:12px;padding:10px 12px}
    .metric3 span{display:block;color:#64748b;font-size:9px;letter-spacing:1.2px}.metric3 b{display:block;color:#e0f2fe;margin-top:4px;font-size:14px}
    .metric-note{display:none;color:#64748b;font-size:10px;margin-top:8px}
    @keyframes spin{to{transform:rotate(360deg)}} @keyframes scan{from{top:-8%}to{top:100%}}
    @media(max-width:760px){.telemetry{grid-template-columns:repeat(2,1fr)}.info{width:100%;margin-left:0}}
  </style>

  <div class="shell">
    <div class="head">
      <div>
        <div class="kicker">PROPOSED MODEL 4 // SIGLIP 2 + ADAPTIVE MOTION MEMORY</div>
        <div class="title">RGB TARGET LOCK // SEMANTIC MEMORY</div>
        <div class="sub">RGB → SAM2 → SigLIP 2 identity verification → motion-aware re-detection → re-lock.</div>
      </div>
      <div id="chip" class="chip">SYSTEM READY</div>
    </div>

    <div class="body">
      <input id="file" type="file" accept="video/*" style="display:none">
      <div class="preview">
        <div id="empty" class="empty">
          <div style="font-size:46px">◉</div>
          <div style="font-size:18px;font-weight:900;color:#cbd5e1">LOAD RGB VIDEO</div>
          <div>Select one target. When SAM2 loses identity, Model 4 searches the full frame with a deep target re-detector and re-locks from a confirmed bounding box.</div>
          <button id="upload" class="upload">UPLOAD RGB VIDEO</button>
        </div>
        <video id="video" controls playsinline style="display:none"></video>
        <canvas id="canvas" style="display:none"></canvas>
        <div id="scan" class="scan"></div>
        <div id="reticle" class="reticle"></div>
        <div id="overlay" class="overlay">
          <div class="ring"></div>
          <div id="otitle" class="otitle">PREPARING</div>
          <div id="osub" class="osub">Waiting…</div>
        </div>
      </div>

      <div class="controls">
        <button id="newVideo" class="btn" style="display:none">NEW VIDEO</button>
        <button id="select" class="btn" disabled>SELECT TARGET</button>
        <button id="track" class="btn primary" disabled>LOCK & TRACK</button>
        <div id="info" class="info">NO VIDEO LOADED</div>
      </div>

      <div id="telemetry" class="telemetry" style="display:none">
        <div class="card"><span>RGB RESOLUTION</span><b id="res">—</b></div>
        <div class="card"><span>FRAME RATE</span><b id="fps">—</b></div>
        <div class="card"><span>FRAMES</span><b id="frames">—</b></div>
        <div class="card"><span>PIPELINE</span><b>SAM2 + SIGLIP2 + MEMORY</b></div>
      </div>
      <div id="progress" class="progress"><div id="bar"></div></div>
      <div id="metrics3" class="metrics3">
        <div class="metric3"><span>LOSS EVENTS</span><b id="mRetention">—</b></div>
        <div class="metric3"><span>GLOBAL SEARCH FRAMES</span><b id="mLoss">—</b></div>
        <div class="metric3"><span>CONFIRMED RE-ACQ</span><b id="mRecovery">—</b></div>
        <div class="metric3"><span>MEAN GLOBAL SCORE</span><b id="mLatency">—</b></div>
        <div class="metric3"><span>IDENTITY VERIFIER</span><b id="mConfidence">—</b></div>
        <div class="metric3"><span>CENTER JUMP</span><b id="mJump">—</b></div>
        <div class="metric3"><span>TRACKING FPS</span><b id="mTrackFps">—</b></div>
        <div class="metric3"><span>TRACKER</span><b id="mAlgorithm">SAM2 + SigLIP 2</b></div>
      </div>
      <div id="metricNote" class="metric-note">Diagnostics only. Without ground truth, these values are not tracking accuracy metrics.</div>
    </div>
  </div>`;

  document.body.appendChild(root);
  const q = s => root.querySelector(s);
  window.TL2 = {
    root, file:q("#file"), upload:q("#upload"), newVideo:q("#newVideo"),
    empty:q("#empty"), video:q("#video"), canvas:q("#canvas"), ctx:q("#canvas").getContext("2d"),
    scan:q("#scan"), reticle:q("#reticle"), overlay:q("#overlay"), otitle:q("#otitle"), osub:q("#osub"),
    select:q("#select"), track:q("#track"), chip:q("#chip"), info:q("#info"),
    telemetry:q("#telemetry"), progress:q("#progress"), bar:q("#bar"),
    res:q("#res"), fps:q("#fps"), frames:q("#frames"), metrics3:q("#metrics3"), metricNote:q("#metricNote"), mRetention:q("#mRetention"), mLoss:q("#mLoss"), mRecovery:q("#mRecovery"), mLatency:q("#mLatency"), mConfidence:q("#mConfidence"), mJump:q("#mJump"), mTrackFps:q("#mTrackFps"), mAlgorithm:q("#mAlgorithm"), selectedFile:null, target:null
  };
  TL2.upload.onclick = () => TL2.file.click();
  TL2.newVideo.onclick = () => TL2.file.click();
  google.colab.output.setIframeHeight(document.body.scrollHeight,true);
})();
"""))

file_meta = output.eval_js(r"""
(async () => {
  const A = window.TL2;
  const f = await new Promise(resolve => {
    A.file.onchange = () => resolve(A.file.files?.[0] || null);
  });
  if(!f) return null;

  A.selectedFile = f;
  A.empty.style.display = "none";
  A.video.style.display = "none";
  A.canvas.style.display = "none";
  A.newVideo.style.display = "inline-block";
  A.overlay.style.display = "flex";
  A.otitle.textContent = "UPLOADING RGB VIDEO";
  A.osub.textContent = "Preparing RGB video for SEMANTIC-MOTION RE-DETECTOR…";
  A.chip.textContent = "RGB INPUT";
  A.info.textContent = f.name.toUpperCase();
  A.progress.style.display = "block";
  A.bar.style.width = "2%";
  return {name:f.name,size:f.size,type:f.type || "video/mp4"};
})()
""")

if not file_meta:
    raise RuntimeError("No video selected.")

name = file_meta["name"]
size = int(file_meta["size"])
suffix = Path(name).suffix or ".mp4"

job = Path(tempfile.mkdtemp(prefix="model4-semantic-motion-", dir="/content"))
video_path = job / f"input{suffix}"
rgb_dir = job / "rgb_frames"
final_video = job / "model4_semantic_motion.mp4"
metrics_json = job / "metrics_semantic_motion.json"
metrics_csv = job / "metrics_semantic_motion.csv"

CHUNK = 512 * 1024
chunks = (size + CHUNK - 1) // CHUNK

with open(video_path, "wb") as f:
    for i in range(chunks):
        start = i * CHUNK
        end = min(size, (i + 1) * CHUNK)
        chunk_b64 = output.eval_js(f"""
        (async () => {{
          const A=window.TL2;
          const buf=await A.selectedFile.slice({start},{end}).arrayBuffer();
          const bytes=new Uint8Array(buf);
          let binary="";
          const step=0x8000;
          for(let j=0;j<bytes.length;j+=step){{
            binary += String.fromCharCode(...bytes.subarray(j,j+step));
          }}
          A.bar.style.width="{2 + int(((i+1)/max(chunks,1))*10)}%";
          A.osub.textContent="Uploading RGB video… {round(((i+1)/max(chunks,1))*100)}%";
          return btoa(binary);
        }})()
        """)
        f.write(base64.b64decode(chunk_b64))


def update_progress(stage, done, total):
    if total <= 0:
        return
    pct = min(100.0, max(0.0, (done / total) * 100.0))
    if stage == "extract":
        bar = 12 + int(pct * 0.13)
        title = "EXTRACTING RGB FRAMES"
        sub = f"Frame {done} / {total}"
    else:
        bar = 68 + int(pct * 0.28)
        title = "SEMANTIC-MOTION RE-DETECTOR TRACKING"
        sub = f"Frame {done} / {total}"

    output.eval_js(f"""(() => {{
      const A=window.TL2;
      A.bar.style.width="{bar}%";
      A.otitle.textContent={json.dumps(title)};
      A.osub.textContent={json.dumps(sub)};
      return true;
    }})()""")


meta = extract_rgb_frames(video_path, rgb_dir, max_side=MAX_SIDE, progress=update_progress)

output.eval_js(f"""(() => {{
  const A=window.TL2;
  A.res.textContent={json.dumps(f'{meta["source_width"]} × {meta["source_height"]}')};
  A.fps.textContent={json.dumps(f'{meta["fps"]:.2f} FPS')};
  A.frames.textContent={json.dumps(str(meta["frames"]))};
  A.telemetry.style.display="grid";
  A.bar.style.width="25%";
  A.otitle.textContent="RGB TARGET LOCKER READY";
  A.osub.textContent="SEMANTIC-MOTION RE-DETECTOR is ready for target selection.";
  A.chip.textContent="RGB READY";
  return true;
}})()""")

rgb0 = first_frame(rgb_dir)

ok, buf = cv2.imencode(".jpg", rgb0, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
if not ok:
    raise RuntimeError("Could not encode first RGB frame.")
img_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

output.eval_js(f"""(async () => {{
  const A=window.TL2;
  const img=new Image();
  img.src={json.dumps(img_url)};
  await new Promise((resolve,reject)=>{{img.onload=resolve;img.onerror=reject;}});
  A.canvas.width=img.naturalWidth;
  A.canvas.height=img.naturalHeight;
  A.ctx.drawImage(img,0,0,A.canvas.width,A.canvas.height);
  A.canvas.style.display="block";
  A.video.style.display="none";
  A.overlay.style.display="none";
  A.progress.style.display="none";
  A.select.disabled=false;
  A.chip.textContent="RGB VIDEO READY";
  A.info.textContent="RGB FRAME 1 // SELECT TARGET";
  A.bar.style.width="65%";
  return true;
}})()""")

selection = output.eval_js(r"""
(async () => {
  const A=window.TL2;
  await new Promise(resolve => { A.select.onclick = () => resolve(); });

  A.select.disabled=true;
  A.select.textContent="CLICK OBJECT";
  A.scan.style.display="block";
  A.chip.textContent="TARGET ACQUISITION";
  A.info.textContent="CLICK THE TARGET ON RGB FRAME 1";

  const click = await new Promise(resolve => {
    A.canvas.onclick = ev => {
      const r=A.canvas.getBoundingClientRect();
      const x=(ev.clientX-r.left)*A.canvas.width/r.width;
      const y=(ev.clientY-r.top)*A.canvas.height/r.height;

      A.target=[x,y];
      A.reticle.style.left=((x/A.canvas.width)*100)+"%";
      A.reticle.style.top=((y/A.canvas.height)*100)+"%";
      A.reticle.style.display="block";
      A.scan.style.display="none";
      A.canvas.onclick=null;

      A.track.disabled=false;
      A.select.textContent="TARGET SELECTED";
      A.chip.textContent="TARGET LOCKED";
      A.info.textContent="PRESS LOCK & TRACK";

      resolve({x,y,cw:A.canvas.width,ch:A.canvas.height});
    };
  });

  await new Promise(resolve => { A.track.onclick=()=>resolve(); });

  A.track.disabled=true;
  A.select.disabled=true;
  A.overlay.style.display="flex";
  A.progress.style.display="block";
  A.otitle.textContent="MODEL 4 TRACKING + RECOVERY";
  A.osub.textContent="Running SEMANTIC-MOTION RE-DETECTOR motion-aware long-term memory tracking…";
  A.chip.textContent="TRACKING";
  A.bar.style.width="68%";

  return click;
})()
""")

tx_preview = int(float(selection["x"]) * rgb0.shape[1] / max(float(selection["cw"]), 1))
ty_preview = int(float(selection["y"]) * rgb0.shape[0] / max(float(selection["ch"]), 1))

final_video, model4_metrics = track_with_semantic_reacquisition(
    rgb_dir,
    (tx_preview, ty_preview),
    meta["fps"],
    final_video,
    metrics_json,
    metrics_csv,
    REPO / "SAM2_streaming-main",
    SAM_CKPT,
    max_side=960,
    progress=update_progress,
)

output.eval_js(f"""(() => {{
  const A=window.TL2;
  A.metrics3.style.display="grid";
  A.metricNote.style.display="block";
  A.mRetention.textContent={json.dumps(str(model4_metrics["loss_events"]))};
  A.mLoss.textContent={json.dumps(str(model4_metrics["global_search_frames"]))};
  A.mRecovery.textContent={json.dumps(str(model4_metrics["confirmed_reacquisitions"]))};
  A.mLatency.textContent={json.dumps(str(model4_metrics["mean_global_best_score"]))};
  A.mConfidence.textContent={json.dumps(str(model4_metrics["mean_identity_verifier_score"]))};
  A.mJump.textContent={json.dumps(str(model4_metrics["mean_normalized_center_jump"]))};
  A.mTrackFps.textContent={json.dumps(str(model4_metrics["tracking_fps"]))};
  A.mAlgorithm.textContent="SAM2 + SigLIP 2 + Motion";
  return true;
}})()""")

if final_video.stat().st_size <= 100 * 1024 * 1024:
    data_url = "data:video/mp4;base64," + base64.b64encode(final_video.read_bytes()).decode("ascii")

    output.eval_js(f"""(() => {{
      const A=window.TL2;
      A.canvas.style.display="none";
      A.reticle.style.display="none";
      A.scan.style.display="none";
      A.video.style.display="block";
      A.video.src={json.dumps(data_url)};
      A.video.load();
      A.overlay.style.display="none";
      A.progress.style.display="none";
      A.chip.textContent="TRACK COMPLETE";
      A.info.textContent="RGB TARGET-LOCKED RESULT";
      A.video.play().catch(()=>{{}});
      return true;
    }})()""")
else:
    output.eval_js("""(() => {
      const A=window.TL2;
      A.overlay.style.display="none";
      A.progress.style.display="none";
      A.chip.textContent="TRACK COMPLETE";
      A.info.textContent="RESULT READY // TOO LARGE FOR INLINE PREVIEW";
      return true;
    })()""")

print("✅ Proposed Model 4 SEMANTIC-MOTION RE-DETECTOR complete")
print("Tracked output:", final_video)
