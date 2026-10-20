"""Passive native-size app recorder. All scene/beam actions are user input.

This helper does not resize the application or change its UI scale.
It records the native application window and actual mouse cursor using FFmpeg.
An external JSON control file supplies captions and stops recording; it cannot
invoke application actions. Encoding tools are not application dependencies.
"""
import argparse
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "sionna-rt-gui/src"))
os.environ["PYTHONPATH"] = str(root / "sionna-rt-gui/src")
os.environ["MPLBACKEND"] = "Agg"
import ctypes
import json
import subprocess
import time
import polyscope as ps
import polyscope.imgui as ui
from sionna_rt_gui.config import load_config
from sionna_rt_gui.gui import SionnaRtGui

parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg", required=True)
parser.add_argument("--work", required=True)
args = parser.parse_args()
work = Path(args.work).resolve()
work.mkdir(parents=True, exist_ok=True)
control = work / "control.json"
os.chdir(root)
app = SionnaRtGui(load_config("polyu-stylized-config.yaml"))
ps.set_user_callback(app.tick)
ps.show(forFrames=6)
width, height = ps.get_buffer_size()
fps = 24
metadata = dict(primary_screen=[ctypes.windll.user32.GetSystemMetrics(0),
                                ctypes.windll.user32.GetSystemMetrics(1)],
                framebuffer=[width, height], ui_scale=app.ui_scale, fps=fps,
                actions="actual mouse/keyboard input", events=[])
(work / "capture-info.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
encoder = subprocess.Popen([args.ffmpeg, "-y", "-loglevel", "warning",
    "-f", "gdigrab", "-framerate", str(fps), "-draw_mouse", "1",
    "-i", "title=Digital Twin Studio", "-an", "-c:v", "libx264",
    "-preset", "ultrafast", "-crf", "19", "-pix_fmt", "yuv420p",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    str(work / "native-raw.mp4")], stdin=subprocess.PIPE)
start = time.monotonic()
frames = 0
last_stage = None
last_down = False
last_progress = 0
def capture_tick():
    global frames, last_stage, last_down, last_progress
    app.tick()
    try:
        settings = json.loads(control.read_text()) if control.exists() else {}
    except (ValueError, OSError):
        settings = {}
    if settings.get("stop") or time.monotonic()-start > 900:
        ps.unshow()
        return
    stage = settings.get("stage", "Scene selection")
    now = time.monotonic()-start
    if stage != last_stage:
        metadata["events"].append(dict(kind="stage", at=now, text=stage))
        last_stage = stage
    io = ui.GetIO()
    down = bool(io.MouseDown[0])
    if down and not last_down:
        metadata["events"].append(dict(kind="click", at=now, xy=list(io.MousePos)))
    last_down = down
    frames = int(now*fps)
    if now-last_progress >= 3:
        air = getattr(app.coverage_jobs, "air", None)
        state = dict(at=now, stage=stage, scene=Path(app.cfg.scene_filename).parent.name,
                     beam=app.beam_demo.selected, refining=app.rm_refining,
                     samples=[int(x["acc"].counts[0]) for x in air.layers] if air else [])
        (work / "progress.json").write_text(json.dumps(state))
        metadata["events"].append(dict(kind="progress", **state))
        (work / "capture-info.json").write_text(json.dumps(metadata, indent=2))
        last_progress = now

ps.set_user_callback(capture_tick)
try:
    ps.show()
finally:
    encoder.stdin.write(b"q")
    encoder.stdin.flush()
    encoder.stdin.close()
    code = encoder.wait()
    metadata["estimated_capture_frames"] = frames
    metadata["capture_elapsed_seconds"] = frames/fps
    (work / "capture-info.json").write_text(json.dumps(metadata, indent=2))
    from sionna_rt_gui.responsive_render import close
    close(app)
    if app.coverage_jobs: app.coverage_jobs.close()
    if app.link_jobs: app.link_jobs.close()
    if code:
        raise RuntimeError(f"Encoder failed: {code}")
