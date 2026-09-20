"""Local browser control surface for keyboard teleoperation.

The MuJoCo viewer owns many keyboard shortcuts.  This module deliberately
captures operator keys in a separate localhost page, calls ``preventDefault``
for every control key, and exposes only a small thread-safe state contract to
the simulator.  No external network interface is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Mapping


KEYBOARD_WEB_CONTROL_VERSION = (
    "edgearm-keyboard-web-control-v3-timeout-disarm-idle-status"
)
MOVEMENT_KEYS = frozenset(
    {"w", "s", "a", "d", "e", "r", "ArrowUp", "ArrowDown"}
)
CONTROL_EVENTS = frozenset({"abort", "quit"})


@dataclass(frozen=True)
class KeyboardWebSnapshot:
    held_keys: frozenset[str]
    connected: bool
    heartbeat_age_s: float
    events: tuple[str, ...]


class _KeyboardWebState:
    def __init__(self, heartbeat_timeout_s: float) -> None:
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._lock = threading.Lock()
        self._held_keys: frozenset[str] = frozenset()
        self._rejected_until_release: set[str] = set()
        self._last_input_sequence = -1
        self._last_heartbeat_ns = 0
        self._events: list[str] = []
        self._status: dict[str, Any] = {
            "state": "STARTING",
            "task_zh": "等待仿真初始化",
            "progress": "0/0",
            "rows": 0,
            "coverage": 0.0,
            "gripper_table_angle_degrees": 90.0,
            "message": "正在建立本地控制连接",
        }
        self._frame_jpeg = b""

    def update_input(self, payload: Mapping[str, Any]) -> None:
        raw_held = payload.get("held", ())
        raw_events = payload.get("events", ())
        if not isinstance(raw_held, list) or not isinstance(raw_events, list):
            raise ValueError("held and events must be JSON arrays")
        held = frozenset(str(value) for value in raw_held)
        events = tuple(str(value) for value in raw_events)
        if not held <= MOVEMENT_KEYS:
            raise ValueError("input contains an unsupported movement key")
        if any(event not in CONTROL_EVENTS for event in events):
            raise ValueError("input contains an unsupported control event")
        raw_sequence = payload.get("sequence")
        if raw_sequence is not None and (
            not isinstance(raw_sequence, int)
            or isinstance(raw_sequence, bool)
            or raw_sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        with self._lock:
            fresh = raw_sequence is None or raw_sequence > self._last_input_sequence
            if fresh:
                # A rejected key remains suppressed while the physical/browser
                # key is still down.  A payload without that key is the release
                # edge that arms it again.
                self._rejected_until_release.intersection_update(held)
                self._held_keys = frozenset(held - self._rejected_until_release)
                self._last_heartbeat_ns = time.monotonic_ns()
                if raw_sequence is not None:
                    self._last_input_sequence = raw_sequence
            # Events are edge-triggered and must not be lost merely because an
            # older input request completed after a newer heartbeat.
            self._events.extend(events)

    def reject_keys_until_release(self, keys: frozenset[str]) -> None:
        if not keys <= MOVEMENT_KEYS:
            raise ValueError("cannot reject an unsupported movement key")
        with self._lock:
            self._rejected_until_release.update(keys)
            self._held_keys = frozenset(self._held_keys - keys)

    def snapshot(self, *, drain_events: bool = True) -> KeyboardWebSnapshot:
        now_ns = time.monotonic_ns()
        with self._lock:
            age_s = (
                float("inf")
                if self._last_heartbeat_ns <= 0
                else (now_ns - self._last_heartbeat_ns) * 1.0e-9
            )
            connected = age_s <= self.heartbeat_timeout_s
            if not connected and self._held_keys:
                # A missed key-up during focus loss or page teardown must
                # never resurrect motion when heartbeats return.  Disarm the
                # last held keys until an explicit empty payload releases them.
                self._rejected_until_release.update(self._held_keys)
                self._held_keys = frozenset()
            held = self._held_keys if connected else frozenset()
            events = tuple(self._events)
            if drain_events:
                self._events.clear()
        return KeyboardWebSnapshot(held, connected, age_s, events)

    def set_status(self, status: Mapping[str, Any]) -> None:
        safe = json.loads(json.dumps(dict(status), allow_nan=False))
        with self._lock:
            self._status = safe

    def status_payload(self) -> dict[str, Any]:
        snapshot = self.snapshot(drain_events=False)
        with self._lock:
            status = dict(self._status)
        status.update(
            {
                "version": KEYBOARD_WEB_CONTROL_VERSION,
                "connected": snapshot.connected,
                "heartbeat_age_ms": (
                    None
                    if snapshot.heartbeat_age_s == float("inf")
                    else round(snapshot.heartbeat_age_s * 1000.0, 1)
                ),
                "held": sorted(snapshot.held_keys),
            }
        )
        return status

    def set_frame(self, jpeg: bytes) -> None:
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError("jpeg must be non-empty bytes")
        with self._lock:
            self._frame_jpeg = jpeg

    def frame(self) -> bytes:
        with self._lock:
            return self._frame_jpeg


_CONTROL_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EdgeArm 键盘数据采集</title>
<style>
:root { color-scheme: dark; font-family: -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #101318; color: #f5f7fa; overflow-x: hidden; }
.shell { width: min(1180px, 100vw); margin: auto; padding: 14px; display: grid; grid-template-columns: minmax(500px, 1fr) 340px; gap: 14px; }
.card { background: #191e26; border: 1px solid #303744; border-radius: 14px; overflow: hidden; box-shadow: 0 8px 28px #0006; }
.header { padding: 12px 15px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
.title { font-size: 18px; font-weight: 720; }
.badge { padding: 5px 10px; border-radius: 999px; font-size: 13px; font-weight: 700; background:#502024; color:#ffb9bd; }
.badge.live { background:#123d2b; color:#83f0b1; }
#scene { display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#080a0d; }
.info { padding: 12px 15px 15px; display:grid; gap:8px; }
.task { font-size: 17px; font-weight: 680; color:#ffd77a; }
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.stat { background:#11151b; border-radius:9px; padding:8px; }
.stat small { display:block; color:#9099a8; margin-bottom:3px; }
.stat strong { font-size:16px; }
.panel { padding: 14px; display:grid; align-content:start; gap:12px; }
.hint { color:#b8c0cc; line-height:1.55; font-size:14px; }
.keys { display:grid; grid-template-columns:repeat(3,74px); gap:8px; justify-content:center; user-select:none; }
.key { height:58px; border:1px solid #4b5667; border-radius:10px; background:#272e39; color:#fff; font-size:16px; font-weight:720; cursor:pointer; touch-action:none; }
.key.on { background:#2563eb; border-color:#75a4ff; transform:translateY(1px); }
.blank { visibility:hidden; }
.actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.action { border:0; border-radius:10px; padding:12px 8px; color:#fff; font-size:15px; font-weight:700; cursor:pointer; }
.abort { background:#a15a12; } .quit { background:#7d2630; }
.message { min-height:44px; padding:10px; border-radius:9px; background:#11151b; color:#c9d1dc; line-height:1.4; }
.axis { font-size:13px; color:#8993a3; text-align:center; }
@media(max-width:900px){ .shell{grid-template-columns:1fr;} .card:first-child{order:2;} .panel{order:1;} }
</style>
</head>
<body>
<main class="shell">
  <section class="card">
    <div class="header"><div class="title">EdgeArm 仿真与轨迹采集</div><div id="badge" class="badge">等待连接</div></div>
    <img id="scene" alt="EdgeArm 实时仿真画面">
    <div class="info">
      <div id="task" class="task">等待任务</div>
      <div class="stats">
        <div class="stat"><small>成功进度</small><strong id="progress">0/0</strong></div>
        <div class="stat"><small>本条帧数</small><strong id="rows">0</strong></div>
        <div class="stat"><small>目标覆盖率</small><strong id="coverage">0%</strong></div>
        <div class="stat"><small>夹爪/桌面</small><strong id="tableAngle">90°</strong></div>
      </div>
    </div>
  </section>
  <section class="card panel">
    <div class="title">独立键盘控制台</div>
    <div class="hint">必须让这个页面保持焦点。按住按键连续移动，松开立即停止；控制键不会再进入 MuJoCo 相机。</div>
    <div class="keys">
      <button class="key blank"></button><button class="key" data-key="w">W<br><small>前 +X</small></button><button class="key blank"></button>
      <button class="key" data-key="a">A<br><small>左 +Y</small></button><button class="key" data-key="s">S<br><small>后 -X</small></button><button class="key" data-key="d">D<br><small>右 -Y</small></button>
      <button class="key blank"></button><button class="key" data-key="ArrowUp">↑<br><small>上升 Z</small></button><button class="key blank"></button>
      <button class="key blank"></button><button class="key" data-key="ArrowDown">↓<br><small>下降 Z</small></button><button class="key blank"></button>
      <button class="key" data-key="e">E<br><small>竖起 +</small></button><button class="key blank"></button><button class="key" data-key="r">R<br><small>压低 −</small></button>
    </div>
    <div class="axis">水平朝向自动对准方块；E/R 调整夹爪宽面与桌面的角度（65°–90°）</div>
    <div class="actions"><button id="abort" class="action abort">本条失败重来 X</button><button id="quit" class="action quit">停止采集 Q</button></div>
    <div id="message" class="message">正在连接本机采集进程……</div>
  </section>
</main>
<script>
const held = new Set(); const blockedUntilKeyup = new Set();
// A page reload must start above every request issued by the previous page.
// Millisecond epoch * 1000 remains a safe JavaScript integer and leaves ample
// room for the per-page counter between reloads.
let events = []; let stopped = false; let sequence = Date.now()*1000;
const movement = new Set(['w','s','a','d','e','r','ArrowUp','ArrowDown']);
function normalizedKey(event){ if(event.key.length===1) return event.key.toLowerCase(); return event.key; }
function paint(){ document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('on',held.has(b.dataset.key))); }
function clearHeld(){ held.clear(); paint(); sendInput(); }
function queueEvent(name){ events.push(name); sendInput(); }
async function sendInput(){
  const payload={held:[...held],events:events.splice(0),sequence:sequence++};
  try{ await fetch('/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),cache:'no-store',keepalive:true}); }catch(_e){}
}
window.addEventListener('keydown',e=>{
  const k=normalizedKey(e);
  if(movement.has(k)){ e.preventDefault(); if(blockedUntilKeyup.has(k)) return; held.add(k); paint(); sendInput(); return; }
  if(!e.repeat && k==='x'){ e.preventDefault(); queueEvent('abort'); }
  if(!e.repeat && (k==='q'||k==='Escape')){ e.preventDefault(); queueEvent('quit'); }
},{capture:true});
window.addEventListener('keyup',e=>{ const k=normalizedKey(e); if(movement.has(k)){e.preventDefault();blockedUntilKeyup.delete(k);held.delete(k);paint();sendInput();} },{capture:true});
window.addEventListener('blur',()=>{blockedUntilKeyup.clear();clearHeld();}); document.addEventListener('visibilitychange',()=>{if(document.hidden){blockedUntilKeyup.clear();clearHeld();}}); window.addEventListener('pagehide',()=>{blockedUntilKeyup.clear();clearHeld();});
document.querySelectorAll('[data-key]').forEach(b=>{
  b.addEventListener('pointerdown',e=>{e.preventDefault();b.setPointerCapture(e.pointerId);if(blockedUntilKeyup.has(b.dataset.key)) return;held.add(b.dataset.key);paint();sendInput();});
  for(const name of ['pointerup','pointercancel','lostpointercapture']) b.addEventListener(name,()=>{blockedUntilKeyup.delete(b.dataset.key);held.delete(b.dataset.key);paint();sendInput();});
});
document.getElementById('abort').onclick=()=>queueEvent('abort'); document.getElementById('quit').onclick=()=>queueEvent('quit');
setInterval(sendInput,60);
async function updateStatus(){
 try{
  const r=await fetch('/status?'+Date.now(),{cache:'no-store'}); const s=await r.json();
  const badge=document.getElementById('badge'); const waiting=['READY_WAIT_INPUT','PAUSED_IDLE','IK_BLOCKED'].includes(s.state); badge.textContent=s.connected?(waiting?'已连接 · 等待操作':'已连接 · 正在采集'):'控制未连接'; badge.classList.toggle('live',s.connected);
  document.getElementById('task').textContent=s.task_zh||'等待任务'; document.getElementById('progress').textContent=s.progress||'0/0';
  document.getElementById('rows').textContent=String(s.rows??0); document.getElementById('coverage').textContent=((Number(s.coverage)||0)*100).toFixed(1)+'%';
  document.getElementById('tableAngle').textContent=(Number(s.gripper_table_angle_degrees)||90).toFixed(1)+'°';
  document.getElementById('message').textContent=s.message||s.state||'';
  if(s.state==='IK_BLOCKED' && held.size){
   let changed=false; const rejected=Array.isArray(s.rejected_keys)?s.rejected_keys:[];
   for(const k of rejected){ if(held.has(k)){blockedUntilKeyup.add(k);held.delete(k);changed=true;} }
   if(changed){paint();sendInput();}
  }
 }catch(_e){ document.getElementById('badge').textContent='采集进程离线'; document.getElementById('badge').classList.remove('live'); }
}
setInterval(updateStatus,250); updateStatus();
const scene=document.getElementById('scene'); function nextFrame(){scene.src='/frame.jpg?'+Date.now();} scene.onload=()=>setTimeout(nextFrame,55); scene.onerror=()=>setTimeout(nextFrame,250); nextFrame();
</script>
</body></html>
"""


def _handler_for(state: _KeyboardWebState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(_CONTROL_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/status":
                payload = json.dumps(
                    state.status_payload(), ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
                self._send(payload, "application/json; charset=utf-8")
                return
            if path == "/frame.jpg":
                frame = state.frame()
                if not frame:
                    self._send(b"frame not ready", "text/plain", HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self._send(frame, "image/jpeg")
                return
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/input":
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("input body must be a JSON object")
                state.update_input(payload)
            except (ValueError, json.JSONDecodeError) as error:
                self._send(str(error).encode("utf-8"), "text/plain", HTTPStatus.BAD_REQUEST)
                return
            self._send(b"{}", "application/json")

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


class KeyboardWebControlServer:
    """Threaded localhost server used by the simulation control loop."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        heartbeat_timeout_s: float = 0.35,
    ) -> None:
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("keyboard control server must remain localhost-only")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer in [0,65535]")
        if heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be positive")
        self._state = _KeyboardWebState(heartbeat_timeout_s)
        self._server = ThreadingHTTPServer((host, port), _handler_for(self._state))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("keyboard web server already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="edgearm-keyboard-web-v1",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> KeyboardWebSnapshot:
        return self._state.snapshot()

    def set_status(self, status: Mapping[str, Any]) -> None:
        self._state.set_status(status)

    def set_frame(self, jpeg: bytes) -> None:
        self._state.set_frame(jpeg)

    def reject_keys_until_release(self, keys: frozenset[str]) -> None:
        self._state.reject_keys_until_release(keys)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None


__all__ = [
    "CONTROL_EVENTS",
    "KEYBOARD_WEB_CONTROL_VERSION",
    "MOVEMENT_KEYS",
    "KeyboardWebControlServer",
    "KeyboardWebSnapshot",
]
