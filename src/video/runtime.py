"""Host-owned playback runtime injected into explainer compositions.

The model only provides a deterministic, seekable frame function:

    window.player = { duration: <seconds>, render(t) { ... } };

This module injects:
  - a small base stylesheet for a consistent, high-quality baseline, and
  - a runtime that owns the clock, play/pause/seek/restart, end detection, and a
    ``postMessage`` bridge so the React host can drive playback across a
    sandboxed iframe.

Message protocol
  host  -> iframe: { source:"host", action:"play"|"pause"|"seek"|"restart", time? }
  iframe -> host : { source:"player", type:"ready"|"time"|"ended"|"noplayer", t?, duration? }
"""

from __future__ import annotations

# Marker the host/tests can look for to confirm the runtime was injected.
RUNTIME_MARKER = "__lf_player_runtime__"

BASE_CSS = """
<style id="__lf_base_css__">
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    background: #0b0f1a;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>
""".strip()

PLAYER_RUNTIME_JS = """
<script id="__lf_player_runtime__">
(function () {
  var DEFAULT_DURATION = 24;
  var clock = { t: 0, playing: false, raf: null, last: 0, duration: DEFAULT_DURATION };

  function player() { return window.player; }

  function duration() {
    var p = player();
    var d = p && typeof p.duration === 'number' && p.duration > 0 ? p.duration : DEFAULT_DURATION;
    clock.duration = d;
    return d;
  }

  function renderAt(t) {
    var p = player();
    if (p && typeof p.render === 'function') {
      try { p.render(t); } catch (e) { /* ignore per-frame draw errors */ }
    }
  }

  function post(msg) {
    msg.source = 'player';
    try { window.parent.postMessage(msg, '*'); } catch (e) {}
  }

  var lastPosted = -1;
  function postTime(force) {
    var now = performance.now();
    if (force || now - lastPosted > 90) {
      lastPosted = now;
      post({ type: 'time', t: clock.t, duration: clock.duration });
    }
  }

  function tick(ts) {
    if (!clock.playing) return;
    if (!clock.last) clock.last = ts;
    var dt = (ts - clock.last) / 1000;
    clock.last = ts;
    clock.t += dt;
    var d = duration();
    if (clock.t >= d) {
      clock.t = d;
      renderAt(clock.t);
      pause();
      postTime(true);
      post({ type: 'ended' });
      return;
    }
    renderAt(clock.t);
    postTime(false);
    clock.raf = requestAnimationFrame(tick);
  }

  function play() {
    if (clock.playing) return;
    if (clock.t >= duration()) clock.t = 0;
    clock.playing = true;
    clock.last = 0;
    clock.raf = requestAnimationFrame(tick);
  }

  function pause() {
    clock.playing = false;
    if (clock.raf) { cancelAnimationFrame(clock.raf); clock.raf = null; }
  }

  function seek(t) {
    var d = duration();
    clock.t = Math.max(0, Math.min(t, d));
    renderAt(clock.t);
    postTime(true);
  }

  function restart() { clock.t = 0; renderAt(0); play(); postTime(true); }

  window.addEventListener('message', function (ev) {
    var m = ev.data;
    if (!m || m.source !== 'host') return;
    if (m.action === 'play') play();
    else if (m.action === 'pause') pause();
    else if (m.action === 'seek') seek(m.time || 0);
    else if (m.action === 'restart') restart();
  });

  function boot(attempt) {
    if (player() && typeof player().render === 'function') {
      var d = duration();
      renderAt(0);
      post({ type: 'ready', duration: d });
      postTime(true);
      play(); // autoplay once; host controls can pause/seek/replay
      return;
    }
    if (attempt > 30) { post({ type: 'noplayer' }); return; }
    setTimeout(function () { boot(attempt + 1); }, 50);
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    boot(0);
  } else {
    document.addEventListener('DOMContentLoaded', function () { boot(0); });
  }
})();
</script>
""".strip()


def inject(html: str) -> str:
    """Insert the base CSS and player runtime into a composition document."""
    out = html

    if "</head>" in out:
        out = out.replace("</head>", BASE_CSS + "\n</head>", 1)
    elif "<head>" in out:
        out = out.replace("<head>", "<head>\n" + BASE_CSS, 1)
    else:
        out = BASE_CSS + "\n" + out

    if "</body>" in out:
        out = out.replace("</body>", PLAYER_RUNTIME_JS + "\n</body>", 1)
    else:
        out = out + "\n" + PLAYER_RUNTIME_JS

    return out
