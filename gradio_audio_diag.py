#!/usr/bin/env python3
"""
gradio_audio_diag.py  --  isolate WHY audio won't play in the labeling GUI.

Renders the SAME 3-second tone three different ways on one page:

  [A] base64 data: URI inside gr.HTML   (what phase2_classify.py does now)
  [B] native gr.Audio from a numpy tuple (the supported, version-robust path)
  [C] native gr.Audio from a temp .wav file on disk

Run this on the Spark box, then open the URL and try each player.

    python3 gradio_audio_diag.py            # serves on 0.0.0.0:7861

WHAT THE RESULT TELLS YOU
-------------------------
* B and/or C play, A spins:
      -> Gradio 6's HTML component is not wiring up <audio> data-URIs.
         The fix is to serve audio via native gr.Audio (the patched
         phase2_classify.py does exactly this).

* NOTHING plays in your normal browser window, BUT everything plays when
  you open the SAME url in a fresh Private/Incognito window:
      -> Your browser cached a stale Gradio frontend from today's version
         swapping. Clear site data for 134.89.11.107 (or just keep using
         Incognito). No code change needed.

* Nothing plays even in Incognito:
      -> The Gradio install itself is inconsistent. Rebuild a clean venv
         pinned to ONE Gradio version (see the commands I gave you).

* The static text + the tiny PNG below show up, confirming gr.HTML renders
  static content even when audio is dead -- matching your report.

IMPORTANT: open this in a PRIVATE / INCOGNITO window the first time, so a
cached frontend can't contaminate the result.
"""
import io, base64, tempfile, os
import numpy as np
import soundfile as sf
import gradio as gr

SR = 32000
DUR = 3.0
t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

# --- int16 PCM, the only WAV encoding every browser reliably decodes -------
pcm16 = (np.clip(tone, -1.0, 1.0) * 32767.0).round().astype(np.int16)

# [A] base64 data: URI, exactly like the current _make_audio_b64 path -------
_buf = io.BytesIO()
sf.write(_buf, pcm16, SR, format="WAV", subtype="PCM_16")
_buf.seek(0)
_wav_bytes = _buf.read()
wav_b64 = "data:audio/wav;base64," + base64.b64encode(_wav_bytes).decode()

# a tiny 1x1 PNG data URI, to confirm gr.HTML renders static media ----------
_png = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c6360000002000155f5a2900000000"
    "049454e44ae426082")).decode()

# [C] temp wav file on disk -------------------------------------------------
_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
sf.write(_tmp.name, pcm16, SR, format="WAV", subtype="PCM_16")
_tmp.close()

html_A = (
    "<div style='background:#1e293b;padding:12px;border-radius:8px;color:#e2e8f0;"
    "font-family:monospace'>"
    "<b>[A] base64 data: URI inside gr.HTML</b> "
    "(the approach that is currently failing)<br>"
    f"<img src='data:image/png;base64,{_png}' "
    "style='width:24px;height:24px;image-rendering:pixelated;border:1px solid #64748b'/>"
    " &larr; if you can see this 1&times;1 png box, gr.HTML renders static media<br>"
    f"<audio controls preload='metadata' style='width:100%;margin-top:8px' "
    f"src='{wav_b64}'></audio>"
    "</div>"
)

print("gradio version:", gr.__version__)
print("soundfile:", sf.__version__, "libsndfile:", sf.__libsndfile_version__)
print("WAV subtype written:", sf.SoundFile(io.BytesIO(_wav_bytes)).subtype)
print("temp wav:", _tmp.name)

with gr.Blocks(title="Gradio audio diagnostic") as demo:
    gr.Markdown("# Gradio audio diagnostic\nTry to play each of the three "
                "players below. Open this page in a **Private/Incognito** "
                "window the first time.")
    gr.HTML(html_A)
    gr.Markdown("**[B] native `gr.Audio` from a numpy tuple** "
                "(the robust, supported path)")
    gr.Audio(value=(SR, pcm16), interactive=False, show_label=False)
    gr.Markdown("**[C] native `gr.Audio` from a temp `.wav` file**")
    gr.Audio(value=_tmp.name, interactive=False, show_label=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, show_error=True)
