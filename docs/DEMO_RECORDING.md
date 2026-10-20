# Radio Map demonstration

`media/radio-map-demo.mp4` records the running native application window and
actual mouse/keyboard interactions. Scene, transmitter and beam selection
were performed through the UI; no application-state scripting was used.
The RSS calculations are real Sionna RT runs.

The approximately 60-second demonstration, followed by a six-second source/attribution card, shows:

1. Choose PolyU campus from the scene menu.
2. Click Place TX, then place the base station on a rooftop.
3. Click Aim, then click a target in the scene.
4. Wait for automatic calculation and six overlaid height maps.
5. Click Select beam, then select a discrete tile on the beam sphere.
6. Show the updated direction and completed six-height results.

Only the Radio Map workspace is shown. Defaults remain 3 m cells and
8M samples per height. Both selected beams (116 and 177) completed
8,000,000 samples at each of the six heights.

The primary screen is 3840 x 2160 at 150% system scaling. The maximized
application content area is 3840 x 2054. The video preserves that native
content resolution and UI scale, without resizing or stretching the UI.
English captions occupy unused space in the existing top bar.

Idle pauses are removed. Calculation sections play at 4x and are labeled;
selection actions play at normal speed. This edited demonstration is not a
frame-rate benchmark. H.264 MP4, 24 fps, 1,580 frames, 65.83 seconds,
no narration or music. Capture events and edit intervals are retained in
`media/radio-map-recording.json`.

To capture another interactive session, install FFmpeg separately and run
from the repository root with the application environment:

```powershell
.\.venv\Scripts\python.exe .\tools\record_interactive_radio_map.py --ffmpeg C:\Tools\ffmpeg.exe --work .\capture
```

Operate the visible application normally. To stop, write `{"stop": true}`
to `capture/control.json`. The helper records `native-raw.mp4` and passive
click/progress metadata. It does not perform UI actions or edit the raw clip.
Keep raw footage and generated caches out of source control. FFmpeg is not
bundled or required by the application itself.
