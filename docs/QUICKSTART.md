# Quick start

## Install

Install Python 3.12 x64 and an NVIDIA GPU driver, then run `setup.ps1` from the
repository root. It creates `.venv` and installs the local application.
`requirements-tested.txt` records versions in the working development runtime;
setup uses these as constraints.

Run `start.bat` or `powershell -ExecutionPolicy Bypass -File .\start.ps1`.
The batch launcher retains the console on failure so errors remain visible.
Paths are resolved relative to the repository, not the original developer PC.

## Radio Map

1. Select a scene.
2. Click **Place TX**, then choose a supported point in the scene.
3. Choose **Aim** and click a target, or use **Select beam** and choose a
   discrete spherical tile. Entering Aim mode alone does not start calculation.
4. Wait for six layer labels to become **Ready**. The default budget is 8M per
   layer and the cell size is 3m.
5. The six layers overlay the model automatically. Use a layer's display
   control to isolate it in 3D, or collapse the results panel for more space.
6. Changing the beam starts a fresh calculation. The cyan target marker shows
   the direction for both selection methods.

Fast and Original change visual lighting only, not the RF solver or material
parameters. Original temporarily uses Fast during navigation and refines after
the camera stops.

## Local files

`.air-cache`, `.coverage-cache`, `.link-cache`, `.display-cache`, Matplotlib
caches, and UI settings are regenerated locally and ignored by Git. Do not
upload the virtual environment or runtime caches.

Scene Z is not a universal AGL or sea-level datum. Campus source heights and
the simplified support geometry have distinct assumptions; consult each
scene's calibration/provenance records.

## Troubleshooting

- If Python is missing, install 3.12 x64 and check `py -3.12 --version`.
- If CUDA initialization fails, check the NVIDIA driver and GPU support.
- A first launch can spend time preparing lighting caches.
- If launch fails, use the console traceback rather than repeatedly
  double-clicking; the launcher leaves the error visible.
- During coverage calculation, worker details are written under `.air-cache`.
- The application uses some version-sensitive Polyscope/DrJit APIs. Keep the
  tested rendering stack until a newer version is explicitly validated.

## Checks

```powershell
.\.venv\Scripts\python.exe .\tools\validate_repository.py
.\.venv\Scripts\python.exe -m unittest discover -s .\sionna-rt-gui\test
```

These checks do not establish electromagnetic accuracy. The first validates
filenames and asset references; the second exercises focused application logic.
