# Gesture Volume Controller

## Overview

Control your Windows system volume with your hand. This app uses your webcam and MediaPipe Hands to detect the distance between your thumb and index finger and maps it to system volume via CoreAudio (pycaw).

## Requirements

- Windows 10/11 (CoreAudio)
- Python 3.10 or 3.11 recommended
- A working webcam

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you see an execution policy error when activating the venv, run PowerShell as Administrator once and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Run

```powershell
python gesture_volume_controller.py
```

### Optional flags

- `--camera-index <n>`: choose a different camera (default 0)
- `--width <px> --height <px>`: set capture size (default 640x480)
- `--no-flip`: disable mirror view

Example:

```powershell
python gesture_volume_controller.py --camera-index 1 --width 1280 --height 720
```

## How it works

- Detects hand landmarks using MediaPipe Hands.
- Measures distance between thumb tip (id 4) and index tip (id 8).
- Maps distance to a 0–100% volume level using `SetMasterVolumeLevelScalar`.
- Shows a simple on-screen volume bar.

## Tips & Troubleshooting

- Black window or no camera: close other apps using the camera; try `--camera-index 1`.
- No volume change: ensure you’re on Windows and not in a remote desktop session; verify the system audio device is available.
- Performance: reduce resolution with `--width 640 --height 480`.
- Exit: press `q` or `Esc`.

## License

For educational/demo use.
