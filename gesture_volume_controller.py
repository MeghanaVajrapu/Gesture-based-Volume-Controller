from __future__ import annotations

import argparse
import math
import platform
import sys
import time
import traceback

import cv2
import numpy as np
from ctypes import POINTER, cast

# Try importing pycaw/comtypes (Windows volume control)
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAS_PYCAW = True
except Exception as exc:  # noqa: BLE001
    HAS_PYCAW = False
    AudioUtilities = None
    IAudioEndpointVolume = None
    CLSCTX_ALL = None
    _pycaw_import_exc = exc

# Try importing mediapipe; if missing, we will provide a non-crashing fallback mode
try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    HAS_MEDIAPIPE = True
except Exception as exc:  # noqa: BLE001
    HAS_MEDIAPIPE = False
    mp = None
    mp_drawing = None
    mp_drawing_styles = None
    mp_hands = None
    _mediapipe_import_exc = exc


def _get_audio_endpoint():
    """Return a pycaw IAudioEndpointVolume object or None if unavailable."""
    if not HAS_PYCAW:
        print("pycaw not available: volume API calls will be simulated (printed).")
        return None
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as exc:
        print(f"Audio initialization failed: {exc}")
        return None


def _set_volume_scalar(volume_obj, scalar: float) -> None:
    """Set system volume scalar [0.0, 1.0] if possible; otherwise print simulated change."""
    scalar = float(np.clip(scalar, 0.0, 1.0))
    if volume_obj is None:
        print(f"[SIM] Set volume to {scalar*100:.1f}%")
        return
    try:
        volume_obj.SetMasterVolumeLevelScalar(scalar, None)
    except Exception as exc:
        print(f"Warning: failed to set system volume: {exc}")


def _compute_3d_distance(lm1, lm2, image_width: int, image_height: int) -> float:
    """Compute pseudo-3D distance between two landmarks."""
    x1, y1, z1 = lm1.x * image_width, lm1.y * image_height, lm1.z * image_width
    x2, y2, z2 = lm2.x * image_width, lm2.y * image_height, lm2.z * image_width
    return float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))


def run_mediapipe_mode(camera_index: int, width: int, height: int, flip: bool) -> int:
    """Run the gesture -> volume controller using mediapipe."""
    if platform.system() != "Windows":
        print("Note: Running on non-Windows OS. pycaw will not be used; volume will be simulated.")

    volume = _get_audio_endpoint()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera index {camera_index}")
        return 4

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    win_name = "Gesture Volume Controller (Mediapipe)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # tuned parameters
    min_dist = 25.0
    max_dist = 260.0
    alpha = 0.25
    set_threshold = 0.01
    api_call_min_interval = 0.03

    smoothed_scalar = None
    last_set_scalar = None
    last_api_call = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    time.sleep(0.03)
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                if flip:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                h, w, _ = frame.shape

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    my_hand = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame,
                        my_hand,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    lm_thumb = my_hand.landmark[4]
                    lm_index = my_hand.landmark[8]

                    length = _compute_3d_distance(lm_thumb, lm_index, w, h)
                    length_clamped = max(min_dist, min(length, max_dist))

                    mapped = float(np.interp(length_clamped, [min_dist, max_dist], [0.0, 1.0]))

                    if smoothed_scalar is None:
                        smoothed_scalar = mapped
                    else:
                        smoothed_scalar = alpha * mapped + (1.0 - alpha) * smoothed_scalar

                    now = time.time()
                    if last_set_scalar is None:
                        need_set = True
                    else:
                        need_set = abs(smoothed_scalar - last_set_scalar) > set_threshold and (now - last_api_call) > api_call_min_interval

                    if need_set:
                        _set_volume_scalar(_get_audio_endpoint(), smoothed_scalar)
                        last_set_scalar = float(np.clip(smoothed_scalar, 0.0, 1.0))
                        last_api_call = now

                    vol_bar = int(np.interp(length_clamped, [min_dist, max_dist], [400, 150]))
                    vol_per = int(np.clip(round(smoothed_scalar * 100), 0, 100))

                    x1, y1 = int(lm_thumb.x * w), int(lm_thumb.y * h)
                    x2, y2 = int(lm_index.x * w), int(lm_index.y * h)
                    cv2.circle(frame, (x1, y1), 12, (255, 255, 255), -1)
                    cv2.circle(frame, (x2, y2), 12, (255, 255, 255), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                else:
                    if smoothed_scalar is not None:
                        smoothed_scalar *= 0.98
                        vol_per = int(np.clip(round(smoothed_scalar * 100), 0, 100))
                        vol_bar = int(np.interp(max(min_dist, min(max_dist, (1 - smoothed_scalar) * max_dist)), [min_dist, max_dist], [400, 150]))
                    else:
                        vol_bar = 400
                        vol_per = 0

                cv2.rectangle(frame, (50, 150), (85, 400), (200, 200, 200), 2)
                cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (50, 50, 50), -1)
                cv2.putText(frame, f"{vol_per} %", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                cv2.imshow(win_name, frame)

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user. Exiting...")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Exit key pressed. Quitting...")
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception:
            print("Unexpected error in mediapipe mode:\n", traceback.format_exc())
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return 0


def run_fallback_mode(camera_index: int, width: int, height: int, flip: bool) -> int:
    """Fallback mode when mediapipe is missing: allow keyboard control and webcam preview."""
    if platform.system() != "Windows":
        print("Note: Running on non-Windows OS. pycaw will not be used; volume will be simulated.")

    volume_obj = _get_audio_endpoint()

    cap = None
    have_camera = False
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        have_camera = cap.isOpened()
        if have_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    except Exception:
        have_camera = False

    win_name = "Gesture Volume Controller (Fallback)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    current_scalar = 0.5
    muted = False
    _set_volume_scalar(volume_obj, current_scalar)

    try:
        while True:
            if have_camera:
                success, frame = cap.read()
                if not success:
                    time.sleep(0.02)
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                if flip:
                    frame = cv2.flip(frame, 1)
            else:
                frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, "Fallback mode (no mediapipe)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(frame, "Use +/- to change volume, m to mute, q to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            vol_bar = int(np.interp(current_scalar, [0.0, 1.0], [400, 150]))
            vol_per = int(round(current_scalar * 100))

            cv2.rectangle(frame, (50, 150), (85, 400), (200, 200, 200), 2)
            cv2.rectangle(frame, (50, vol_bar), (85, 400), (50, 50, 50), -1)
            status = f"{vol_per}% {'MUTED' if muted else ''}".strip()
            cv2.putText(frame, status, (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            cv2.imshow(win_name, frame)

            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Exiting...")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord('+'), ord('=')):
                if not muted:
                    current_scalar = float(min(1.0, current_scalar + 0.05))
                    _set_volume_scalar(volume_obj, current_scalar)
            elif key == ord('-'):
                if not muted:
                    current_scalar = float(max(0.0, current_scalar - 0.05))
                    _set_volume_scalar(volume_obj, current_scalar)
            elif key == ord('m'):
                if volume_obj is None:
                    muted = not muted
                    print(f"[SIM] Mute toggled: {muted}")
                else:
                    try:
                        if hasattr(volume_obj, 'SetMute'):
                            muted = not muted
                            volume_obj.SetMute(int(muted), None)
                        else:
                            muted = not muted
                            if muted:
                                _set_volume_scalar(volume_obj, 0.0)
                            else:
                                _set_volume_scalar(volume_obj, current_scalar)
                    except Exception as exc:
                        print(f"Warning toggling mute: {exc}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


# --- small internal tests ---
class _DummyLM:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def self_test() -> int:
    """Run lightweight tests useful for CI or debugging in restricted environments."""
    print("Running self-tests...")

    lm1 = _DummyLM(0.1, 0.2, 0.0)
    lm2 = _DummyLM(0.4, 0.6, 0.05)
    w, h = 640, 480
    d = _compute_3d_distance(lm1, lm2, w, h)
    ex = math.sqrt(((lm2.x - lm1.x) * w) ** 2 + ((lm2.y - lm1.y) * h) ** 2 + ((lm2.z - lm1.z) * w) ** 2)
    print(f"3D distance computed: {d:.4f}, expected: {ex:.4f}")
    if not np.isclose(d, ex, atol=1e-6):
        print("Distance test failed")
        return 2

    print("mediapipe available:", HAS_MEDIAPIPE)
    if not HAS_MEDIAPIPE:
        print("mediapipe import error:", getattr(globals(), '_mediapipe_import_exc', 'N/A'))

    print("pycaw available:", HAS_PYCAW)
    if not HAS_PYCAW:
        print("pycaw import error:", getattr(globals(), '_pycaw_import_exc', 'N/A'))

    print("Self-tests passed")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Control system volume with hand gestures (robust)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--no-flip", action="store_true", help="Do not mirror the camera image")
    parser.add_argument("--fallback", action="store_true", help="Force fallback mode (do not attempt mediapipe)")
    parser.add_argument("--self-test", action="store_true", help="Run internal self-tests and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    use_mediapipe = HAS_MEDIAPIPE and (not args.fallback)
    if not HAS_MEDIAPIPE and not args.fallback:
        print("Warning: mediapipe is not available. Falling back to keyboard/camera demo mode.")
        print("Install mediapipe to enable real gesture control: python -m pip install mediapipe")

    if use_mediapipe:
        return run_mediapipe_mode(args.camera_index, args.width, args.height, flip=not args.no_flip)
    else:
        return run_fallback_mode(args.camera_index, args.width, args.height, flip=not args.no_flip)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("Fatal error:\n", traceback.format_exc())
        sys.exit(1)
