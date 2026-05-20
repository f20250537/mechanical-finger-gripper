# Mechanical Finger Gripper

Vision-assisted 2-finger robotic gripper control app — BITS Pilani Dubai, Spring 2026.

## Team

| Name | ID |
|------|----|
| Pranav Gadde | 2025ACPS0549U |
| Shrujan Ravi Gowda | 2025ACPS0544U |
| Bharath Kshathriya | 2025ACPS0552U |
| Karthik Pamarthi | 2025ACPS0537U |

Python handles webcam vision, YOLOv8 object detection, Arduino serial communication, and a state machine that drives the full grip cycle. The desktop UI runs inside a `pywebview` window served from `app.html`.

The app is fully demoable without hardware. If no Arduino is found, it enters **simulated mode** — all serial commands are echoed in the log, the servo angle is tracked virtually, and the full state machine runs normally.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

YOLOv8n weights (`yolov8n.pt`) are downloaded automatically by `ultralytics` on first run.

## Arduino

1. Open `arduino/gripper.ino` in the Arduino IDE.
2. Select your board and port.
3. Upload.
4. Run `python main.py` — the app auto-connects.

The app scans `/dev/cu.usbmodem*`, `/dev/cu.usbserial*` (macOS) and `COM*` (Windows). Falls back to simulated mode if nothing is found.

## Wiring

| Wire | Connection |
|------|-----------|
| Servo signal | Arduino pin 9 |
| Servo power (red) | External 5V supply |
| Servo ground (black) | Shared GND with Arduino |

> Do **not** power the servo from the Arduino 5V pin. Use an external supply.

## Serial Protocol

Baud rate: `9600`

| Command | Description |
|---------|-------------|
| `OPEN` | Smooth move to 0° |
| `CLOSE:N` | Smooth move to N° (0–90) |
| `STATUS` | Returns `ANGLE:<current>` |
| `PING` | Returns `PONG` |

## State Machine Flow

```
IDLE → DETECTED → CALCULATING → INSERTING → GRIPPING → SUCCESS/FAIL → IDLE
```

- **IDLE**: Gripper open, scanning for objects
- **DETECTED**: Object locked on (2 consecutive frames), grip angle calculated
- **CALCULATING**: Confirming open position (1.2s)
- **INSERTING**: Waiting for object to be placed (4s)
- **GRIPPING**: Servo closes to calculated angle
- **SUCCESS/FAIL**: Result shown, gripper returns to open

Manual controls (OPEN / CLOSE / GRIP / RESET / EMERGENCY STOP) override the state machine at any time.

## Camera

The app tries camera indices 0 → 1 → 2 using macOS AVFoundation. On first run, grant Camera permission in **System Settings → Privacy & Security → Camera**.

```bash
# Check available serial ports when plugging/unplugging Arduino
ls /dev/cu.*
```

## Requirements

- Python 3.9+
- OpenCV, ultralytics (YOLOv8), pyserial, pywebview, websockets
- Arduino Uno/Nano with a standard RC servo on pin 9
