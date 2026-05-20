#!/usr/bin/env python3
import asyncio
import base64
import json
import sys
import glob
import threading
import time
import http.server
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed. pip install websockets")
    exit(1)

try:
    import webview
except ImportError:
    webview = None

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[SERIAL] pyserial not installed — simulated mode")

# ============================================================================
# CAMERA INITIALIZATION (deferred — called after webview window is up)
# ============================================================================

cap = None
CAM_FOUND = False

def init_camera():
    """Open USB camera using AVFoundation, trying indices 0→1→2."""
    global cap, CAM_FOUND
    for idx in [0, 1, 2]:
        try:
            c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if not c.isOpened():
                print(f"[CAM] index {idx}: not opened")
                c.release()
                continue
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_FPS, 30)
            ret, frame = c.read()
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[CAM] index {idx}: ret={ret} size={w}x{h}")
            if ret and frame is not None and frame.size > 0:
                cap = c
                CAM_FOUND = True
                print(f"[CAM] OK — using index {idx} ({w}x{h})")
                return
            c.release()
        except Exception as e:
            print(f"[CAM] index {idx} error: {e}")
    CAM_FOUND = False
    print("[CAM] FAILED — simulated mode")

# ============================================================================
# YOLO MODEL + GRIP MAP
# ============================================================================

GRIP_MAP = {
    "cell phone":  65,
    "pen":         78,
    "pencil":      78,
    "marker":      72,
    "scissors":    68,
    "remote":      62,
    "book":        45,
    "notebook":    45,
    "ruler":       80,
    "mouse":       55,
    "calculator":  58,
    "credit card": 82,
    "knife":       75,
    "fork":        76,
    "spoon":       76,
    "toothbrush":  74,
    "bottle":      42,
    "cup":         48,
    "vase":        44,
    "banana":      58,
    "apple":       52,
    "orange":      50,
    "sandwich":    40,
    "laptop":      30,
    "keyboard":    35,
    "clock":       50,
    "teddy bear":  45,
    "hair drier":  55,
    "umbrella":    60,
    "tie":         75,
    "default":     65,
}

# Only objects in this set trigger the gripper state machine
GRIPPABLE = set(k for k in GRIP_MAP if k != "default")

model = None
if YOLO:
    try:
        model = YOLO('yolov8n.pt')
        print("[MODEL] YOLOv8n loaded")
    except Exception as e:
        print(f"[MODEL] Load failed: {e}")

# ============================================================================
# SERIAL — Arduino communication + simulation
# ============================================================================

serial_conn      = None
currentAngle_sim = 0
serial_log_queue = deque()
serial_log       = []

def push_serial_log(msg):
    serial_log_queue.append(msg)
    serial_log.append(msg)

def send_serial(cmd):
    """Send a command to Arduino (or simulate if not connected)."""
    global serial_conn, currentAngle_sim
    if serial_conn:
        try:
            serial_conn.write((cmd + '\n').encode())
            push_serial_log(f'SER > {cmd}')
            return True
        except Exception as e:
            push_serial_log(f'SER ERR > {e}')
            serial_conn = None
            # fall through to sim

    # ── Simulated responses ──────────────────────────────────────────────
    if cmd == 'OPEN':
        currentAngle_sim = 0
        push_serial_log('SIM > OPEN → DONE')
    elif cmd.startswith('CLOSE:'):
        try:
            currentAngle_sim = int(cmd.split(':')[1])
        except ValueError:
            pass
        push_serial_log(f'SIM > {cmd} → DONE')
    elif cmd == 'STATUS':
        push_serial_log(f'SIM > ANGLE:{currentAngle_sim}')
    elif cmd == 'PING':
        push_serial_log('SIM > PONG')
    else:
        push_serial_log(f'SIM > {cmd}')
    return True

async def read_serial_response(timeout=2.0):
    """Read one line response from Arduino (or return simulated value)."""
    global serial_conn, currentAngle_sim
    if serial_conn is None:
        await asyncio.sleep(0.05)           # tiny sim latency
        return f'ANGLE:{currentAngle_sim}'
    loop = asyncio.get_event_loop()
    try:
        def _read():
            serial_conn.timeout = timeout
            return serial_conn.readline().decode().strip()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _read),
            timeout=timeout + 0.5
        )
    except Exception:
        return None

def init_serial():
    """Scan common ports for Arduino, fall back to sim if not found."""
    global serial_conn
    if not SERIAL_AVAILABLE:
        push_serial_log('SIM > pyserial missing — simulated mode')
        return
    ports = []
    if sys.platform == 'darwin':
        ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/cu.usbserial*')
    elif sys.platform.startswith('linux'):
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    elif sys.platform == 'win32':
        ports = ['COM%d' % i for i in range(1, 20)]

    for port in sorted(ports):
        try:
            conn = pyserial.Serial(port, 9600, timeout=3)
            time.sleep(2)                   # wait for Arduino bootloader
            conn.write(b'PING\n')
            resp = conn.readline().decode().strip()
            if resp in ('PONG', 'READY'):
                serial_conn = conn
                push_serial_log(f'SER > Arduino on {port}')
                print(f'[SERIAL] Arduino connected on {port}')
                return
            conn.close()
        except Exception as e:
            print(f'[SERIAL] {port}: {e}')

    push_serial_log('SIM > No Arduino — simulated mode')
    print('[SERIAL] No Arduino found — simulated mode')

# ============================================================================
# GLOBAL STATE
# ============================================================================

clients   = set()
state_ref = {
    'state':      'idle',
    'object':     None,
    'grip':       65,
    'confidence': 0,
    'bbox':       None,
    'locked':     False,
}
frame_count        = 0
last_frame_time    = time.time()
feed_lost_reported = False

# ============================================================================
# FRAME LOOP (camera @ 20fps)
# ============================================================================

async def camera_loop(clients):
    global frame_count, cap, CAM_FOUND, last_frame_time, feed_lost_reported

    # Wait for webview window so camera permission dialog can appear
    await asyncio.sleep(2.5)
    init_camera()

    last_retry = time.time()

    while True:
        await asyncio.sleep(1 / 20)
        if not clients:
            continue

        frame = None
        frame_acquired = False

        if CAM_FOUND and cap is not None:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frame_acquired = True
                last_frame_time = time.time()
                feed_lost_reported = False
            else:
                if time.time() - last_frame_time > 2.0 and time.time() - last_retry > 3.0:
                    if cap:
                        cap.release()
                    init_camera()
                    last_retry = time.time()
                if not feed_lost_reported:
                    print("[CAM] Feed loss detected")
                    feed_lost_reported = True
        else:
            frame_acquired = True
            last_frame_time = time.time()
            feed_lost_reported = False

        if not frame_acquired:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cx, cy = 320, 240
            cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (30, 30, 30), 1)
            cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (30, 30, 30), 1)
            cv2.putText(frame, "FEED LOST - RECONNECTING", (160, 240),
                        cv2.FONT_HERSHEY_PLAIN, 1, (100, 50, 50), 1)
        else:
            if not (CAM_FOUND and cap is not None):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cx, cy = 320, 240
                cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (30, 30, 30), 1)
                cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (30, 30, 30), 1)
                cv2.putText(frame, "SIMULATED FEED", (220, 460),
                            cv2.FONT_HERSHEY_PLAIN, 1, (25, 25, 25), 1)

        frame_count += 1
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buf).decode()
        msg = json.dumps({
            "type":  "frame",
            "data":  b64,
            "count": frame_count,
            "real":  CAM_FOUND and frame_acquired
        })
        dead = set()
        for ws in clients:
            try:
                await ws.send(msg)
            except:
                dead.add(ws)
        clients -= dead

# ============================================================================
# DETECTION LOOP (every 3rd frame, idle state only)
# ============================================================================

async def detection_loop(clients, state_ref):
    global cap, CAM_FOUND, model
    detect_frame = 0
    lock_count   = 0
    locked_obj   = None

    while True:
        await asyncio.sleep(0.15)
        if not CAM_FOUND or cap is None or model is None:
            continue
        if state_ref['state'] != 'idle':
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        detect_frame += 1
        if detect_frame % 2 != 0:
            continue

        try:
            results = model(frame, verbose=False, conf=0.25, imgsz=320)
            if not results or not results[0].boxes:
                lock_count = 0
                locked_obj = None
                continue

            box    = results[0].boxes[0]
            cls_id = int(box.cls[0])
            name   = model.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            if name == locked_obj:
                lock_count += 1
            else:
                locked_obj = name
                lock_count = 1

            det_msg = json.dumps({
                "type":       "detection",
                "object":     name,
                "confidence": round(conf, 2),
                "bbox":       [x1, y1, x2, y2],
                "locked":     lock_count >= 3
            })
            for ws in list(clients):
                try:
                    await ws.send(det_msg)
                except:
                    pass

            if lock_count >= 2:
                state_ref['state']      = 'detected'
                state_ref['object']     = name
                state_ref['grip']       = GRIP_MAP.get(name, GRIP_MAP['default'])
                state_ref['confidence'] = conf
                state_ref['bbox']       = [x1, y1, x2, y2]
                state_ref['locked']     = True
                lock_count = 0

        except Exception as e:
            print(f"[DETECT] Error: {e}")

# ============================================================================
# STATE MACHINE
#
# IDLE → (1.5s) DETECTED → (1.2s) CALCULATING → (4s) INSERTING
#      → GRIPPING → SUCCESS/FAIL → (4s/3s) → IDLE
# ============================================================================

async def state_machine_loop(clients):
    prev_state = None

    while True:
        s = state_ref['state']

        # ── Broadcast + serial entry actions on state change ─────────────
        if s != prev_state:
            prev_state = s
            await broadcast(clients, json.dumps({
                "type":       "state",
                "state":      s,
                "object":     state_ref.get('object'),
                "grip":       state_ref.get('grip', 65),
                "confidence": state_ref.get('confidence', 0),
            }))
            # Send OPEN whenever entering idle / calculating / inserting
            if s in ('idle', 'calculating', 'inserting'):
                send_serial('OPEN')

        # ── Auto-advance ─────────────────────────────────────────────────

        if s == 'idle':
            await asyncio.sleep(0.2)

        elif s == 'detected':
            # Wait, then kick off calculation
            await asyncio.sleep(1.5)
            if state_ref['state'] == 'detected':
                state_ref['state'] = 'calculating'

        elif s == 'calculating':
            # OPEN already sent on entry; wait, then request insertion
            await asyncio.sleep(1.2)
            if state_ref['state'] == 'calculating':
                state_ref['state'] = 'inserting'

        elif s == 'inserting':
            # OPEN already sent on entry; wait 4s for human to present object
            await asyncio.sleep(4.0)
            if state_ref['state'] == 'inserting':
                state_ref['state'] = 'gripping'

        elif s == 'gripping':
            # Close to calculated angle
            grip = state_ref.get('grip', 65)
            send_serial(f'CLOSE:{grip}')
            await asyncio.sleep(1.5)
            # Request confirmation angle from Arduino
            send_serial('STATUS')
            response = await read_serial_response(timeout=2.0)
            if state_ref['state'] == 'gripping':
                # Accept real match or sim timeout — always go to success
                state_ref['state'] = 'success'

        elif s == 'success':
            await asyncio.sleep(4.0)
            if state_ref['state'] == 'success':
                send_serial('OPEN')
                state_ref['state']      = 'idle'
                state_ref['object']     = None
                state_ref['confidence'] = 0
                state_ref['bbox']       = None

        elif s == 'fail':
            send_serial('OPEN')
            await asyncio.sleep(3.0)
            if state_ref['state'] == 'fail':
                state_ref['state']      = 'idle'
                state_ref['object']     = None
                state_ref['confidence'] = 0
                state_ref['bbox']       = None

        else:
            await asyncio.sleep(0.2)

# ============================================================================
# SERIAL BROADCAST LOOP — drains serial_log_queue to WS clients
# ============================================================================

async def serial_broadcast_loop(clients):
    while True:
        await asyncio.sleep(0.1)
        while serial_log_queue:
            log = serial_log_queue.popleft()
            dead = set()
            for ws in clients:
                try:
                    await ws.send(json.dumps({'type': 'serial', 'log': log}))
                except:
                    dead.add(ws)
            clients -= dead

# ============================================================================
# WEBSOCKET HANDLER
# ============================================================================

async def handler(ws):
    clients.add(ws)
    # Send current state on connect
    await ws.send(json.dumps({
        "type":  "state",
        "state": state_ref['state'],
        "object": state_ref.get('object'),
        "grip":  state_ref.get('grip', 65),
        "cam":   CAM_FOUND,
    }))

    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
                if data['type'] == 'cmd':
                    cmd = data['cmd']
                    if cmd == 'open':
                        print("[CMD] OPEN")
                        send_serial('OPEN')
                        await broadcast(clients, json.dumps({
                            'type': 'serial_rx',
                            'msg':  'MANUAL > OPEN sent'
                        }))

                    elif cmd == 'close':
                        angle = state_ref.get('grip', 65)
                        print(f"[CMD] CLOSE:{angle}")
                        send_serial(f'CLOSE:{angle}')
                        await broadcast(clients, json.dumps({
                            'type': 'serial_rx',
                            'msg':  f'MANUAL > CLOSE:{angle} sent'
                        }))

                    elif cmd == 'grip':
                        angle = state_ref.get('grip', 65)
                        print(f"[CMD] GRIP @ {angle}°")
                        send_serial(f'CLOSE:{angle}')
                        state_ref['state'] = 'gripping'
                        await broadcast(clients, json.dumps({
                            'type':   'state',
                            'state':  'gripping',
                            'grip':   angle,
                            'object': state_ref.get('object')
                        }))

                    elif cmd == 'reset':
                        print("[CMD] RESET")
                        send_serial('OPEN')
                        state_ref['state']  = 'idle'
                        state_ref['object'] = None
                        state_ref['grip']   = 65
                        await broadcast(clients, json.dumps({
                            'type':   'state',
                            'state':  'idle',
                            'grip':   0,
                            'object': None
                        }))

                    elif cmd == 'estop':
                        print("[CMD] EMERGENCY STOP")
                        send_serial('OPEN')
                        state_ref['state']  = 'idle'
                        state_ref['object'] = None
                        await broadcast(clients, json.dumps({
                            'type':   'state',
                            'state':  'idle',
                            'grip':   0,
                            'object': None
                        }))
                        await broadcast(clients, json.dumps({
                            'type': 'serial_rx',
                            'msg':  '⚠ EMERGENCY STOP — gripper opened'
                        }))
            except json.JSONDecodeError:
                pass
    finally:
        clients.discard(ws)

# ============================================================================
# BROADCAST
# ============================================================================

async def broadcast(clients, msg):
    dead = set()
    for ws in clients:
        try:
            await ws.send(msg)
        except:
            dead.add(ws)
    clients -= dead

# ============================================================================
# MAIN ASYNC SERVER
# ============================================================================

async def main():
    print("[WS] Starting server on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.gather(
            camera_loop(clients),
            detection_loop(clients, state_ref),
            state_machine_loop(clients),
            serial_broadcast_loop(clients),
        )

def start_backend():
    asyncio.run(main())

# ============================================================================
# ENTRY POINT
# ============================================================================

def start_http_server(app_dir, port=8766):
    """Serve app.html over HTTP so WKWebView can reach ws://localhost."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(app_dir), **kw)
        def log_message(self, *_):
            pass
    server = http.server.HTTPServer(('localhost', port), Handler)
    server.serve_forever()

if __name__ == '__main__':
    app_dir = Path(__file__).parent

    # Start HTTP server for app.html
    http_thread = threading.Thread(target=start_http_server, args=(app_dir,), daemon=True)
    http_thread.start()

    # Start Arduino serial init (runs in background, falls back to sim)
    serial_thread = threading.Thread(target=init_serial, daemon=True)
    serial_thread.start()

    # Start WebSocket backend
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    time.sleep(1.5)

    if webview:
        webview.create_window(
            'Mechanical Gripper',
            'http://localhost:8766/app.html',
            width=1280, height=800,
            resizable=True,
            min_size=(900, 600)
        )
        webview.start()
    else:
        print("[APP] webview not available. Open http://localhost:8766/app.html in a browser.")
        print("[WS] Server running on ws://localhost:8765")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN]")
