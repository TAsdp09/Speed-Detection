import cv2
HEADLESS = False  # Jalankan tanpa GUI (cv2.imshow, namedWindow, dll)

from datetime import datetime
from ultralytics import YOLO
import numpy as np
import threading
import telebot
import time
import os
import logging
import firebase_admin
from firebase_admin import credentials, db
import uuid
import torch
import queue
import socket
import requests
import csv
import glob
import json
import math
import sys
import subprocess
import concurrent.futures

# ====================== INITIALIZATION SECTION ======================
# Logger Setup
logger = logging.getLogger("SpeedDetectionSystem")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler("speed_detection_activity.log", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Firebase Initialization
try:
    cred = credentials.Certificate("speed-detection-ta-firebase-adminsdk-fbsvc-7f6c195880.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://speed-detection-ta-default-rtdb.asia-southeast1.firebasedatabase.app/',
    })
    firebase_initialized = True
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    firebase_initialized = False

# Firebase References
if firebase_initialized:
    violations_ref = db.reference('violations')
    system_stats_ref = db.reference('system_stats')
    cameras_ref = db.reference('cameras')
    daily_stats_ref = db.reference('daily_stats')
    people_ref = db.reference('people_detections')
    reports_ref = db.reference('reports')
    calibration_ref = db.reference('calibration')
else:
    class DummyRef:
        def child(self, *args, **kwargs): return self
        def set(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def transaction(self, *args, **kwargs): pass
    violations_ref = system_stats_ref = cameras_ref = daily_stats_ref = people_ref = calibration_ref = DummyRef()

# Telegram Bot Setup
BOT_TOKEN = "7633151627:AAFowoEJTa9In8nYpHccAi9fSBP92Vw5lik"
CHAT_ID = "-4742577963"
bot = None
telegram_available = False

# Initialize Telegram Bot
def init_telegram_bot():
    global bot, telegram_available
    try:
        bot = telebot.TeleBot(BOT_TOKEN)
        telegram_available = True
        logger.info("Telegram bot initialized successfully")
    except Exception as e:
        logger.error(f"Telegram bot initialization failed: {e}")

# Constants
FPS = 15  # Default FPS for video processing
ACTUAL_DISTANCE_M = 14.0  # Jarak aktual antara garis start dan finish (meter)
SPEED_THRESHOLD = 30
CONFIDENCE_THRESHOLD = 0.5
NOTIFICATION_COOLDOWN = 10
ACTIVITY_LOG_INTERVAL = 1800  # 30 menit (dari 60 detik)
MIN_DISPLACEMENT_PX = 5
MIN_MOVEMENT_FRAMES = 2  # Lebih responsif
PERSON_COOLDOWN = 30
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds between retries
CALIBRATION_SAMPLE_SIZE = 10  # Minimum samples per speed category

# Network Status
network_available = False
last_network_check = 0
NETWORK_CHECK_INTERVAL = 30  # seconds

# Initialize YOLO Model
model = YOLO("yolov8n.pt").to('cpu')
torch.set_num_threads(4)

# Object Classes
COCO_CLASSES = {
    0: "Orang",
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
    1: "Sepeda"
}

# ====================== CALIBRATION MODULE ======================
CALIBRATION_DATA_DIR = r"C:\Users\thega\OneDrive\Documents\Speed Detection Program\Speed detection YOLO\dataset\Video"
CALIBRATION_FILE = "calibration.json"
CALIBRATION_FACTOR = 1.0  # Default factor

def calibrate_system():
    """Calibrate speed detection using sample frames from dataset"""
    global CALIBRATION_FACTOR
    calibration_factors = {}
    logger.info("🚀 Starting system calibration...")
    
    try:
        # Process each speed category
        for speed in [10, 20, 30, 40]:
            speed_dir = os.path.join(CALIBRATION_DATA_DIR, f"{speed}km")
            frame_files = glob.glob(os.path.join(speed_dir, "*.jpg"))
            
            if not frame_files or len(frame_files) < 2:
                logger.warning(f"⚠️ Insufficient calibration frames for {speed}km/h ({len(frame_files)} found)")
                continue
                
            measured_speeds = []
            logger.info(f"🔧 Processing {len(frame_files)} frames for {speed}km/h")
            
            # Process each calibration frame pair
            prev_positions = {}
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                if frame is None:
                    continue
                    
                # Process frame with YOLO
                results = model(frame)
                
                # Track detected vehicles
                positions = {}
                for r in results:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        class_id = int(cls)
                        if conf > CONFIDENCE_THRESHOLD and class_id in [2, 3, 5, 7]:  # Vehicles
                            x1, y1, x2, y2 = map(int, box)
                            # Gunakan titik tengah bawah (bagian bawah bbox) untuk kalkulasi perpindahan
                            bottom_center = ((x1 + x2) // 2, y2)
                            positions[len(positions)] = {
                                "bottom_center": bottom_center,
                                "timestamp": time.time()
                            }
                
                # Calculate distances between consecutive frames
                if i > 0:
                    for obj_id, data in positions.items():
                        if obj_id in prev_positions:
                            prev_center = prev_positions[obj_id]["bottom_center"]
                            curr_center = data["bottom_center"]
                            time_delta = 0.1  # Simulated time difference
                            
                            # Hanya perpindahan horizontal (X-axis)
                            distance_px = abs(curr_center[0] - prev_center[0])
                            
                            measured_speed = (distance_px * (ACTUAL_DISTANCE_M / 100) * 3.6) / time_delta
                            measured_speeds.append(measured_speed)
                
                prev_positions = positions.copy()
            
            # Calculate calibration factor
            if measured_speeds and len(measured_speeds) >= CALIBRATION_SAMPLE_SIZE:
                avg_speed = np.mean(measured_speeds)
                factor = speed / avg_speed if avg_speed > 0 else 1.0
                calibration_factors[speed] = factor
                logger.info(f"🎯 Calibration for {speed}km/h: factor={factor:.4f}")
            else:
                logger.warning(f"⚠️ Insufficient samples for {speed}km/h ({len(measured_speeds)} samples)")
        
        # Save calibration data
        if calibration_factors:
            # Use weighted average factor based on sample size
            total_samples = sum(len(calibration_factors[k]) for k in calibration_factors)
            avg_factor = sum(calibration_factors[k] * (len(calibration_factors[k])/total_samples) 
                          for k in calibration_factors)
            
            calibration_data = {
                "factor": avg_factor, 
                "timestamp": datetime.now().isoformat(),
                "samples": total_samples
            }
            
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(calibration_data, f)
                
            # Save to Firebase
            if firebase_initialized:
                try:
                    calibration_ref.set(calibration_data)
                except Exception as e:
                    logger.error(f"Failed to save calibration to Firebase: {str(e)}")
            
            logger.info(f"✅ Calibration complete! Global factor: {avg_factor:.4f}")
            CALIBRATION_FACTOR = avg_factor
            return avg_factor
        else:
            logger.warning("⚠️ No valid calibration data collected")
            return 1.0
    
    except Exception as e:
        logger.error(f"❌ Calibration failed: {str(e)}")
        return 1.0

def load_calibration():
    """Load calibration data from file"""
    try:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            logger.info(f"📥 Loaded calibration factor: {data['factor']:.4f}")
            return data["factor"]
        
        # Try loading from Firebase
        if firebase_initialized:
            try:
                data = calibration_ref.get()
                if data and 'factor' in data:
                    with open(CALIBRATION_FILE, "w") as f:
                        json.dump(data, f)
                    logger.info(f"☁️ Loaded calibration from Firebase: {data['factor']:.4f}")
                    return data['factor']
            except Exception as e:
                logger.error(f"Failed to load calibration from Firebase: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load calibration: {str(e)}")
    return 1.0

# Initialize calibration
CALIBRATION_FACTOR = load_calibration() or 1.0

# ====================== SYSTEM STATE ======================
system_stats = {
    "start_time": datetime.now(),
    "cameras": {},
    "total_vehicles_detected": 0,
    "total_violations": 0,
    "total_people_detected": 0,
    "last_activity_report": time.time(),
    "pending_notifications": 0,
    "pending_person_notifications": 0,
    "calibration_factor": CALIBRATION_FACTOR
}

# Tambahkan flag untuk restart
restart_requested = False

# Notification queues
violation_queue = queue.Queue()
person_queue = queue.Queue()

# Tracking states
last_notification = {}
last_person_notification = {}
system_running = True

# ====================== UTILITY FUNCTIONS ======================
def init_csv_log():
    """Initialize CSV log file"""
    CSV_LOG_FILE = "detection_log.csv"
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Camera", "ObjectType", "Speed(km/h)", 
                "ViolationStatus", "ViolationDescription", "Confidence", "BoundingBox"
            ])

def log_to_csv(camera_name, object_type, speed, violation_status, violation_description, confidence, bbox):
    """Log detection event to CSV file"""
    CSV_LOG_FILE = "detection_log.csv"
    timestamp = datetime.now().isoformat()
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else ""
    
    if not violation_description and violation_status:
        violation_description = "Exceeded speed limit" if object_type != "Orang" else "Orang terdeteksi saat malam hari"
    
    row = [
        timestamp,
        camera_name,
        object_type,
        round(speed, 2) if speed is not None else "",
        "True" if violation_status else "False",
        violation_description,
        round(confidence, 4),
        bbox_str
    ]
    
    try:
        with open(CSV_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to write to CSV: {str(e)}")

def check_network_connection():
    """Check network connectivity with multiple methods"""
    global network_available
    try:
        # Method 1: DNS Check
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        network_available = True
        return True
    except OSError:
        try:
            # Method 2: HTTP Request
            requests.get("http://www.google.com", timeout=5)
            network_available = True
            return True
        except:
            try:
                # Method 3: Socket Connection
                socket.create_connection(("1.1.1.1", 80), timeout=5)
                network_available = True
                return True
            except:
                network_available = False
                return False

# Perbaikan Fungsi Hitung Kecepatan dengan Stabilitas
MIN_TIME_DELTA = 0.05  # Minimum time delta (50 ms) to avoid division by very small numbers
MAX_REASONABLE_SPEED = 200  # Maximum reasonable speed (km/h) to filter out outliers

def calculate_speed(prev_center, curr_center, time_delta):
    """Calculate speed with calibration factor and stability fixes"""
    if time_delta < MIN_TIME_DELTA:
        return 0.0  # Skip calculation on unstable time delta
        
    # Hanya perpindahan horizontal (X-axis) yang dihitung
    distance_px = abs(curr_center[0] - prev_center[0])
    
    # Ignore small movements (likely detection noise)
    if distance_px < 3:  # Less than 3 pixels movement
        return 0.0

    # PERBAIKAN 1: Gunakan jarak aktual untuk konversi kecepatan
    # Hitung jarak aktual berdasarkan perpindahan pixel
    distance_m = distance_px * (ACTUAL_DISTANCE_M / 100)
    
    # PERBAIKAN 2: Konversi kecepatan dengan formula yang benar
    speed_mps = distance_m / time_delta
    speed_kmh = speed_mps * 3.6
    calibrated_speed = speed_kmh * CALIBRATION_FACTOR
    
    # PERBAIKAN 3: Filter kecepatan tidak realistis
    if calibrated_speed > MAX_REASONABLE_SPEED:
        return 0.0
        
    return calibrated_speed

def is_night_time():
    """Check if current time is during night hours"""
    current_hour = datetime.now().hour
    return current_hour >= 18 or current_hour < 6

def log_activity(message, send_to_telegram=False):
    """Log activity to file and optionally to Telegram"""
    logger.info(message)
    if send_to_telegram and telegram_available and CHAT_ID:
        try:
            bot.send_message(chat_id=CHAT_ID, text=message)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")

# ====================== FIREBASE OPERATIONS ======================
def save_violation_to_firebase(camera_name, vehicle_type, speed, image_path=None):
    """Save violation metadata to Firebase with Telegram-like format"""
    if not firebase_initialized:
        return None
        
    for attempt in range(MAX_RETRIES):
        try:
            violation_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Format data seperti pesan Telegram
            violation_data = {
                'id': violation_id,
                'message_type': 'PERINGATAN KECEPATAN BERLEBIH',
                'alert_emoji': '⚠️',
                'camera_info': {
                    'name': camera_name,
                    'emoji': '📹'
                },
                'vehicle_info': {
                    'type': vehicle_type,
                    'emoji': '🚗'
                },
                'speed_info': {
                    'detected_speed': round(speed, 2),
                    'speed_limit': SPEED_THRESHOLD,
                    'unit': 'km/jam',
                    'emoji': '⚡'
                },
                'timestamp_info': {
                    'detection_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'iso_time': current_time.isoformat(),
                    'unix_timestamp': int(time.time()),
                    'emoji': '⏰'
                },
                'violation_message': f"Kendaraan melebihi batas kecepatan {SPEED_THRESHOLD} km/jam!",
                'formatted_message': f"⚠️ PERINGATAN KECEPATAN BERLEBIH! ⚠️\n\n📹 Kamera: {camera_name}\n🚗 Tipe Kendaraan: {vehicle_type}\n⚡ Kecepatan: {round(speed, 2)} km/jam\n⏰ Waktu: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n\nKendaraan melebihi batas kecepatan {SPEED_THRESHOLD} km/jam!",
                'location': {
                    'camera_id': camera_name.lower().replace(' ', '_'),
                    'coordinates': None
                },
                'system_info': {
                    'calibration_factor': CALIBRATION_FACTOR,
                    'confidence_threshold': CONFIDENCE_THRESHOLD,
                    'actual_distance': ACTUAL_DISTANCE_M
                },
                'notification_status': {
                    'telegram_sent': True,
                    'firebase_saved': True,
                    'csv_logged': True
                },
                'status': 'VIOLATION_DETECTED',
                'severity': 'HIGH' if speed > SPEED_THRESHOLD * 1.5 else 'MEDIUM',
                'created_at': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Tambahkan info gambar jika ada
            if image_path and os.path.exists(image_path):
                violation_data['image_info'] = {
                    'path': image_path,
                    'filename': os.path.basename(image_path),
                    'exists': True
                }
            
            # Simpan ke Firebase
            violations_ref.child(violation_id).set(violation_data)
            
            # Update daily statistics dengan format yang lebih detail
            today = current_time.strftime('%Y-%m-%d')
            daily_stats_ref.child(today).transaction(lambda current_value: {
                'date': today,
                'total_violations': (current_value.get('total_violations', 0) if current_value else 0) + 1,
                'last_violation': current_time.isoformat(),
                'violations_by_camera': {
                    **((current_value.get('violations_by_camera', {}) if current_value else {})),
                    camera_name: (current_value.get('violations_by_camera', {}).get(camera_name, 0) if current_value else 0) + 1
                },
                'violations_by_vehicle': {
                    **((current_value.get('violations_by_vehicle', {}) if current_value else {})),
                    vehicle_type: (current_value.get('violations_by_vehicle', {}).get(vehicle_type, 0) if current_value else 0) + 1
                }
            })
            
            # Update camera-specific statistics dengan format detail
            camera_stats_ref = cameras_ref.child(camera_name.lower().replace(" ", "_"))
            camera_stats_ref.child('violation_stats').transaction(lambda current_value: {
                'total_violations': (current_value.get('total_violations', 0) if current_value else 0) + 1,
                'last_violation': {
                    'time': current_time.isoformat(),
                    'vehicle_type': vehicle_type,
                    'speed': round(speed, 2),
                    'violation_id': violation_id
                },
                'average_violation_speed': None,  # Bisa dihitung nanti
                'violations_by_vehicle': {
                    **((current_value.get('violations_by_vehicle', {}) if current_value else {})),
                    vehicle_type: (current_value.get('violations_by_vehicle', {}).get(vehicle_type, 0) if current_value else 0) + 1
                }
            })
            
            logger.info(f"🔥 Violation saved to Firebase: {violation_id}")
            return violation_id
            
        except Exception as e:
            logger.error(f"Error saving violation (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
            
    logger.error("Failed to save violation to Firebase after multiple attempts")
    return None

def save_person_detection_to_firebase(camera_name, image_path=None):
    """Save person detection metadata to Firebase with Telegram-like format"""
    if not firebase_initialized:
        return None
        
    for attempt in range(MAX_RETRIES):
        try:
            detection_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Format data seperti pesan Telegram
            detection_data = {
                'id': detection_id,
                'message_type': 'ORANG TERDETEKSI PADA MALAM HARI',
                'alert_emoji': '⚠️',
                'detection_emoji': '👤',
                'camera_info': {
                    'name': camera_name,
                    'emoji': '📹'
                },
                'timestamp_info': {
                    'detection_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'iso_time': current_time.isoformat(),
                    'unix_timestamp': int(time.time()),
                    'emoji': '⏰'
                },
                'detection_message': "Orang terdeteksi di area pada malam hari.",
                'formatted_message': f"⚠️ ORANG TERDETEKSI PADA MALAM HARI! ⚠️\n\n📹 Kamera: {camera_name}\n⏰ Waktu: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n\nOrang terdeteksi di area pada malam hari.",
                'location': {
                    'camera_id': camera_name.lower().replace(' ', '_'),
                    'coordinates': None
                },
                'time_info': {
                    'is_night_time': True,
                    'hour': current_time.hour,
                    'night_detection': True
                },
                'system_info': {
                    'confidence_threshold': CONFIDENCE_THRESHOLD,
                    'person_cooldown': PERSON_COOLDOWN
                },
                'notification_status': {
                    'telegram_sent': True,
                    'firebase_saved': True,
                    'csv_logged': True
                },
                'status': 'PERSON_DETECTED_NIGHT',
                'severity': 'MEDIUM',
                'created_at': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Tambahkan info gambar jika ada
            if image_path and os.path.exists(image_path):
                detection_data['image_info'] = {
                    'path': image_path,
                    'filename': os.path.basename(image_path),
                    'exists': True
                }
            
            # Simpan ke Firebase
            people_ref.child(detection_id).set(detection_data)
            
            # Update daily statistics untuk deteksi orang
            today = current_time.strftime('%Y-%m-%d')
            daily_stats_ref.child(today).transaction(lambda current_value: {
                'date': today,
                'total_people_detected': (current_value.get('total_people_detected', 0) if current_value else 0) + 1,
                'last_person_detection': current_time.isoformat(),
                'people_by_camera': {
                    **((current_value.get('people_by_camera', {}) if current_value else {})),
                    camera_name: (current_value.get('people_by_camera', {}).get(camera_name, 0) if current_value else 0) + 1
                }
            })
            
            logger.info(f"👤 Person detection saved to Firebase: {detection_id}")
            return detection_id
            
        except Exception as e:
            logger.error(f"Error saving person detection (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
            
    logger.error("Failed to save person detection to Firebase")
    return None

def update_system_stats_firebase():
    """Update system statistics in Firebase with detailed format"""
    if not firebase_initialized:
        return
        
    for attempt in range(MAX_RETRIES):
        try:
            current_time = datetime.now()
            uptime = current_time - system_stats["start_time"]
            uptime_str = str(uptime).split('.')[0]
            
            # Format seperti laporan Telegram
            firebase_stats = {
                'system_info': {
                    'title': 'SISTEM PEMANTAUAN KECEPATAN - LAPORAN AKTIVITAS',
                    'title_emoji': '📊',
                    'status': 'ACTIVE',
                    'last_update': current_time.isoformat(),
                    'last_update_formatted': current_time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'runtime_info': {
                    'start_time': system_stats["start_time"].isoformat(),
                    'uptime_seconds': int(uptime.total_seconds()),
                    'uptime_formatted': uptime_str,
                    'emoji': '⏱️'
                },
                'detection_stats': {
                    'total_vehicles': {
                        'count': system_stats["total_vehicles_detected"],
                        'emoji': '🚗'
                    },
                    'total_violations': {
                        'count': system_stats["total_violations"],
                        'emoji': '🚨'
                    },
                    'total_people_night': {
                        'count': system_stats["total_people_detected"],
                        'emoji': '👤'
                    }
                },
                'queue_info': {
                    'pending_notifications': {
                        'count': system_stats["pending_notifications"],
                        'emoji': '⏳'
                    },
                    'pending_person_notifications': {
                        'count': system_stats["pending_person_notifications"],
                        'emoji': '⏳'
                    }
                },
                'system_config': {
                    'calibration_factor': {
                        'value': CALIBRATION_FACTOR,
                        'emoji': '🎯'
                    },
                    'speed_threshold': {
                        'value': SPEED_THRESHOLD,
                        'unit': 'km/jam',
                        'emoji': '⚙️'
                    },
                    'network_status': {
                        'available': network_available,
                        'emoji': '🌐',
                        'status_text': '✅ Available' if network_available else '❌ Unavailable'
                    },
                    'actual_distance': {
                        'value': ACTUAL_DISTANCE_M,
                        'unit': 'meter',
                        'emoji': '📏'
                    }
                },
                'formatted_report': f"📊 SISTEM PEMANTAUAN KECEPATAN - LAPORAN AKTIVITAS 📊\n\n⏱️ Waktu Operasi: {uptime_str}\n🚗 Total Kendaraan Terdeteksi: {system_stats['total_vehicles_detected']}\n🚨 Total Pelanggaran: {system_stats['total_violations']}\n👤 Total Orang Terdeteksi (Malam): {system_stats['total_people_detected']}\n⏳ Pending Notifications: {system_stats['pending_notifications']}\n⏳ Pending Person Notifications: {system_stats['pending_person_notifications']}\n🎯 Calibration Factor: {CALIBRATION_FACTOR:.4f}\n📏 Jarak Aktual: {ACTUAL_DISTANCE_M} meter\n\n⚙️ Batas Kecepatan: {SPEED_THRESHOLD} km/jam\n🌐 Network Status: {'✅ Available' if network_available else '❌ Unavailable'}"
            }
            
            # Tambahkan info kamera dengan format detail
            camera_status_section = {}
            for cam_name, stats in system_stats["cameras"].items():
                status_text = "✅ Aktif" if stats["active"] else "❌ Tidak Aktif"
                camera_status_section[cam_name.lower().replace(' ', '_')] = {
                    'name': cam_name,
                    'status': {
                        'active': stats["active"],
                        'text': status_text,
                        'emoji': '✅' if stats["active"] else '❌'
                    },
                    'statistics': {
                        'vehicles_detected': {
                            'count': stats["vehicles_detected"],
                            'emoji': '🚗'
                        },
                        'violations': {
                            'count': stats["violations"],
                            'emoji': '🚨'
                        },
                        'people_detected_night': {
                            'count': stats.get("people_detected", 0),
                            'emoji': '👤'
                        }
                    },
                    'last_frame_time': stats["last_frame_time"],
                    'last_update': current_time.isoformat(),
                    'formatted_status': f"- {cam_name}: {status_text}\n  Kendaraan: {stats['vehicles_detected']}\n  Pelanggaran: {stats['violations']}\n  Orang (Malam): {stats.get('people_detected', 0)}\n"
                }
            
            firebase_stats['camera_status'] = camera_status_section
            
            # Simpan ke Firebase
            system_stats_ref.set(firebase_stats)
            
            logger.info("Updated system stats on Firebase with Telegram format")
            return
            
        except Exception as e:
            logger.error(f"Error updating Firebase stats (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
            
    logger.error("Failed to update Firebase stats")

def save_activity_report_to_firebase():
    """Save detailed activity report to Firebase (sama seperti yang dikirim ke Telegram)"""
    if not firebase_initialized:
        return
        
    try:
        current_time = datetime.now()
        uptime = current_time - system_stats["start_time"]
        uptime_str = str(uptime).split('.')[0]
        
        # Format laporan persis seperti yang dikirim ke Telegram
        report_message = (
            f"📊 SISTEM PEMANTAUAN KECEPATAN - LAPORAN AKTIVITAS 📊\n\n"
            f"⏱️ Waktu Operasi: {uptime_str}\n"
            f"🚗 Total Kendaraan Terdeteksi: {system_stats['total_vehicles_detected']}\n"
            f"🚨 Total Pelanggaran: {system_stats['total_violations']}\n"
            f"👤 Total Orang Terdeteksi (Malam): {system_stats['total_people_detected']}\n"
            f"⏳ Pending Notifications: {system_stats['pending_notifications']}\n"
            f"⏳ Pending Person Notifications: {system_stats['pending_person_notifications']}\n"
            f"🎯 Calibration Factor: {CALIBRATION_FACTOR:.4f}\n"
            f"📏 Jarak Aktual: {ACTUAL_DISTANCE_M} meter\n\n"
            f"📷 STATUS KAMERA:\n"
        )
        
        # Tambahkan status kamera
        camera_status_text = ""
        for cam_name, stats in system_stats["cameras"].items():
            status = "✅ Aktif" if stats["active"] else "❌ Tidak Aktif"
            camera_status_text += (
                f"- {cam_name}: {status}\n"
                f"  Kendaraan: {stats['vehicles_detected']}\n"
                f"  Pelanggaran: {stats['violations']}\n"
                f"  Orang (Malam): {stats.get('people_detected', 0)}\n"
            )
        
        report_message += camera_status_text
        report_message += (
            f"\n⚙️ Batas Kecepatan: {SPEED_THRESHOLD} km/jam\n"
            f"🌐 Network Status: {'✅ Available' if network_available else '❌ Unavailable'}"
        )
        
        # Simpan laporan ke Firebase
        report_data = {
            'report_id': str(uuid.uuid4()),
            'timestamp': current_time.isoformat(),
            'report_type': 'ACTIVITY_REPORT',
            'full_message': report_message,
            'report_data': {
                'uptime': uptime_str,
                'total_vehicles': system_stats['total_vehicles_detected'],
                'total_violations': system_stats['total_violations'],
                'total_people_night': system_stats['total_people_detected'],
                'pending_notifications': system_stats['pending_notifications'],
                'pending_person_notifications': system_stats['pending_person_notifications'],
                'calibration_factor': CALIBRATION_FACTOR,
                'speed_threshold': SPEED_THRESHOLD,
                'actual_distance': ACTUAL_DISTANCE_M,
                'network_available': network_available,
                'camera_status': {cam_name: stats for cam_name, stats in system_stats["cameras"].items()}
            }
        }
        
        # Simpan ke node reports
        reports_ref = db.reference('reports')
        reports_ref.child(current_time.strftime('%Y-%m-%d')).child(str(int(time.time()))).set(report_data)
        
        logger.info("Activity report saved to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to save activity report to Firebase: {str(e)}")


# ====================== NOTIFICATION HANDLING ======================
def send_telegram_notification(camera_name, vehicle_type, speed, frame):
    """Send violation notification with queuing"""
    violation_dir = "violation_images"
    os.makedirs(violation_dir, exist_ok=True)
    
    image_path = f"{violation_dir}/violation_{int(time.time())}.jpg"
    cv2.imwrite(image_path, frame)
    
    notification_data = {
        "camera_name": camera_name,
        "vehicle_type": vehicle_type,
        "speed": speed,
        "image_path": image_path,
        "timestamp": time.time(),
        "attempts": 0
    }
    
    violation_queue.put(notification_data)
    system_stats["pending_notifications"] += 1
    logger.info(f"📥 Violation queued: {vehicle_type} at {speed:.2f} km/h")

def send_person_notification(camera_name, frame):
    """Send person detection notification with queuing"""
    person_dir = "person_images"
    os.makedirs(person_dir, exist_ok=True)
    
    image_path = f"{person_dir}/person_{int(time.time())}.jpg"
    cv2.imwrite(image_path, frame)
    
    notification_data = {
        "camera_name": camera_name,
        "image_path": image_path,
        "timestamp": time.time(),
        "attempts": 0
    }
    
    person_queue.put(notification_data)
    system_stats["pending_person_notifications"] += 1
    logger.info(f"📥 Person detection queued: {camera_name}")

def process_notification_queues():
    """Process queued notifications"""
    global network_available, last_network_check
    
    while system_running:
        try:
            # Network status check
            current_time = time.time()
            if current_time - last_network_check > NETWORK_CHECK_INTERVAL:
                network_available = check_network_connection()
                last_network_check = current_time
                logger.info(f"🌐 Network status: {'Connected' if network_available else 'Disconnected'}")
            
            # Process violation queue
            if not violation_queue.empty() and network_available:
                notification_data = violation_queue.get()
                
                message = (
                    f"⚠️ PERINGATAN KECEPATAN BERLEBIH! ⚠️\n\n"
                    f"📹 Kamera: {notification_data['camera_name']}\n"
                    f"🚗 Tipe Kendaraan: {notification_data['vehicle_type']}\n"
                    f"⚡ Kecepatan: {notification_data['speed']:.2f} km/jam\n"
                    f"⏰ Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"\nKendaraan melebihi batas kecepatan {SPEED_THRESHOLD} km/jam!"
                )
                
                try:
                    with open(notification_data["image_path"], 'rb') as photo:
                        msg = bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
                    
                    # Save to Firebase
                    violation_id = save_violation_to_firebase(
                        notification_data["camera_name"],
                        notification_data["vehicle_type"],
                        notification_data["speed"],
                        notification_data["image_path"]
                    )
                    
                    # Update message with ID
                    if violation_id:
                        try:
                            update_message = message + f"\n🆔 ID Pelanggaran: {violation_id}"
                            bot.edit_message_caption(
                                chat_id=CHAT_ID,
                                message_id=msg.message_id,
                                caption=update_message
                            )
                        except:
                            pass
                    
                    logger.info(f"📤 Violation notification sent")
                    system_stats["pending_notifications"] -= 1
                    system_stats["total_violations"] += 1
                except Exception as e:
                    # Retry later
                    notification_data["attempts"] += 1
                    if notification_data["attempts"] < MAX_RETRIES:
                        violation_queue.put(notification_data)
                        logger.warning(f"Failed to send violation notification (retry {notification_data['attempts']}): {str(e)}")
                    else:
                        logger.error(f"Permanently failed to send violation notification: {str(e)}")
                        system_stats["pending_notifications"] -= 1
                        try:
                            os.remove(notification_data["image_path"])
                        except:
                            pass
            
            # Process person queue
            if not person_queue.empty() and network_available:
                notification_data = person_queue.get()
                
                message = (
                    f"⚠️ ORANG TERDETEKSI PADA MALAM HARI! ⚠️\n\n"
                    f"📹 Kamera: {notification_data['camera_name']}\n"
                    f"⏰ Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"\nOrang terdeteksi di area pada malam hari."
                )
                
                try:
                    with open(notification_data["image_path"], 'rb') as photo:
                        msg = bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
                    
                    # Save to Firebase
                    detection_id = save_person_detection_to_firebase(
                        notification_data["camera_name"],
                        notification_data["image_path"]
                    )
                    
                    # Update message with ID
                    if detection_id:
                        try:
                            update_message = message + f"\n🆔 ID Deteksi: {detection_id}"
                            bot.edit_message_caption(
                                chat_id=CHAT_ID,
                                message_id=msg.message_id,
                                caption=update_message
                            )
                        except:
                            pass
                    
                    logger.info(f"📤 Person notification sent")
                    system_stats["pending_person_notifications"] -= 1
                    system_stats["total_people_detected"] += 1
                except Exception as e:
                    # Retry later
                    notification_data["attempts"] += 1
                    if notification_data["attempts"] < MAX_RETRIES:
                        person_queue.put(notification_data)
                        logger.warning(f"Failed to send person notification (retry {notification_data['attempts']}): {str(e)}")
                    else:
                        logger.error(f"Permanently failed to send person notification: {str(e)}")
                        system_stats["pending_person_notifications"] -= 1
                        try:
                            os.remove(notification_data["image_path"])
                        except:
                            pass
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Notification queue error: {str(e)}")
            time.sleep(5)

# ====================== REPORTING & TELEGRAM COMMANDS ======================
def send_activity_report():
    """Send comprehensive activity report to Telegram and save to Firebase"""
    if not telegram_available or not CHAT_ID:
        return
    
    try:
        current_time = datetime.now()
        uptime = current_time - system_stats["start_time"]
        uptime_str = str(uptime).split('.')[0]
        
        message = (
            f"📊 SISTEM PEMANTAUAN KECEPATAN - LAPORAN AKTIVITAS 📊\n\n"
            f"⏱️ Waktu Operasi: {uptime_str}\n"
            f"🚗 Total Kendaraan Terdeteksi: {system_stats['total_vehicles_detected']}\n"
            f"🚨 Total Pelanggaran: {system_stats['total_violations']}\n"
            f"👤 Total Orang Terdeteksi (Malam): {system_stats['total_people_detected']}\n"
            f"⏳ Pending Notifications: {system_stats['pending_notifications']}\n"
            f"⏳ Pending Person Notifications: {system_stats['pending_person_notifications']}\n"
            f"🎯 Calibration Factor: {CALIBRATION_FACTOR:.4f}\n"
            f"📏 Jarak Aktual: {ACTUAL_DISTANCE_M} meter\n\n"
            f"📷 STATUS KAMERA:\n"
        )
        
        # Add camera-specific stats
        for cam_name, stats in system_stats["cameras"].items():
            status = "✅ Aktif" if stats["active"] else "❌ Tidak Aktif"
            message += (
                f"- {cam_name}: {status}\n"
                f"  Kendaraan: {stats['vehicles_detected']}\n"
                f"  Pelanggaran: {stats['violations']}\n"
                f"  Orang (Malam): {stats.get('people_detected', 0)}\n"
            )
        
        message += (
            f"\n⚙️ Batas Kecepatan: {SPEED_THRESHOLD} km/jam\n"
            f"🌐 Network Status: {'✅ Available' if network_available else '❌ Unavailable'}"
        )
        
        # Kirim ke Telegram
        bot.send_message(chat_id=CHAT_ID, text=message)
        
        # Simpan ke Firebase dengan format yang sama
        save_activity_report_to_firebase()
        
        system_stats["last_activity_report"] = time.time()
        logger.info("Activity report sent to Telegram and saved to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to send activity report: {str(e)}")

# Handler Telegram hanya didaftarkan sekali
handlers_registered = False

def register_handlers():
    """Daftarkan handler Telegram sekali saja"""
    global handlers_registered
    if handlers_registered:
        return
        
    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        try:
            help_text = (
                "🚦 SISTEM PEMANTAUAN KECEPATAN KENDARAAN 🚦\n\n"
                "Perintah yang tersedia:\n"
                "/status - Status sistem saat ini\n"
                "/report - Laporan aktivitas terbaru\n"
                "/threshold [nilai] - Ubah batas kecepatan\n"
                "/log - Kirim file log aktivitas\n"
                "/csvlog - Kirim log deteksi (CSV)\n"
                "/calibrate - Kalibrasi sistem\n"
                "/getcalibration - Faktor kalibrasi saat ini\n"
                "/restart - Restart sistem\n"
                "/shutdown - Matikan sistem\n"
                "/help - Tampilkan bantuan ini"
            )
            bot.reply_to(message, help_text)
        except Exception as e:
            logger.error(f"Failed to process /help: {str(e)}")

    @bot.message_handler(commands=['status'])
    def send_status(message):
        try:
            send_activity_report()
        except Exception as e:
            logger.error(f"Failed to process /status: {str(e)}")

    @bot.message_handler(commands=['threshold'])
    def change_threshold(message):
        try:
            global SPEED_THRESHOLD
            parts = message.text.split()
            if len(parts) < 2:
                bot.reply_to(message, "Format: /threshold [nilai]")
                return
            
            new_threshold = float(parts[1])
            if new_threshold > 0:
                old_threshold = SPEED_THRESHOLD
                SPEED_THRESHOLD = new_threshold
                response = f"Batas kecepatan diubah: {old_threshold} → {SPEED_THRESHOLD} km/jam"
                bot.reply_to(message, response)
                logger.info(response)
            else:
                bot.reply_to(message, "Batas kecepatan harus > 0")
        except Exception as e:
            logger.error(f"Failed to process /threshold: {str(e)}")

    @bot.message_handler(commands=['report'])
    def request_report(message):
        try:
            send_activity_report()
            bot.reply_to(message, "Laporan aktivitas telah dikirim")
        except Exception as e:
            logger.error(f"Failed to process /report: {str(e)}")

    @bot.message_handler(commands=['log'])
    def send_log_file(message):
        try:
            with open("speed_detection_activity.log", 'rb') as log_file:
                bot.send_document(message.chat.id, log_file, caption="Log Aktivitas Sistem")
        except Exception as e:
            logger.error(f"Failed to send log file: {str(e)}")
    
    @bot.message_handler(commands=['csvlog'])
    def send_csv_log(message):
        try:
            with open("detection_log.csv", 'rb') as csv_file:
                bot.send_document(message.chat.id, csv_file, caption="Log Deteksi (CSV)")
        except Exception as e:
            logger.error(f"Failed to send CSV log: {str(e)}")

    @bot.message_handler(commands=['calibrate'])
    def calibrate_command(message):
        try:
            bot.reply_to(message, "🚀 Memulai kalibrasi sistem...")
            global CALIBRATION_FACTOR
            CALIBRATION_FACTOR = calibrate_system()
            response = f"✅ Kalibrasi berhasil! Faktor: {CALIBRATION_FACTOR:.4f}"
            bot.reply_to(message, response)
            logger.info(response)
        except Exception as e:
            bot.reply_to(message, f"❌ Gagal kalibrasi: {str(e)}")
            logger.error(f"Calibration failed: {str(e)}")
    
    @bot.message_handler(commands=['getcalibration'])
    def get_calibration(message):
        try:
            response = f"🎯 Faktor kalibrasi: {CALIBRATION_FACTOR:.4f}"
            bot.reply_to(message, response)
        except Exception as e:
            logger.error(f"Failed to get calibration: {str(e)}")

    @bot.message_handler(commands=['restart'])
    def restart_system(message):
        global system_running, restart_requested
        try:
            bot.reply_to(message, "⚠️ Memulai ulang sistem...")
            logger.info("Restart command received")
            restart_requested = True
            system_running = False
        except Exception as e:
            logger.error(f"Failed to process /restart: {str(e)}")

    @bot.message_handler(commands=['shutdown'])
    def shutdown_system(message):
        global system_running
        try:
            bot.reply_to(message, "⚠️ Mematikan sistem...")
            logger.info("Shutdown command received")
            system_running = False
        except Exception as e:
            logger.error(f"Failed to process /shutdown: {str(e)}")
    
    handlers_registered = True

def handle_telegram_commands():
    """Process Telegram commands with enhanced resilience"""
    global telegram_available, system_running
    
    # Daftarkan handler sekali saja
    if bot and not handlers_registered:
        register_handlers()
    
    last_success = time.time()
    consecutive_errors = 0
    
    while system_running:
        try:
            if not telegram_available:
                init_telegram_bot()
                time.sleep(5)
                continue
            
            logger.info("Starting Telegram polling...")
            bot.infinity_polling(timeout=30, long_polling_timeout=45)  # Enhanced timeout settings
            
            # Reset error counter on success
            consecutive_errors = 0
            last_success = time.time()
            
        except Exception as e:
            logger.error(f"Telegram error: {str(e)}")
            consecutive_errors += 1
            
            # Exponential backoff
            sleep_time = min(300, 2 ** consecutive_errors)  # Max 5 minutes
            logger.warning(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            
            # Attempt full reinitialization after 3 errors
            if consecutive_errors >= 3:
                telegram_available = False
                try:
                    bot.stop_polling()
                except:
                    pass
                time.sleep(5)

# ====================== CAMERA PROCESSING ======================
def monitor_system_health():
    """Monitor camera health and system status"""
    # ... [Kode sebelumnya tetap sama] ...

def process_camera(camera_index, camera_name):
    """Main camera processing function with brightness/contrast/exposure settings"""
    global system_running
    
    logger.info(f"Starting camera {camera_index} ({camera_name})")
    if not HEADLESS:
        cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)
    if not HEADLESS:
        cv2.resizeWindow(camera_name, 800, 600)
    
    # Initialize camera stats
    if camera_name not in system_stats["cameras"]:
        system_stats["cameras"][camera_name] = {
            "active": False,
            "vehicles_detected": 0,
            "violations": 0,
            "people_detected": 0,
            "last_frame_time": None
        }
    
    # Initialize camera with 1280x720 resolution
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        system_stats["cameras"][camera_name]["active"] = False
        if not HEADLESS:
            cv2.destroyWindow(camera_name)
        return

    # ===== KAMERA SETTINGS =====
    # Atur properti kamera untuk mengurangi kecerahan berlebihan
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Kurangi kecerahan (range: 0-1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.4)  # 50% kecerahan
    
    # Kurangi kontras (range: 0-1)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)    # 50% kontras
    
    # Atur eksposur (nilai negatif = mode manual)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)      # Kurangi eksposur
    
    # Atur saturasi warna (range: 0-1)
    cap.set(cv2.CAP_PROP_SATURATION, 1)   # 60% saturasi
    
    # Atur gamma (range: 0-1)
    cap.set(cv2.CAP_PROP_GAMMA, 0.7)        # 70% gamma
    
    # Set gain (range: 0-100)
    cap.set(cv2.CAP_PROP_GAIN, 30)          # 30% gain
    
    # ===== VERIFIKASI SETTINGS =====
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    
    logger.info(f"Camera {camera_name} settings:")
    logger.info(f"Resolution: {actual_width}x{actual_height}")
    logger.info(f"Brightness: {brightness:.2f}")
    logger.info(f"Contrast: {contrast:.2f}")
    logger.info(f"Exposure: {exposure:.2f}")
    
    system_stats["cameras"][camera_name]["active"] = True
    prev_positions = {}
    prev_timestamps = {}
    movement_counters = {}
    last_frame_count = 0
    frame_count = 0
    fps_start_time = time.time()
    
    # Garis deteksi untuk perhitungan kecepatan
    DETECTION_LINES = {}
    LINE_COOLDOWN = {}  # Untuk mencegah deteksi ganda
    
    # PERBAIKAN 1: Inisialisasi variabel untuk koreksi kecepatan
    PIXEL_PER_METER = 100 / ACTUAL_DISTANCE_M  # Pixel per meter
    
    while cap.isOpened() and system_running:
        current_time = time.time()
        # Grab frame with timeout
        ret = cap.grab()
        if not ret:
            logger.warning(f"Grab failed for camera {camera_name}")
            time.sleep(0.1)
            continue
            
        ret, frame = cap.retrieve()
        if not ret:
            logger.warning(f"Retrieve failed for camera {camera_name}")
            time.sleep(0.1)
            continue

        # ===== AUTO BRIGHTNESS CORRECTION =====
        # Otomatis sesuaikan kecerahan jika masih terlalu terang
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        brightness_level = np.mean(v)
        
        if brightness_level > 160:  # Jika terlalu terang
            # Kurangi nilai V (value/brightness) di HSV
            scale = 0.8
            v = np.clip(v * scale, 0, 255).astype(np.uint8)
            hsv[:,:,2] = v
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            logger.debug(f"Applied auto brightness correction: {brightness_level:.1f} -> {np.mean(v):.1f}")

        frame_count += 1
        system_stats["cameras"][camera_name]["last_frame_time"] = current_time
        display_frame = frame.copy()
        
        # Tentukan posisi garis deteksi (40% dan 60% tinggi frame)
        frame_height, frame_width = frame.shape[:2]
        start_line_y = int(frame_height * 0.4)
        finish_line_y = int(frame_height * 0.6)
        
        # PERBAIKAN 2: Lebarkan jarak garis deteksi
        # start_line_y = int(frame_height * 0.3)
        # finish_line_y = int(frame_height * 0.7)
        
        # Gambar garis deteksi
        cv2.line(display_frame, (0, start_line_y), (frame_width, start_line_y), (0, 255, 255), 2)
        cv2.putText(display_frame, "Start", (10, start_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.line(display_frame, (0, finish_line_y), (frame_width, finish_line_y), (0, 255, 255), 2)
        cv2.putText(display_frame, "Finish", (10, finish_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2) 

        # Perform object detection
        inference_start = time.time()
        with torch.no_grad():
            results = model(frame)
        inference_time = time.time() - inference_start
        
        new_positions = {}
        vehicles_in_frame = 0
        person_detected = False

        # Process detections
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_id = int(cls)
                confidence = float(conf)
                
                if confidence < CONFIDENCE_THRESHOLD or class_id not in COCO_CLASSES:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                # Gunakan titik tengah bawah (bagian bawah bbox) untuk tracking
                bottom_center = ((x1 + x2) // 2, y2)
                label = COCO_CLASSES[class_id]
                
                # Store object data
                data = {
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "center": bottom_center,  # Gunakan titik bawah
                    "confidence": confidence
                }
                new_positions[len(new_positions)] = data
                
                # Draw bounding box
                color = (0, 255, 0)  # Default green
                if label == "Orang":
                    color = (0, 255, 255)  # Yellow
                    night = is_night_time()
                    log_to_csv(
                        camera_name=camera_name,
                        object_type=label,
                        speed=None,
                        violation_status=night,
                        violation_description="Orang terdeteksi saat malam hari" if night else "",
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    )
                    if night:
                        person_detected = True
                        system_stats["cameras"][camera_name]["people_detected"] += 1
                elif label in ["Mobil", "Motor", "Bus", "Truk"]:
                    color = (0, 0, 255)  # Red
                    vehicles_in_frame += 1
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Handle person detection at night
        if person_detected and is_night_time():
            camera_id = f"{camera_name}"
            if (camera_id not in last_person_notification or 
                current_time - last_person_notification.get(camera_id, 0) > PERSON_COOLDOWN):
                
                notification_frame = frame.copy()
                for r in results:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        if int(cls) == 0 and float(conf) > CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(notification_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            cv2.putText(notification_frame, "ORANG", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                threading.Thread(
                    target=send_person_notification,
                    args=(camera_name, notification_frame)
                ).start()
                last_person_notification[camera_id] = current_time

        # Update vehicle counts
        if vehicles_in_frame != last_frame_count:
            diff = vehicles_in_frame - last_frame_count
            system_stats["total_vehicles_detected"] += diff
            system_stats["cameras"][camera_name]["vehicles_detected"] += diff
            last_frame_count = vehicles_in_frame

        # Track objects and calculate speeds
        for object_id, data in new_positions.items():
            if object_id in prev_positions:
                prev_data = prev_positions[object_id]
                label = data["label"]
                
                if label not in ["Orang"]:  # Only track vehicles
                    prev_center = prev_data["center"]
                    curr_center = data["center"]
                    prev_time = prev_timestamps.get(object_id, current_time)
                    time_delta = current_time - prev_time
                    
                    # PERBAIKAN 3: Filter time_delta yang terlalu kecil
                    if time_delta < 0.01:  # Skip jika time_delta terlalu kecil
                        continue
                    
                    # Calculate displacement
                    displacement = abs(curr_center[0] - prev_center[0])  # Hanya horizontal
                    
                    # Initialize movement counter
                    if object_id not in movement_counters:
                        movement_counters[object_id] = 0
                    
                    # Update movement counter
                    if displacement > MIN_DISPLACEMENT_PX:
                        movement_counters[object_id] += 1
                    else:
                        movement_counters[object_id] = max(0, movement_counters[object_id] - 1)
                    
                    # Calculate speed if moving consistently
                    if movement_counters[object_id] >= MIN_MOVEMENT_FRAMES:
                        speed = calculate_speed(prev_center, curr_center, time_delta)
                    else:
                        speed = 0.0
                    
                    # Display speed
                    speed_text = f"{speed:.1f} km/h"
                    speed_color = (0, 255, 0)  # Green
                    if speed > SPEED_THRESHOLD:
                        speed_color = (0, 0, 255)  # Red
                    x1, y1, x2, y2 = data["bbox"]
                    cv2.putText(display_frame, speed_text, 
                              (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
                    
                    # Gambar garis vektor perpindahan
                    cv2.line(display_frame, prev_center, curr_center, (0, 255, 255), 2)
                    
                    # Handle garis deteksi
                    # Periksa apakah objek melewati garis start
                    if prev_center[1] < start_line_y and curr_center[1] >= start_line_y:
                        DETECTION_LINES[object_id] = {"start_time": current_time}
                        logger.info(f"🚦 Vehicle crossed start line: {label}")
                    
                    # Periksa apakah objek melewati garis finish
                    if prev_center[1] < finish_line_y and curr_center[1] >= finish_line_y:
                        if object_id in DETECTION_LINES and "start_time" in DETECTION_LINES[object_id]:
                            # Hitung waktu tempuh
                            time_diff = current_time - DETECTION_LINES[object_id]["start_time"]
                            if time_diff > 0:
                                # Gunakan jarak aktual 14 meter
                                speed_line = (ACTUAL_DISTANCE_M / time_diff) * 3.6 * CALIBRATION_FACTOR
                                
                                # PERBAIKAN 4: Filter kecepatan tidak realistis
                                if speed_line < MAX_REASONABLE_SPEED:
                                    # Tampilkan kecepatan yang lebih akurat
                                    cv2.putText(display_frame, f"{speed_line:.1f} km/h (Line)", 
                                              (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                    
                                    # Log to CSV untuk setiap kendaraan yang melewati garis finish
                                    violation_line = speed_line > SPEED_THRESHOLD
                                    log_to_csv(
                                        camera_name=camera_name,
                                        object_type=label,
                                        speed=speed_line,
                                        violation_status=violation_line,
                                        violation_description="",
                                        confidence=data["confidence"],
                                        bbox=data["bbox"]
                                    )
                                    
                                    # Log kecepatan yang terdeteksi
                                    logger.info(f"📏 Vehicle passed finish line: {label} at {speed_line:.1f} km/h")
                                    
                                    # Handle speeding violation
                                    if violation_line:
                                        vehicle_id = f"{camera_name}_{object_id}"
                                        # Cooldown untuk notifikasi
                                        if (vehicle_id not in last_notification or 
                                            current_time - last_notification.get(vehicle_id, 0) > NOTIFICATION_COOLDOWN):
                                            
                                            notification_frame = frame.copy()
                                            cv2.rectangle(notification_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                            cv2.putText(notification_frame, f"SPEEDING: {speed_line:.2f} km/h", 
                                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                            
                                            # Kirim notifikasi dalam thread terpisah
                                            threading.Thread(
                                                target=send_telegram_notification,
                                                args=(camera_name, label, speed_line, notification_frame)
                                            ).start()
                                            
                                            last_notification[vehicle_id] = current_time
                                            system_stats["cameras"][camera_name]["violations"] += 1
                            
                            # Hapus dari tracking
                            del DETECTION_LINES[object_id]
            
            prev_timestamps[object_id] = current_time

        prev_positions = new_positions
        
        # Hapus objek yang tidak terdeteksi lagi dari DETECTION_LINES
        active_ids = set(new_positions.keys())
        for obj_id in list(DETECTION_LINES.keys()):
            if obj_id not in active_ids:
                del DETECTION_LINES[obj_id]
        
        # Display frame
        if not HEADLESS:
            cv2.imshow(camera_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            break
        elif key == ord('s'):
            send_activity_report()
        
        # Log performance
        if frame_count >= 50:
            elapsed = time.time() - fps_start_time
            actual_fps = frame_count / elapsed
            logger.info(f"📷 Camera {camera_name}: {actual_fps:.1f} FPS | Inference: {inference_time:.3f}s")
            fps_start_time = time.time()
            frame_count = 0
        
        # Send periodic reports
        if current_time - system_stats["last_activity_report"] > ACTIVITY_LOG_INTERVAL:
            threading.Thread(target=send_activity_report).start()

    # Cleanup
    system_stats["cameras"][camera_name]["active"] = False
    cap.release()
    if not HEADLESS:
        cv2.destroyWindow(camera_name)
    logger.info(f"Camera {camera_name} stopped")

# ====================== MAIN APPLICATION ======================
if __name__ == "__main__":
    logger.info("🚀 Starting Speed Detection System")
    
    # Initialize components
    init_telegram_bot()
    init_csv_log()
    network_available = check_network_connection()
    last_network_check = time.time()
    
    # Run initial calibration if needed
    if not os.path.exists(CALIBRATION_FILE):
        logger.info("🔧 Running initial calibration...")
        calibrate_system()
    
    # Start background threads
    threads = [
        threading.Thread(target=handle_telegram_commands, daemon=True),
        threading.Thread(target=process_notification_queues, daemon=True),
        threading.Thread(target=monitor_system_health, daemon=True),
        threading.Thread(
            target=lambda: (
                (lambda: [update_system_stats_firebase() or time.sleep(60) for _ in iter(lambda: system_running, False)])()
            ),
            daemon=True
        )
    ]
    
    for t in threads:
        t.start()
    
    # Start camera processing
    logger.info("📷 Starting camera processing...")
    cameras = [
        (0, "Kamera Utara"),
        (1, "Kamera Selatan")
    ]
    
    # Gunakan ThreadPoolExecutor untuk manajemen thread kamera
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_camera, index, name) for index, name in cameras}
        
        try:
            # Tunggu hingga semua thread selesai (atau hingga system_running False)
            for future in concurrent.futures.as_completed(futures, timeout=1.0):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Camera error: {str(e)}")
        except concurrent.futures.TimeoutError:
            # Timeout tidak masalah karena kita ingin thread terus berjalan
            pass
        
        # Tunggu sampai system_running menjadi False (shutdown atau restart)
        while system_running:
            time.sleep(1)
    
    # Cleanup
    logger.info("🔴 Shutting down system...")
    if telegram_available:
        try:
            if restart_requested:
                bot.send_message(CHAT_ID, "🔄 Sistem sedang memulai ulang...")
            else:
                bot.send_message(CHAT_ID, "🔴 Sistem dimatikan")
            bot.stop_polling()
        except:
            pass
    
    cv2.destroyAllWindows()
    logger.info("✅ System shutdown complete")
    
    # Restart jika diminta
    if restart_requested:
        logger.info("🔄 Restarting system...")
        python = sys.executable
        os.execl(python, python, *sys.argv)
    else:
        os._exit(0)