import numpy as np
import joblib
import cv2
import math
import pandas as pd
import mediapipe as mp
import socket
import pickle
import struct
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import PIL.Image, PIL.ImageTk
import time
import subprocess
import os
import platform
import logging
from queue import Queue, Empty
import json
from cryptography.fernet import Fernet
import winsound  # For sound notifications on Windows
from datetime import datetime

# Setup logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s"
)

# Define default configuration
DEFAULT_CONFIG = {
    "default_port": 65432,
    "video_frame_rate": 15,
    "video_quality": 50,
    "video_width": 400,
    "video_height": 300,
    "min_frame_rate": 5,
    "max_frame_rate": 30,
    "low_quality": 30,
    "high_quality": 90
}

# Load or create configuration file
CONFIG_FILE = "config.json"
try:
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    config = {**DEFAULT_CONFIG, **config}
except FileNotFoundError:
    config = DEFAULT_CONFIG
with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=4)

# Constants from config
DEFAULT_PORT = config["default_port"]
VIDEO_FRAME_RATE = config["video_frame_rate"]
VIDEO_QUALITY = config["video_quality"]
VIDEO_WIDTH = config["video_width"]
VIDEO_HEIGHT = config["video_height"]
MIN_FRAME_RATE = config["min_frame_rate"]
MAX_FRAME_RATE = config["max_frame_rate"]
LOW_QUALITY = config["low_quality"]
HIGH_QUALITY = config["high_quality"]

# Load the trained model and scaler (with fallback)
MODEL_FILENAME = "new_model3.pkl"
SCALER_FILENAME = "scaler3.pkl"
rf_model = None
scaler = None
try:
    rf_model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.warning(f"Model and scaler not found: {str(e)}. Gesture recognition will be disabled.")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Generate a symmetric encryption key
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

# Helper functions for hand gesture recognition
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x) ** 2 +
                     (landmark2.y - landmark1.y) ** 2 +
                     (landmark2.z - landmark1.z) ** 2)

def calculate_angle(landmark1, landmark2, landmark3):
    vec1 = [landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z]
    vec2 = [landmark3.x - landmark2.x, landmark3.y - landmark2.y, landmark3.z - landmark2.z]
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    return 0.0 if magnitude1 == 0 or magnitude2 == 0 else math.degrees(math.acos(dot_product / (magnitude1 * magnitude2)))

class TailscaleManager:
    def __init__(self):
        self.tailscale_status = False
        self.tailscale_ip = None
    
    def check_tailscale_installed(self):
        try:
            system = platform.system()
            if system == "Windows":
                return os.path.exists(r"C:\Program Files\Tailscale\tailscale.exe")
            else:
                result = subprocess.run(["which", "tailscale"], capture_output=True, text=True)
                return result.returncode == 0
        except Exception as e:
            logging.error(f"Error checking Tailscale installation: {str(e)}")
            return False
    
    def check_tailscale_status(self):
        try:
            system = platform.system()
            cmd = ["tailscale", "ip", "-4"] if system != "Windows" else [r"C:\Program Files\Tailscale\tailscale.exe", "ip", "-4"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                ip = result.stdout.strip()
                if ip and ip.count('.') == 3:
                    self.tailscale_status = True
                    self.tailscale_ip = ip
                    logging.info(f"Tailscale IP detected: {self.tailscale_ip}")
                    return True, self.tailscale_ip
            self.tailscale_status = False
            return False, None
        except Exception as e:
            logging.error(f"Error checking Tailscale status: {str(e)}")
            return False, None
    
    def start_tailscale(self):
        if self.tailscale_status:
            return True
        messagebox.showwarning("Tailscale Not Running", 
                              "Tailscale is not running. Please start Tailscale manually.")
        return False

class SignLanguageChatApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.tailscale = TailscaleManager()
        self.is_host = False
        self.connected = False
        self.client_socket = None
        self.server_socket = None
        self.partner_addr = None
        self.socket_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.key_received = False
        self.cipher = cipher
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            logging.error("Failed to open video capture device")
            messagebox.showerror("Video Error", "Unable to access webcam.")
            self.vid = None
        self.current_prediction = "None"
        self.last_sent_prediction = "None"
        self.message_buffer = []
        self.partner_frame = None
        self.hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.running = True
        self.video_streaming = True
        self.last_video_send_time = 0
        self.current_frame_rate = VIDEO_FRAME_RATE
        self.current_quality = VIDEO_QUALITY
        self.latency_buffer = []
        self.video_feed_active = False
        self.last_gesture_time = 0
        self.my_photo = None
        self.partner_photo = None
        self.message_queue = Queue(maxsize=100)
        self.recording = False
        self.session_log = []
        self.dark_mode = False
        
        self.setup_ui()
        if not self.check_tailscale():
            self.start_tailscale_service()
        self.update()
        self.window.mainloop()
    
    def setup_ui(self):
        """Set up an enhanced, organized, and attractive UI."""
        self.window.configure(bg="#f0f0f0")
        self.window.geometry("1200x700")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10, "bold"))
        style.configure("Tab.TNotebook", background="#e0e0e0", padding=[5, 5])
        style.configure("Tab.TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=[10, 5])
        style.configure("Accent.TButton", background="#0078d7", foreground="white")

        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.connection_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.connection_tab, text="Connection")
        self._setup_connection_tab()

        self.video_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="Video Streams")
        self._setup_video_tab()

        self.chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_tab, text="Chat")
        self._setup_chat_tab()

        self.status_frame = ttk.Frame(self.window, relief=tk.SUNKEN)
        self.status_frame.pack(fill=tk.X, padx=10, pady=2)
        self.status_bar = ttk.Label(self.status_frame, text="Ready", anchor=tk.W, font=("Helvetica", 9))
        self.status_bar.pack(fill=tk.X)

    def _setup_connection_tab(self):
        tailscale_frame = ttk.LabelFrame(self.connection_tab, text="Tailscale Network", padding=10)
        tailscale_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.tailscale_status_label = ttk.Label(tailscale_frame, text="Tailscale: Checking...", foreground="blue", font=("Helvetica", 10, "italic"))
        self.tailscale_status_label.pack(side=tk.LEFT, padx=5)
        
        self.start_tailscale_btn = ttk.Button(tailscale_frame, text="Start Tailscale", command=self.start_tailscale_service, state=tk.DISABLED)
        self.start_tailscale_btn.pack(side=tk.RIGHT, padx=5)

        conn_frame = ttk.LabelFrame(self.connection_tab, text="Connect to Partner", padding=10)
        conn_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(conn_frame, text="IP/URL:").grid(row=0, column=0, padx=5, pady=5)
        self.ip_entry = ttk.Entry(conn_frame, width=25)
        self.ip_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=2, padx=5, pady=5)
        self.port_entry = ttk.Entry(conn_frame, width=6)
        self.port_entry.grid(row=0, column=3, padx=5, pady=5)
        self.port_entry.insert(0, str(DEFAULT_PORT))
        
        self.host_btn = ttk.Button(conn_frame, text="Host Session", command=self.start_hosting, style="Accent.TButton")
        self.host_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Join Session", command=self.connect_to_host, style="Accent.TButton")
        self.connect_btn.grid(row=1, column=2, columnspan=2, pady=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", command=self.handle_disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=2, column=0, columnspan=4, pady=5)

        self.connection_status = ttk.Label(conn_frame, text="Not Connected", foreground="red")
        self.connection_status.grid(row=3, column=0, columnspan=4, pady=5)

        self.network_health = ttk.Label(conn_frame, text="Network: N/A", foreground="gray")
        self.network_health.grid(row=4, column=0, columnspan=4, pady=5)

        peers_frame = ttk.LabelFrame(self.connection_tab, text="Available Peers", padding=10)
        peers_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        self.peer_listbox = tk.Listbox(peers_frame, height=6, font=("Helvetica", 10))
        self.peer_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        peer_btn_frame = ttk.Frame(peers_frame)
        peer_btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.refresh_peers_btn = ttk.Button(peer_btn_frame, text="Refresh", command=self.refresh_peers)
        self.refresh_peers_btn.pack(pady=2)
        
        self.use_selected_peer_btn = ttk.Button(peer_btn_frame, text="Use Peer", command=self.use_selected_peer)
        self.use_selected_peer_btn.pack(pady=2)

        self.connection_tab.columnconfigure((0, 1), weight=1)
        self.connection_tab.rowconfigure(1, weight=1)

    def _setup_video_tab(self):
        my_video_frame = ttk.LabelFrame(self.video_tab, text="Your Video Feed", padding=10)
        my_video_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.my_canvas = tk.Canvas(my_video_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, bg="black")
        self.my_canvas.pack(fill=tk.BOTH, expand=True)
        
        controls_frame = ttk.Frame(my_video_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.current_gesture = ttk.Label(controls_frame, text="Current Gesture: None", font=("Helvetica", 10))
        self.current_gesture.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(controls_frame, text="Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_var = tk.StringVar(value="Medium")
        quality_menu = ttk.OptionMenu(controls_frame, self.quality_var, "Medium", "Low", "Medium", "High", command=self.update_video_quality)
        quality_menu.pack(side=tk.LEFT, padx=5)
        
        self.streaming_var = tk.BooleanVar(value=True)
        self.streaming_toggle = ttk.Checkbutton(controls_frame, text="Stream Video", variable=self.streaming_var, command=self.toggle_streaming)
        self.streaming_toggle.pack(side=tk.LEFT, padx=5)

        partner_video_frame = ttk.LabelFrame(self.video_tab, text="Partner's Video Feed", padding=10)
        partner_video_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.partner_canvas = tk.Canvas(partner_video_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, bg="black")
        self.partner_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.partner_gesture = ttk.Label(partner_video_frame, text="Partner Gesture: None", font=("Helvetica", 10))
        self.partner_gesture.pack(pady=5)
        
        self.video_feed_status = ttk.Label(partner_video_frame, text="No video feed", foreground="red")
        self.video_feed_status.pack(pady=5)

        self.video_tab.columnconfigure((0, 1), weight=1)
        self.video_tab.rowconfigure(0, weight=1)

    def _setup_chat_tab(self):
        chat_frame = ttk.LabelFrame(self.chat_tab, text="Chat Window", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=40, height=20, font=("Helvetica", 10), bg="white", fg="black")
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.text_input = ttk.Entry(input_frame, width=30, font=("Helvetica", 10))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_text, style="Accent.TButton")
        self.send_btn.pack(side=tk.RIGHT, padx=5)

        controls_frame = ttk.Frame(chat_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.record_var = tk.BooleanVar(value=False)
        self.record_toggle = ttk.Checkbutton(controls_frame, text="Record Session", variable=self.record_var, command=self.toggle_recording)
        self.record_toggle.pack(side=tk.LEFT, padx=10)
        
        self.theme_var = tk.BooleanVar(value=False)
        self.theme_toggle = ttk.Checkbutton(controls_frame, text="Dark Mode", variable=self.theme_var, command=self.toggle_theme)
        self.theme_toggle.pack(side=tk.LEFT, padx=10)

    def update_video_quality(self, *args):
        quality = self.quality_var.get()
        if quality == "Low":
            self.current_quality = LOW_QUALITY
        elif quality == "Medium":
            self.current_quality = VIDEO_QUALITY
        elif quality == "High":
            self.current_quality = HIGH_QUALITY
        logging.info(f"Updated video quality to {quality} (JPEG quality: {self.current_quality})")
    
    def toggle_streaming(self):
        self.video_streaming = self.streaming_var.get()
        state = "enabled" if self.video_streaming else "disabled"
        logging.info(f"Video streaming {state}")
    
    def toggle_recording(self):
        self.recording = self.record_var.get()
        if self.recording:
            self.session_log = []
            logging.info("Started recording session")
        else:
            if self.session_log:
                filename = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.session_log))
                logging.info(f"Saved session log to {filename}")
            self.session_log = []
    
    def toggle_theme(self):
        self.dark_mode = self.theme_var.get()
        style = ttk.Style()
        if self.dark_mode:
            self.window.configure(bg="#2d2d2d")
            style.configure("TFrame", background="#2d2d2d", foreground="white")
            style.configure("TLabel", background="#2d2d2d", foreground="white")
            style.configure("TButton", background="#4a4a4a", foreground="white")
            style.configure("Tab.TNotebook", background="#3c3f41")
            style.configure("Tab.TNotebook.Tab", background="#3c3f41", foreground="white")
            style.configure("Accent.TButton", background="#1e90ff", foreground="white")
            self.chat_display.configure(bg="#3c3f41", fg="white")
            self.my_canvas.configure(bg="#1a1a1a")
            self.partner_canvas.configure(bg="#1a1a1a")
            self.status_frame.configure(style="Dark.TFrame")
            logging.info("Switched to dark mode")
        else:
            self.window.configure(bg="#f0f0f0")
            style.configure("TFrame", background="#f0f0f0", foreground="black")
            style.configure("TLabel", background="#f0f0f0", foreground="black")
            style.configure("TButton", background="#e0e0e0", foreground="black")
            style.configure("Tab.TNotebook", background="#e0e0e0")
            style.configure("Tab.TNotebook.Tab", background="#e0e0e0", foreground="black")
            style.configure("Accent.TButton", background="#0078d7", foreground="white")
            self.chat_display.configure(bg="white", fg="black")
            self.my_canvas.configure(bg="black")
            self.partner_canvas.configure(bg="black")
            self.status_frame.configure(style="TFrame")
            logging.info("Switched to light mode")
    
    def check_tailscale(self):
        if not self.tailscale.check_tailscale_installed():
            messagebox.showerror("Tailscale Error", 
                                "Tailscale is not installed. Please install Tailscale.")
            self.tailscale_status_label.config(text="Tailscale: Not Installed", foreground="red")
            return False
            
        status, ip = self.tailscale.check_tailscale_status()
        if status:
            self.tailscale_status_label.config(text=f"Tailscale: Connected ({ip})", foreground="green")
            self.ip_entry.delete(0, tk.END)
            self.ip_entry.insert(0, ip)
            return True
        else:
            self.tailscale_status_label.config(text="Tailscale: Not Connected", foreground="orange")
            self.start_tailscale_btn.config(state=tk.NORMAL)
            return False
    
    def start_tailscale_service(self):
        if self.tailscale.start_tailscale():
            status, ip = self.tailscale.check_tailscale_status()
            if status:
                self.tailscale_status_label.config(text=f"Tailscale: Connected ({ip})", foreground="green")
                self.ip_entry.delete(0, tk.END)
                self.ip_entry.insert(0, ip)
                self.start_tailscale_btn.config(state=tk.DISABLED)
                return True
        return False
    
    def refresh_peers(self):
        try:
            system = platform.system()
            cmd = ["tailscale", "status"] if system != "Windows" else [r"C:\Program Files\Tailscale\tailscale.exe", "status"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                self.peer_listbox.delete(0, tk.END)
                lines = result.stdout.splitlines()
                for line in lines:
                    if not line.strip() or "Tailscale" in line or "IP" in line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    ip = parts[0]
                    if ip.count('.') == 3 and all(part.isdigit() and 0 <= int(part) <= 255 for part in ip.split('.')):
                        hostname = parts[1] if len(parts) > 1 else "Unknown"
                        self.peer_listbox.insert(tk.END, f"{hostname} ({ip})")
            else:
                self.status_bar.config(text="Error getting Tailscale peers")
                logging.error(f"tailscale status failed: {result.stderr}")
        except subprocess.SubprocessError as e:
            self.status_bar.config(text=f"Error refreshing peers: {str(e)}")
            logging.error(f"Error refreshing peers: {str(e)}")
    
    def use_selected_peer(self):
        selection = self.peer_listbox.curselection()
        if not selection:
            messagebox.showinfo("Peer Selection", "Please select a peer first")
            return
            
        peer_info = self.peer_listbox.get(selection[0])
        ip_match = peer_info.split("(")[1].split(")")[0] if "(" in peer_info else peer_info
        
        self.ip_entry.delete(0, tk.END)
        self.ip_entry.insert(0, ip_match)
        self.status_bar.config(text=f"Selected peer: {peer_info}")
    
    def encrypt_data(self, data):
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            encrypted = self.cipher.encrypt(data)
            return encrypted
        except Exception as e:
            logging.error(f"Encryption error: {str(e)}")
            return None
    
    def decrypt_data(self, encrypted_data, is_binary=False):
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return decrypted if is_binary else decrypted.decode('utf-8')
        except Exception as e:
            logging.error(f"Decryption error: {str(e)}")
            return None
    
    def send_socket_data(self, message_type, content):
        if not self.connected or self.client_socket is None:
            return False
        
        try:
            if message_type != "key":
                encrypted_content = self.encrypt_data(content)
            else:
                encrypted_content = content
            
            if encrypted_content is None and message_type != "key":
                return False
            
            message = {"type": message_type, "content": encrypted_content}
            data = pickle.dumps(message)
            message_size = struct.pack("!I", len(data))
            
            start_time = time.time()
            with self.socket_lock:
                self.client_socket.sendall(message_size + data)
            send_time = time.time() - start_time
            
            self.latency_buffer.append(send_time)
            if len(self.latency_buffer) > 10:
                self.latency_buffer.pop(0)
            avg_latency = sum(self.latency_buffer) / len(self.latency_buffer)
            if avg_latency > 0.1:
                self.current_frame_rate = max(MIN_FRAME_RATE, self.current_frame_rate - 1)
            elif avg_latency < 0.05:
                self.current_frame_rate = min(MAX_FRAME_RATE, self.current_frame_rate + 1)
            
            if avg_latency > 0.1:
                self.network_health.config(text="Network: Poor", foreground="red")
            elif avg_latency > 0.05:
                self.network_health.config(text="Network: Fair", foreground="orange")
            else:
                self.network_health.config(text="Network: Good", foreground="green")
            
            return True
        except (socket.error, AttributeError) as e:
            logging.error(f"Error sending {message_type}: {str(e)}")
            self.handle_disconnect()
            return False
    
    def update(self):
        if not self.running:
            return
        
        if self.vid is None:
            self.window.after(30, self.update)
            return
        
        ret, frame = self.vid.read()
        if not ret:
            self.window.after(30, self.update)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        processed_image = image.copy()
        
        current_time = time.time()
        if current_time - self.last_gesture_time >= 0.5:
            results = self.hands.process(image)
            self.last_gesture_time = current_time
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(processed_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if rf_model and scaler:
                        features = []
                        for i in range(21):
                            for j in range(i + 1, 21):
                                features.append(calculate_distance(hand_landmarks.landmark[i], hand_landmarks.landmark[j]))
                        for i in range(21):
                            for j in range(i + 1, 21):
                                for k in range(j + 1, 21):
                                    features.append(calculate_angle(hand_landmarks.landmark[i], hand_landmarks.landmark[j], hand_landmarks.landmark[k]))
                        
                        features_df = pd.DataFrame([features])
                        features_df = pd.DataFrame(scaler.transform(features_df))
                        prediction = rf_model.predict(features_df)[0]
                        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
                        self.current_prediction = prediction
                        self.current_gesture.config(text=f"Current Gesture: {prediction} ({timestamp})")
                        
                        if prediction != self.last_sent_prediction and prediction != "None":
                            self.message_buffer.append(prediction)
                            self.last_sent_prediction = prediction
                            if self.connected and (self.is_host or self.key_received):
                                self.send_socket_data("gesture", prediction)
        
        cv2.putText(processed_image, f'Gesture: {self.current_prediction}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        try:
            self.my_photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(processed_image))
            self.my_canvas.create_image(0, 0, image=self.my_photo, anchor=tk.NW)
        except Exception as e:
            logging.error(f"Error rendering local video: {str(e)}")
        
        current_time = time.time()
        if self.connected and self.video_streaming and (self.is_host or self.key_received) and (current_time - self.last_video_send_time) >= (1.0 / self.current_frame_rate):
            try:
                small_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                ret, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.current_quality])
                if ret and self.send_socket_data("video", buffer.tobytes()):
                    self.last_video_send_time = current_time
            except Exception as e:
                logging.error(f"Error sending video: {str(e)}")
                self.handle_disconnect()
        
        if self.partner_frame is not None:
            try:
                self.partner_photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.partner_frame))
                self.partner_canvas.create_image(0, 0, image=self.partner_photo, anchor=tk.NW)
                if not self.video_feed_active:
                    self.video_feed_status.config(text="Receiving video...", foreground="green")
                    self.video_feed_active = True
            except Exception as e:
                logging.error(f"Error rendering partner video: {str(e)}")
        else:
            if self.video_feed_active:
                self.video_feed_status.config(text="No video feed", foreground="red")
                self.video_feed_active = False
        
        try:
            while True:
                message_type, content = self.message_queue.get_nowait()
                if message_type == "gesture":
                    self.update_chat("Partner", f"[{content}]")
                    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
                    self.partner_gesture.config(text=f"Partner Gesture: {content} ({timestamp})")
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                elif message_type == "text":
                    self.update_chat("Partner", content)
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                elif message_type == "video":
                    nparr = np.frombuffer(content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.partner_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif message_type == "key":
                    self.cipher = Fernet(content)
                    self.key_received = True
                    logging.info("Received and set encryption key from host")
        except Empty:
            pass
        
        self.window.after(30, self.update)
    
    def start_hosting(self):
        if self.connected:
            return
        
        if not self.check_tailscale():
            messagebox.showerror("Tailscale Error", "Please connect to Tailscale before hosting.")
            return
        
        try:
            host = "0.0.0.0"
            tailscale_ip = self.tailscale.tailscale_ip or "127.0.0.1"
            port = int(self.port_entry.get() or DEFAULT_PORT)
            
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                test_socket.bind((host, port))
                test_socket.close()
            except socket.error:
                messagebox.showerror("Port Error", f"Port {port} is in use.")
                return
            
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((host, port))
            self.server_socket.listen(1)
            
            self.is_host = True
            self.connection_status.config(text="Waiting for connection...", foreground="orange")
            self.status_bar.config(text=f"Hosting on {tailscale_ip}:{port}")
            
            accept_thread = threading.Thread(target=self.accept_connection)
            accept_thread.daemon = True
            accept_thread.start()
            
            messagebox.showinfo("Connection Information", 
                                f"Your Tailscale IP: {tailscale_ip}\nPort: {port}\nShare these details.")
            self.update_chat("System", f"Hosting on {tailscale_ip}:{port}")
            self.disconnect_btn.config(state=tk.NORMAL)
            self.host_btn.config(state=tk.DISABLED)
            self.connect_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            logging.error(f"Error hosting: {str(e)}")
            self.status_bar.config(text=f"Error hosting: {str(e)}")
    
    def accept_connection(self):
        try:
            client_socket, addr = self.server_socket.accept()
            with self.socket_lock:
                self.client_socket = client_socket
                self.partner_addr = addr
                self.connected = True
                self.stop_event.clear()
            
            if not self.send_socket_data("key", ENCRYPTION_KEY):
                self.handle_disconnect()
                return
            
            self.window.after(0, lambda: self.connection_status.config(text=f"Connected to {addr[0]}", foreground="green"))
            self.window.after(0, lambda: self.status_bar.config(text=f"Connected to {addr[0]}:{addr[1]}"))
            
            if self.client_socket:
                receive_thread = threading.Thread(target=self.receive_messages)
                receive_thread.daemon = True
                receive_thread.start()
            
            self.window.after(0, lambda: self.update_chat("System", f"Connected to {addr[0]}:{addr[1]}"))
        
        except socket.error as e:
            self.window.after(0, lambda: self.status_bar.config(text=f"Connection error: {str(e)}"))
            self.handle_disconnect()
    
    def connect_to_host(self):
        if self.connected:
            return
        
        host_input = self.ip_entry.get()
        port = int(self.port_entry.get() or DEFAULT_PORT)
        
        if not host_input:
            messagebox.showerror("Connection Error", "Please enter a host IP or URL.")
            return
        
        if host_input.startswith("https://"):
            url_parts = host_input.split(":")
            hostname = url_parts[1].replace("//", "")
            if len(url_parts) > 2:
                port = int(url_parts[2])
        else:
            hostname = host_input
            if not host_input.startswith("100.") and not self.check_tailscale():
                messagebox.showerror("Tailscale Error", "Please connect to Tailscale.")
                return
        
        try:
            self.connection_status.config(text="Connecting...", foreground="orange")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(30)
            self.client_socket.connect((hostname, port))
            
            with self.socket_lock:
                self.is_host = False
                self.connected = True
                self.stop_event.clear()
            
            self.connection_status.config(text=f"Connected to {hostname}", foreground="green")
            self.status_bar.config(text=f"Connected to {hostname}:{port}")
            self.disconnect_btn.config(state=tk.NORMAL)
            self.host_btn.config(state=tk.DISABLED)
            self.connect_btn.config(state=tk.DISABLED)
            
            if self.client_socket:
                receive_thread = threading.Thread(target=self.receive_messages)
                receive_thread.daemon = True
                receive_thread.start()
            
            self.update_chat("System", f"Connected to {hostname}:{port}")
        
        except socket.timeout:
            self.status_bar.config(text=f"Connection timed out: {hostname}:{port}")
            self.handle_disconnect()
        except ConnectionRefusedError:
            self.status_bar.config(text=f"Connection refused: {hostname}:{port}")
            self.handle_disconnect()
        except socket.error as e:
            self.status_bar.config(text=f"Connection error: {str(e)}")
            self.handle_disconnect()
    
    def send_gesture(self, gesture):
        if (self.is_host or self.key_received) and self.send_socket_data("gesture", gesture) and self.connected:
            self.update_chat("You", f"[{gesture}]")
    
    def send_text(self):
        text = self.text_input.get()
        if not text:
            return
        
        if (self.is_host or self.key_received) and self.send_socket_data("text", text) and self.connected:
            self.update_chat("You", text)
            self.text_input.delete(0, tk.END)
    
    def receive_messages(self):
        while not self.stop_event.is_set():
            try:
                with self.socket_lock:
                    if self.client_socket is None or not self.connected:
                        break
                
                message_size_data = self.client_socket.recv(4)
                if not message_size_data:
                    break
                
                message_size = struct.unpack("!I", message_size_data)[0]
                received_data = b""
                remaining_size = message_size
                
                while remaining_size > 0:
                    chunk = self.client_socket.recv(min(4096, remaining_size))
                    if not chunk:
                        break
                    received_data += chunk
                    remaining_size -= len(chunk)
                
                if not received_data or remaining_size > 0:
                    break
                
                message = pickle.loads(received_data)
                message_type = message["type"]
                encrypted_content = message["content"]
                
                if message_type == "key" and not self.is_host:
                    self.cipher = Fernet(encrypted_content)
                    self.key_received = True
                    continue
                
                if not self.is_host and not self.key_received:
                    continue
                
                if message_type in ("text", "gesture"):
                    content = self.decrypt_data(encrypted_content, is_binary=False)
                elif message_type == "video":
                    content = self.decrypt_data(encrypted_content, is_binary=True)
                else:
                    continue
                
                if content is not None:
                    self.message_queue.put_nowait((message_type, content))
            
            except (socket.error, AttributeError) as e:
                logging.error(f"Connection error: {str(e)}")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                break
        
        self.handle_disconnect()
    
    def update_chat(self, sender, message):
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {sender}: {message}"
        
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, formatted_message + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        if self.recording:
            self.session_log.append(formatted_message)
    
    def handle_disconnect(self):
        with self.socket_lock:
            if not self.connected:
                return
            
            self.connected = False
            self.stop_event.set()
            self.connection_status.config(text="Disconnected", foreground="red")
            self.disconnect_btn.config(state=tk.DISABLED)
            self.host_btn.config(state=tk.NORMAL)
            self.connect_btn.config(state=tk.NORMAL)
            
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            
            self.client_socket = None
            self.server_socket = None
            self.is_host = False
            self.partner_frame = None
            self.key_received = False
            self.video_feed_active = False
            self.video_feed_status.config(text="No video feed", foreground="red")
    
    def on_closing(self):
        self.running = False
        self.stop_event.set()
        
        if self.vid and self.vid.isOpened():
            self.vid.release()
        
        if self.recording and self.session_log:
            filename = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(self.session_log))
            logging.info(f"Saved session log to {filename}")
        
        self.handle_disconnect()
        self.hands.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageChatApp(root, "Sign Language Video Chat")