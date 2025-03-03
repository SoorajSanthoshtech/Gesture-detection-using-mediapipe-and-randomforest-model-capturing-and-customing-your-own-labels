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
    # Merge with DEFAULT_CONFIG to ensure all keys are present
    config = {**DEFAULT_CONFIG, **config}
except FileNotFoundError:
    config = DEFAULT_CONFIG
# Always write the complete config back to ensure all keys are present
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

# Generate a symmetric encryption key (host generates, client receives)
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

# Helper functions for hand gesture recognition
def calculate_distance(landmark1, landmark2):
    """Calculate the Euclidean distance between two landmarks."""
    return math.sqrt((landmark2.x - landmark1.x) ** 2 +
                     (landmark2.y - landmark1.y) ** 2 +
                     (landmark2.z - landmark1.z) ** 2)

def calculate_angle(landmark1, landmark2, landmark3):
    """Calculate the angle between three landmarks in degrees."""
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
        """Check if Tailscale is installed on the system."""
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
        """Check if Tailscale is running and get the Tailscale IPv4 address."""
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
        """Prompt the user to start Tailscale manually if it's not running."""
        if self.tailscale_status:
            return True
        
        messagebox.showwarning("Tailscale Not Running", 
                              "Tailscale is not running. Please start Tailscale manually by opening a Command Prompt as an administrator and running 'tailscale up', then sign in with your account.")
        return False

class SignLanguageChatApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize Tailscale manager
        self.tailscale = TailscaleManager()
        
        # Connection state
        self.is_host = False
        self.connected = False
        self.client_socket = None
        self.server_socket = None
        self.partner_addr = None
        self.socket_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.key_received = False  # Track if the client has received the key
        
        # Encryption state
        self.cipher = cipher  # Host uses the generated cipher; client will update this after receiving the key
        
        # Video and gesture state
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            logging.error("Failed to open video capture device")
            messagebox.showerror("Video Error", "Unable to access the webcam. Please ensure a webcam is connected and accessible.")
            self.vid = None
        self.current_prediction = "None"
        self.last_sent_prediction = "None"
        self.message_buffer = []
        self.partner_frame = None
        self.hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.running = True
        self.video_streaming = True  # Can be toggled to disable video streaming
        self.last_video_send_time = 0
        self.current_frame_rate = VIDEO_FRAME_RATE
        self.current_quality = VIDEO_QUALITY
        self.latency_buffer = []
        self.video_feed_active = False  # Track if partner video feed is active
        self.last_gesture_time = 0  # For caching gesture detection
        
        # References to prevent garbage collection
        self.my_photo = None
        self.partner_photo = None
        
        # Message queue for thread-safe communication with size limit
        self.message_queue = Queue(maxsize=100)
        
        # Recording state
        self.recording = False
        self.session_log = []
        
        # Theme state
        self.dark_mode = False
        
        # Setup UI
        self.setup_ui()
        
        # Check Tailscale status (without attempting to start it)
        if not self.check_tailscale():
            self.start_tailscale_service()
        
        # Start the update loop
        self.update()
        self.window.mainloop()
    
    def setup_ui(self):
        """Set up the UI components."""
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tailscale frame
        tailscale_frame = ttk.LabelFrame(self.main_frame, text="Tailscale Status")
        tailscale_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.tailscale_status_label = ttk.Label(tailscale_frame, text="Tailscale: Checking...", foreground="blue")
        self.tailscale_status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.start_tailscale_btn = ttk.Button(tailscale_frame, text="Start Tailscale", 
                                             command=self.start_tailscale_service, state=tk.DISABLED)
        self.start_tailscale_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Connection frame
        connection_frame = ttk.LabelFrame(self.main_frame, text="Connection")
        connection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(connection_frame, text="IP/URL:").grid(row=0, column=0, padx=5, pady=5)
        self.ip_entry = ttk.Entry(connection_frame)
        self.ip_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(connection_frame, text="Port:").grid(row=0, column=2, padx=5, pady=5)
        self.port_entry = ttk.Entry(connection_frame, width=6)
        self.port_entry.grid(row=0, column=3, padx=5, pady=5)
        self.port_entry.insert(0, str(DEFAULT_PORT))
        
        connection_buttons_frame = ttk.Frame(connection_frame)
        connection_buttons_frame.grid(row=0, column=4, columnspan=3, padx=5, pady=5)
        
        self.host_btn = ttk.Button(connection_buttons_frame, text="Host", command=self.start_hosting)
        self.host_btn.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = ttk.Button(connection_buttons_frame, text="Connect", command=self.connect_to_host)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(connection_buttons_frame, text="Disconnect", command=self.handle_disconnect, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        self.connection_status = ttk.Label(connection_frame, text="Not Connected", foreground="red")
        self.connection_status.grid(row=0, column=7, padx=5, pady=5)
        
        # Network health indicator
        self.network_health = ttk.Label(connection_frame, text="Network: N/A", foreground="gray")
        self.network_health.grid(row=0, column=8, padx=5, pady=5)
        
        # Tailscale peers frame
        peers_frame = ttk.LabelFrame(self.main_frame, text="Tailscale Peers")
        peers_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.peer_listbox = tk.Listbox(peers_frame, height=3)
        self.peer_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.refresh_peers_btn = ttk.Button(peers_frame, text="Refresh Peers", command=self.refresh_peers)
        self.refresh_peers_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.use_selected_peer_btn = ttk.Button(peers_frame, text="Use Selected Peer", command=self.use_selected_peer)
        self.use_selected_peer_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Video and chat frame
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side - my video
        my_video_frame = ttk.LabelFrame(content_frame, text="Your Video")
        my_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.my_canvas = tk.Canvas(my_video_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.my_canvas.pack(fill=tk.BOTH, expand=True)
        
        video_controls = ttk.Frame(my_video_frame)
        video_controls.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_gesture = ttk.Label(video_controls, text="Current Gesture: None")
        self.current_gesture.pack(side=tk.LEFT, padx=20)
        
        # Video quality toggle
        ttk.Label(video_controls, text="Video Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_var = tk.StringVar(value="Medium")
        quality_menu = ttk.OptionMenu(video_controls, self.quality_var, "Medium", "Low", "Medium", "High", command=self.update_video_quality)
        quality_menu.pack(side=tk.LEFT, padx=5)
        
        # Video streaming toggle
        self.streaming_var = tk.BooleanVar(value=True)
        self.streaming_toggle = ttk.Checkbutton(video_controls, text="Stream Video", variable=self.streaming_var, command=self.toggle_streaming)
        self.streaming_toggle.pack(side=tk.LEFT, padx=5)
        
        # Middle - partner video
        partner_video_frame = ttk.LabelFrame(content_frame, text="Partner's Video")
        partner_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.partner_canvas = tk.Canvas(partner_video_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.partner_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.partner_gesture = ttk.Label(partner_video_frame, text="Partner Gesture: None")
        self.partner_gesture.pack(padx=5, pady=5)
        
        self.video_feed_status = ttk.Label(partner_video_frame, text="No video feed", foreground="red")
        self.video_feed_status.pack(padx=5, pady=5)
        
        # Right side - chat
        chat_frame = ttk.LabelFrame(content_frame, text="Communication")
        chat_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=5, pady=5, ipadx=5, ipady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=25, height=15, background="white", foreground="black")
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.text_input = ttk.Entry(input_frame, width=20)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_text)
        self.send_btn.pack(side=tk.RIGHT, padx=5)
        
        # Recording and theme controls
        controls_frame = ttk.Frame(chat_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.record_var = tk.BooleanVar(value=False)
        self.record_toggle = ttk.Checkbutton(controls_frame, text="Record Session", variable=self.record_var, command=self.toggle_recording)
        self.record_toggle.pack(side=tk.LEFT, padx=5)
        
        self.theme_var = tk.BooleanVar(value=False)
        self.theme_toggle = ttk.Checkbutton(controls_frame, text="Dark Mode", variable=self.theme_var, command=self.toggle_theme)
        self.theme_toggle.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=5, pady=2)
    
    def update_video_quality(self, *args):
        """Update video quality based on user selection."""
        quality = self.quality_var.get()
        if quality == "Low":
            self.current_quality = LOW_QUALITY
        elif quality == "Medium":
            self.current_quality = VIDEO_QUALITY
        elif quality == "High":
            self.current_quality = HIGH_QUALITY
        logging.info(f"Updated video quality to {quality} (JPEG quality: {self.current_quality})")
    
    def toggle_streaming(self):
        """Toggle video streaming on or off."""
        self.video_streaming = self.streaming_var.get()
        state = "enabled" if self.video_streaming else "disabled"
        logging.info(f"Video streaming {state}")
    
    def toggle_recording(self):
        """Toggle session recording on or off."""
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
        """Toggle between light and dark mode."""
        self.dark_mode = self.theme_var.get()
        if self.dark_mode:
            self.window.configure(bg="#2d2d2d")
            self.main_frame.configure(style="Dark.TFrame")
            self.chat_display.configure(background="#3c3f41", foreground="white")
            style = ttk.Style()
            style.configure("Dark.TFrame", background="#2d2d2d", foreground="white")
            style.configure("Dark.TLabel", background="#2d2d2d", foreground="white")
            style.configure("Dark.TButton", background="#2d2d2d", foreground="white")
            logging.info("Switched to dark mode")
        else:
            self.window.configure(bg="SystemButtonFace")
            self.main_frame.configure(style="TFrame")
            self.chat_display.configure(background="white", foreground="black")
            style = ttk.Style()
            style.configure("TFrame", background="SystemButtonFace")
            style.configure("TLabel", background="SystemButtonFace", foreground="black")
            style.configure("TButton", background="SystemButtonFace", foreground="black")
            logging.info("Switched to light mode")
    
    def check_tailscale(self):
        """Check Tailscale status and update the UI accordingly."""
        if not self.tailscale.check_tailscale_installed():
            messagebox.showerror("Tailscale Error", 
                                "Tailscale is not installed. Please install Tailscale and try again.\n"
                                "Visit https://tailscale.com/download to download Tailscale.")
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
        """Prompt the user to start Tailscale manually."""
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
        """Refresh the list of Tailscale peers."""
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
                        logging.debug(f"Skipping line in tailscale status: {line}")
            else:
                self.status_bar.config(text="Error getting Tailscale peers")
                logging.error(f"tailscale status failed: {result.stderr}")
        except subprocess.SubprocessError as e:
            self.status_bar.config(text=f"Error refreshing peers: {str(e)}")
            logging.error(f"Error refreshing peers: {str(e)}")
    
    def use_selected_peer(self):
        """Use the selected peer from the list."""
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
        """Encrypt data using Fernet symmetric encryption."""
        try:
            # Ensure the data is bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            encrypted = self.cipher.encrypt(data)
            logging.debug(f"Encrypted data of size {len(data)} bytes to {len(encrypted)} bytes")
            return encrypted
        except Exception as e:
            logging.error(f"Encryption error: {str(e)}")
            return None
    
    def decrypt_data(self, encrypted_data, is_binary=False):
        """Decrypt data using Fernet symmetric encryption. If is_binary=True, return raw bytes."""
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            if is_binary:
                logging.debug(f"Decrypted binary data of size {len(encrypted_data)} bytes to {len(decrypted)} bytes")
                return decrypted
            else:
                decoded = decrypted.decode('utf-8')
                logging.debug(f"Decrypted text data of size {len(encrypted_data)} bytes to {len(decoded)} characters")
                return decoded
        except Exception as e:
            logging.error(f"Decryption error: {str(e)}")
            return None
    
    def send_socket_data(self, message_type, content):
        """Helper method to send data over the socket."""
        if not self.connected or self.client_socket is None:
            return False
        
        try:
            # Encrypt the content (except for the key message)
            if message_type != "key":
                encrypted_content = self.encrypt_data(content)
            else:
                encrypted_content = content  # Key is sent unencrypted
            
            if encrypted_content is None and message_type != "key":
                logging.error(f"Failed to encrypt {message_type} content")
                return False
            
            message = {"type": message_type, "content": encrypted_content}
            data = pickle.dumps(message)
            message_size = struct.pack("!I", len(data))
            
            start_time = time.time()
            with self.socket_lock:
                self.client_socket.sendall(message_size + data)
            send_time = time.time() - start_time
            logging.info(f"Sent {message_type} message of size {len(data)} bytes in {send_time:.3f} seconds")
            
            # Update latency buffer for dynamic frame rate adjustment and network health
            self.latency_buffer.append(send_time)
            if len(self.latency_buffer) > 10:
                self.latency_buffer.pop(0)
            avg_latency = sum(self.latency_buffer) / len(self.latency_buffer)
            # Adjust frame rate based on latency
            if avg_latency > 0.1:  # High latency, reduce frame rate
                self.current_frame_rate = max(MIN_FRAME_RATE, self.current_frame_rate - 1)
            elif avg_latency < 0.05:  # Low latency, increase frame rate
                self.current_frame_rate = min(MAX_FRAME_RATE, self.current_frame_rate + 1)
            logging.debug(f"Adjusted frame rate to {self.current_frame_rate} FPS based on avg latency {avg_latency:.3f} seconds")
            
            # Update network health indicator
            if avg_latency > 0.1:
                self.network_health.config(text="Network: Poor", foreground="red")
            elif avg_latency > 0.05:
                self.network_health.config(text="Network: Fair", foreground="orange")
            else:
                self.network_health.config(text="Network: Good", foreground="green")
            
            return True
        except (socket.error, AttributeError) as e:
            logging.error(f"Error sending {message_type}: {str(e)}")
            self.status_bar.config(text=f"Error sending {message_type}: {str(e)}")
            self.handle_disconnect()
            return False
    
    def update(self):
        """Update the video feed, process gestures, and handle messages."""
        if not self.running:
            return
        
        # Process video frame
        if self.vid is None:
            self.window.after(30, self.update)
            return
        
        ret, frame = self.vid.read()
        if not ret:
            logging.error("Failed to read frame from video capture")
            self.window.after(30, self.update)
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        processed_image = image.copy()
        
        # Process hand gestures (cache results to reduce CPU usage)
        current_time = time.time()
        if current_time - self.last_gesture_time >= 0.5:  # Process gestures every 0.5 seconds
            results = self.hands.process(image)
            self.last_gesture_time = current_time
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(processed_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if rf_model and scaler:  # Check if model and scaler are available
                        features = []
                        for i in range(21):
                            for j in range(i + 1, 21):
                                features.append(calculate_distance(hand_landmarks.landmark[i], hand_landmarks.landmark[j]))
                        for i in range(21):
                            for j in range(i + 1, 21):
                                for k in range(j + 1, 21):
                                    features.append(calculate_angle(hand_landmarks.landmark[i], hand_landmarks.landmark[j], hand_landmarks.landmark[k]))
                        
                        features_df = pd.DataFrame([features])
                        try:
                            features_df = pd.DataFrame(scaler.transform(features_df))
                            prediction = rf_model.predict(features_df)[0]
                            logging.info(f"Predicted gesture: {prediction}")
                            timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
                            self.current_prediction = prediction
                            self.current_gesture.config(text=f"Current Gesture: {prediction} ({timestamp})")
                            
                            if prediction != self.last_sent_prediction and prediction != "None":
                                self.message_buffer.append(prediction)
                                self.last_sent_prediction = prediction
                                if self.connected and (self.is_host or self.key_received):
                                    self.send_socket_data("gesture", prediction)
                        except Exception as e:
                            logging.error(f"Prediction error: {str(e)}")
                            self.current_prediction = "None"
                            self.current_gesture.config(text="Current Gesture: None")
        
        cv2.putText(processed_image, f'Gesture: {self.current_prediction}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Render local video
        try:
            self.my_photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(processed_image))
            self.my_canvas.create_image(0, 0, image=self.my_photo, anchor=tk.NW)
            logging.debug("Successfully rendered local video frame")
        except Exception as e:
            logging.error(f"Error rendering local video: {str(e)}")
        
        # Send video frame at controlled frame rate
        current_time = time.time()
        if self.connected and self.video_streaming and (self.is_host or self.key_received) and (current_time - self.last_video_send_time) >= (1.0 / self.current_frame_rate):
            try:
                small_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                ret, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.current_quality])
                if not ret:
                    logging.error("Failed to encode video frame")
                    self.window.after(30, self.update)
                    return
                logging.debug(f"Encoded video frame of size {len(buffer)} bytes")
                if self.send_socket_data("video", buffer.tobytes()):
                    self.last_video_send_time = current_time
                else:
                    logging.error("Failed to send video message")
            except Exception as e:
                logging.error(f"Error sending video: {str(e)}")
                self.handle_disconnect()
        
        # Update partner video
        if self.partner_frame is not None:
            try:
                self.partner_photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.partner_frame))
                self.partner_canvas.create_image(0, 0, image=self.partner_photo, anchor=tk.NW)
                logging.debug("Successfully rendered partner video frame")
                if not self.video_feed_active:
                    self.video_feed_status.config(text="Receiving video...", foreground="green")
                    self.video_feed_active = True
            except Exception as e:
                logging.error(f"Error rendering partner video: {str(e)}")
        else:
            if self.video_feed_active:
                self.video_feed_status.config(text="No video feed", foreground="red")
                self.video_feed_active = False
        
        # Process queued messages
        try:
            while True:
                message_type, content = self.message_queue.get_nowait()
                if message_type == "gesture":
                    self.update_chat("Partner", f"[{content}]")
                    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
                    self.partner_gesture.config(text=f"Partner Gesture: {content} ({timestamp})")
                    # Play a sound notification
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                elif message_type == "text":
                    self.update_chat("Partner", content)
                    # Play a sound notification
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                elif message_type == "video":
                    nparr = np.frombuffer(content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.partner_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        logging.info(f"Successfully updated partner frame of shape {frame.shape}")
                    else:
                        logging.error("Failed to decode partner video frame")
                elif message_type == "key":
                    # Client receives the encryption key from the host
                    self.cipher = Fernet(content)
                    self.key_received = True
                    logging.info("Received and set encryption key from host")
        except Empty:
            pass
        except Queue.Full:
            logging.warning("Message queue full, dropping message")
        
        self.window.after(30, self.update)
    
    def start_hosting(self):
        """Start hosting a session."""
        if self.connected:
            return
        
        if not self.check_tailscale():
            messagebox.showerror("Tailscale Error", "Please connect to Tailscale before hosting.")
            return
        
        try:
            host = "0.0.0.0"  # Listen on all interfaces
            tailscale_ip = self.tailscale.tailscale_ip if self.tailscale.tailscale_ip else "127.0.0.1"
            port = int(self.port_entry.get() or DEFAULT_PORT)
            
            # Check if the port is already in use
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                test_socket.bind((host, port))
                test_socket.close()
            except socket.error as e:
                messagebox.showerror("Port Error", 
                                    f"Port {port} is already in use. Please try a different port or close the application using this port.")
                return
            
            # Try binding to the port, with fallback to alternative ports
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    logging.info(f"Attempting to bind to {host}:{port} (Attempt {attempt + 1}/{max_attempts})")
                    self.server_socket.bind((host, port))
                    break
                except socket.error as se:
                    if attempt < max_attempts - 1:
                        port += 1  # Try the next port
                        logging.warning(f"Port {port - 1} in use, trying port {port}")
                    else:
                        raise socket.error(f"Failed to bind after {max_attempts} attempts: {str(se)}. Ensure the port is not in use and check your firewall.")
            
            self.server_socket.listen(1)
            logging.info(f"Server successfully bound to {host}:{port}")
            
            self.is_host = True
            self.connection_status.config(text="Waiting for connection...", foreground="orange")
            self.status_bar.config(text=f"Hosting on {tailscale_ip}:{port}")
            
            accept_thread = threading.Thread(target=self.accept_connection)
            accept_thread.daemon = True
            accept_thread.start()
            
            message = (
                "CONNECTION INFORMATION:\n"
                f"Your Tailscale IP: {tailscale_ip}\n"
                f"Port: {port}\n\n"
                "Share these details with your partner.\n"
                "Ensure your firewall allows inbound traffic on this port.\n"
                "If using different Tailscale accounts, consider using a manual IP or Tailscale Funnel URL."
            )
            messagebox.showinfo("Connection Information", message)
            self.update_chat("System", f"Hosting on {tailscale_ip}:{port}")
            self.disconnect_btn.config(state=tk.NORMAL)
            self.host_btn.config(state=tk.DISABLED)
            self.connect_btn.config(state=tk.DISABLED)
            
        except socket.error as se:
            logging.error(f"Socket error while hosting: {str(se)}")
            self.status_bar.config(text=f"Error hosting: Failed to bind to port {port}. Port might be in use or blocked by firewall. {str(se)}")
        except Exception as e:
            logging.error(f"Error hosting: {str(e)}")
            self.status_bar.config(text=f"Error hosting: {str(e)}")
    
    def accept_connection(self):
        """Accept an incoming connection."""
        try:
            client_socket, addr = self.server_socket.accept()
            with self.socket_lock:
                self.client_socket = client_socket
                self.partner_addr = addr
                self.connected = True
                self.stop_event.clear()
            
            # Send the encryption key to the client
            if not self.send_socket_data("key", ENCRYPTION_KEY):
                logging.error("Failed to send encryption key to client")
                self.handle_disconnect()
                return
            
            self.window.after(0, lambda: self.connection_status.config(text=f"Connected to {addr[0]}", foreground="green"))
            self.window.after(0, lambda: self.status_bar.config(text=f"Connection established with {addr[0]}:{addr[1]}"))
            
            if self.client_socket is not None:
                receive_thread = threading.Thread(target=self.receive_messages)
                receive_thread.daemon = True
                receive_thread.start()
            else:
                logging.error("Failed to start receive thread: client_socket is None")
                self.handle_disconnect()
                return
            
            self.window.after(0, lambda: self.update_chat("System", f"Connected to {addr[0]}:{addr[1]}"))
        
        except socket.error as e:
            self.window.after(0, lambda: self.status_bar.config(text=f"Connection error: {str(e)}"))
            self.handle_disconnect()
    
    def connect_to_host(self):
        """Connect to a host."""
        if self.connected:
            return
        
        host_input = self.ip_entry.get()
        port = int(self.port_entry.get() or DEFAULT_PORT)
        
        if not host_input:
            messagebox.showerror("Connection Error", "Please enter a host IP address or Tailscale Funnel URL.")
            return
        
        # Check if the input is a Tailscale Funnel URL (e.g., https://hostname.tailscale.net:65432)
        if host_input.startswith("https://"):
            try:
                # Extract hostname and port from the URL
                url_parts = host_input.split(":")
                hostname = url_parts[1].replace("//", "")
                if len(url_parts) > 2:
                    port = int(url_parts[2])
                else:
                    port = int(self.port_entry.get() or DEFAULT_PORT)
            except Exception as e:
                messagebox.showerror("Connection Error", f"Invalid Tailscale Funnel URL: {str(e)}")
                return
        else:
            hostname = host_input
            # Optional: Check Tailscale if the IP isn't manually entered
            if not host_input.startswith("100."):
                if not self.check_tailscale():
                    messagebox.showerror("Tailscale Error", "Please connect to Tailscale before connecting to a host, or enter a manual IP address or Funnel URL.")
                    return
        
        try:
            self.connection_status.config(text="Connecting...", foreground="orange")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(30)  # Increased timeout to 30 seconds
            self.client_socket.connect((hostname, port))
            
            with self.socket_lock:
                self.is_host = False
                self.connected = True
                self.stop_event.clear()
            
            self.connection_status.config(text=f"Connected to {hostname}", foreground="green")
            self.status_bar.config(text=f"Connected to host at {hostname}:{port}")
            self.disconnect_btn.config(state=tk.NORMAL)
            self.host_btn.config(state=tk.DISABLED)
            self.connect_btn.config(state=tk.DISABLED)
            
            if self.client_socket is not None:
                receive_thread = threading.Thread(target=self.receive_messages)
                receive_thread.daemon = True
                receive_thread.start()
            else:
                logging.error("Failed to start receive thread: client_socket is None")
                self.handle_disconnect()
                return
            
            self.update_chat("System", f"Connected to {hostname}:{port}")
        
        except socket.timeout:
            self.status_bar.config(text=f"Connection error: Timed out while trying to connect to {hostname}:{port}.")
            self.handle_disconnect()
        except ConnectionRefusedError:
            self.status_bar.config(text=f"Connection error: The target machine ({hostname}:{port}) actively refused the connection. Ensure the host is running and the port is not blocked by a firewall.")
            self.handle_disconnect()
        except socket.error as e:
            self.status_bar.config(text=f"Connection error: {str(e)}")
            self.handle_disconnect()
    
    def send_gesture(self, gesture):
        """Send a gesture to the connected partner."""
        if self.is_host or self.key_received:
            if self.send_socket_data("gesture", gesture) and self.connected:
                self.update_chat("You", f"[{gesture}]")
                logging.info(f"Successfully sent gesture: {gesture}")
    
    def send_text(self):
        """Send a text message to the connected partner."""
        text = self.text_input.get()
        if not text:
            return
        
        if self.is_host or self.key_received:
            if self.send_socket_data("text", text) and self.connected:
                self.update_chat("You", text)
                logging.info(f"Successfully sent text: {text}")
                self.text_input.delete(0, tk.END)
    
    def receive_messages(self):
        """Receive messages from the connected partner."""
        while not self.stop_event.is_set():
            try:
                with self.socket_lock:
                    if self.client_socket is None or not self.connected:
                        logging.info("Stopping receive_messages: client_socket is None or not connected")
                        break
                
                message_size_data = self.client_socket.recv(4)
                if not message_size_data:
                    logging.info("No data received, connection likely closed")
                    break
                
                message_size = struct.unpack("!I", message_size_data)[0]
                received_data = b""
                remaining_size = message_size
                
                while remaining_size > 0 and not self.stop_event.is_set():
                    chunk = self.client_socket.recv(min(4096, remaining_size))
                    if not chunk:
                        logging.info("No chunk received, connection likely closed")
                        break
                    received_data += chunk
                    remaining_size -= len(chunk)
                
                if not received_data or remaining_size > 0:
                    logging.info("Incomplete data received, breaking receive loop")
                    break
                
                message = pickle.loads(received_data)
                message_type = message["type"]
                encrypted_content = message["content"]
                
                # Process key message for clients
                if message_type == "key" and not self.is_host:
                    content = encrypted_content  # Key is not encrypted
                    # Client receives the encryption key from the host
                    self.cipher = Fernet(content)
                    self.key_received = True
                    logging.info("Received and set encryption key from host")
                    continue
                
                # Skip messages on the client until the key is received
                if not self.is_host and not self.key_received:
                    logging.warning(f"Received {message_type} message before receiving encryption key, skipping")
                    continue
                
                # Process the message
                if message_type == "text" or message_type == "gesture":
                    content = self.decrypt_data(encrypted_content, is_binary=False)
                    if content is None:
                        logging.error(f"Failed to decrypt {message_type} message")
                        continue
                elif message_type == "video":
                    content = self.decrypt_data(encrypted_content, is_binary=True)
                    if content is None:
                        logging.error("Failed to decrypt video message")
                        continue
                else:
                    logging.warning(f"Unknown message type: {message_type}")
                    continue
                
                try:
                    self.message_queue.put_nowait((message_type, content))
                except Queue.Full:
                    logging.warning("Message queue full, dropping message")
            
            except (socket.error, AttributeError) as e:
                logging.error(f"Connection error in receive_messages: {str(e)}")
                self.status_bar.config(text=f"Connection lost: {str(e)}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in receive_messages: {str(e)}")
                break
        
        self.handle_disconnect()
    
    def update_chat(self, sender, message):
        """Update the chat display with a new message."""
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {sender}: {message}"
        
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, formatted_message + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Log to session if recording
        if self.recording:
            self.session_log.append(formatted_message)
    
    def handle_disconnect(self):
        """Handle disconnection from the partner."""
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
                try:
                    self.client_socket.close()
                except:
                    pass
            
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
            
            self.client_socket = None
            self.server_socket = None
            self.is_host = False
            self.partner_frame = None
            self.key_received = False
            self.video_feed_active = False
            self.video_feed_status.config(text="No video feed", foreground="red")
    
    def on_closing(self):
        """Handle window closing."""
        self.running = False
        self.stop_event.set()
        
        if self.vid and self.vid.isOpened():
            self.vid.release()
        
        # Save session log if recording
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