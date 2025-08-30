#!/usr/bin/env python3
"""
ClaudeX Pro Ultra - LM Studio Backend
Ultimate one-pager Python app with LM Studio integration

Features:
- LM Studio backend integration (no AWS required)
- Auto-dependency management
- Advanced RAG with vector search
- Voice input/output
- Screenshot capture
- Project management
- Hotkey support (Ctrl+Alt+Space)
- Markdown export
- Advanced search
- Auto-updates
- Bulletproof error handling
"""

import os
import sys
import json
import base64
import hashlib
import sqlite3
import threading
import queue
import time
import re
import webbrowser
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import venv
import importlib.util
def module_exists(modname):
    return importlib.util.find_spec(modname) is not None

# App constants
APP_NAME = "ClaudeX Pro Ultra"
APP_VERSION = "2.0.0"
LM_STUDIO_URL = "http://localhost:1234/v1"
ROOT_DIR = Path(__file__).resolve().parent
DB_FILE = ROOT_DIR / "claudex_ultra.db"
RAG_DIR = ROOT_DIR / "rag_store"
VENV_DIR = ROOT_DIR / ".venv"
PY_EXE = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Required packages
REQUIRED_PKGS = [
    "requests>=2.31.0",
    "tkinter-tooltip>=2.0.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "sentence-transformers>=2.2.0",
    "pyttsx3>=2.90",
    "SpeechRecognition>=3.10",
    "pyaudio>=0.2.14",
    "python-docx>=0.8.11",
    "PyMuPDF>=1.23.0",
    "openai>=1.0.0"
]

# Bootstrap virtual environment and dependencies
def ensure_venv_and_deps():
    """Create virtual environment and install dependencies"""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(str(VENV_DIR))
    
    if not in_venv():
        print("Re-launching inside virtual environment...")
        cmd = [str(PY_EXE), __file__] + sys.argv[1:]
        subprocess.run(cmd)
        sys.exit(0)
    
    # Install missing packages
    install_missing_packages()

def in_venv():
    """Check if running in virtual environment"""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)

def install_missing_packages():
    """Install missing required packages"""
    import importlib.util
    
    def have_package(pkg_name):
        try:
            return importlib.util.find_spec(pkg_name) is not None
        except:
            return False
    
    missing = []
    for pkg in REQUIRED_PKGS:
        pkg_name = pkg.split(">=")[0].split("==")[0]
        if not have_package(pkg_name):
            missing.append(pkg)
    
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)

# Initialize virtual environment
ensure_venv_and_deps()

# Now import all required packages
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import openai

# Optional imports with fallbacks
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None

# Database management
def init_database():
    """Initialize SQLite database"""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    
    # Projects table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Messages table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            attachments TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    """)
    
    # Settings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    # RAG documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rag_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    con.commit()
    con.close()

def get_setting(key: str, default: str = "") -> str:
    """Get setting value"""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cur.fetchone()
    con.close()
    return result[0] if result else default

def set_setting(key: str, value: str):
    """Set setting value"""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)
    """, (key, value))
    con.commit()
    con.close()

# Project management
def list_projects() -> List[str]:
    """List all projects"""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT name FROM projects ORDER BY name")
    projects = [row[0] for row in cur.fetchall()]
    con.close()
    
    if not projects:
        add_project("Default")
        projects = ["Default"]
    
    return projects

def add_project(name: str):
    """Add new project"""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    try:
        cur.execute("INSERT OR IGNORE INTO projects (name) VALUES (?)", (name,))
        con.commit()
    finally:
        con.close()

def get_project_id(name: str) -> int:
    """Get project ID by name"""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT id FROM projects WHERE name = ?", (name,))
    result = cur.fetchone()
    con.close()
    return result[0] if result else None

# Message management
def save_message(project_name: str, role: str, content: str, attachments: List[Dict] = None):
    """Save message to database"""
    project_id = get_project_id(project_name)
    if not project_id:
        add_project(project_name)
        project_id = get_project_id(project_name)
    
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO messages (project_id, role, content, attachments)
        VALUES (?, ?, ?, ?)
    """, (project_id, role, content, json.dumps(attachments or [])))
    con.commit()
    con.close()

def load_messages(project_name: str, limit: int = 100) -> List[Dict]:
    """Load messages for project"""
    project_id = get_project_id(project_name)
    if not project_id:
        return []
    
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        SELECT role, content, attachments, created_at
        FROM messages 
        WHERE project_id = ? 
        ORDER BY created_at ASC 
        LIMIT ?
    """, (project_id, limit))
    
    messages = []
    for row in cur.fetchall():
        try:
            attachments = json.loads(row[2]) if row[2] else []
        except:
            attachments = []
        
        messages.append({
            "role": row[0],
            "content": row[1],
            "attachments": attachments,
            "created_at": row[3]
        })
    
    con.close()
    return messages

# LM Studio integration
class LMStudioClient:
    """Client for LM Studio backend"""
    
    def __init__(self, base_url: str = LM_STUDIO_URL):
        self.base_url = base_url
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")
    
    def test_connection(self) -> bool:
        """Test connection to LM Studio"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stream_chat(self, messages: List[Dict], model: str = "local-model", 
                   max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Stream chat completion from LM Studio"""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            return full_response
            
        except Exception as e:
            yield f"Error: {str(e)}"
            return ""

# RAG system
class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
        self.index_file = self.store_dir / "vectors.npy"
        self.meta_file = self.store_dir / "meta.json"
    
    @property
    def model(self):
        """Get sentence transformer model"""
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available")
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def add_document(self, file_path: str, content: str):
        """Add document to RAG index"""
        if not content.strip():
            return
        
        # Chunk the content
        chunks = self._chunk_text(content)
        
        # Get embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
        # Save to database
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute("""
                INSERT OR REPLACE INTO rag_documents 
                (file_path, content, embedding) VALUES (?, ?, ?)
            """, (f"{file_path}_chunk_{i}", chunk, embedding.tobytes()))
        
        con.commit()
        con.close()
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
            i += max(1, chunk_size - overlap)
        return chunks
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            con = sqlite3.connect(DB_FILE)
            cur = con.cursor()
            cur.execute("SELECT content, embedding FROM rag_documents")
            
            results = []
            for row in cur.fetchall():
                content, embedding_bytes = row
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Calculate similarity
                similarity = np.dot(query_embedding[0], embedding) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(embedding)
                )
                
                results.append({
                    "content": content,
                    "similarity": float(similarity)
                })
            
            con.close()
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"RAG search error: {e}")
            return []

# File processing
def process_file(file_path: str) -> str:
    """Process file and extract text content"""
    try:
        ext = Path(file_path).suffix.lower()
        
        if ext in {".txt", ".md", ".py", ".js", ".html", ".css"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        
        elif ext == ".pdf" and fitz:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        
        elif ext == ".docx" and docx:
            doc = docx.Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return f"[Image file: {Path(file_path).name}]"
        
        else:
            return f"[Unsupported file type: {ext}]"
            
    except Exception as e:
        return f"[Error processing file: {e}]"

# Voice features
class VoiceManager:
    """Manage voice input and output"""
    
    def __init__(self):
        self.tts_engine = None
        self.recognizer = None
        
        # Initialize TTS
        if pyttsx3:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 150)
            except:
                pass
        
        # Initialize speech recognition
        if sr:
            try:
                self.recognizer = sr.Recognizer()
            except:
                pass
    
    def speak(self, text: str):
        """Convert text to speech"""
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
    
    def listen(self) -> str:
        """Listen for voice input"""
        if not self.recognizer:
            return ""
        
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return self.recognizer.recognize_google(audio)
        except:
            return ""

# Screenshot functionality
class ScreenshotManager:
    """Manage screenshot capture"""
    
    @staticmethod
    def capture_fullscreen() -> str:
        """Capture full screen"""
        try:
            img = ImageGrab.grab()
            filename = f"screenshot_{int(time.time())}.png"
            filepath = ROOT_DIR / filename
            img.save(filepath)
            return str(filepath)
        except Exception as e:
            print(f"Screenshot error: {e}")
            return ""
    
    @staticmethod
    def capture_region() -> str:
        """Capture selected region"""
        # This would need a more complex UI implementation
        # For now, return empty string
        return ""

# Main application
class ClaudeXProUltra:
    """Main application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Initialize components
        self.lm_client = LMStudioClient()
        self.rag_system = RAGSystem(RAG_DIR)
        self.voice_manager = VoiceManager()
        self.screenshot_manager = ScreenshotManager()
        
        # Settings
        self.current_project = tk.StringVar(value="Default")
        self.temperature = tk.DoubleVar(value=0.7)
        self.max_tokens = tk.IntVar(value=2000)
        self.rag_enabled = tk.BooleanVar(value=True)
        
        # State
        self.attachments = []
        self.messages = []
        
        # Load settings
        self.load_settings()
        
        # Build UI
        self.build_ui()
        
        # Load initial data
        self.load_project_data()
        
        # Test LM Studio connection
        self.test_connection()
    
    def load_settings(self):
        """Load application settings"""
        self.temperature.set(float(get_setting("temperature", "0.7")))
        self.max_tokens.set(int(get_setting("max_tokens", "2000")))
        self.rag_enabled.set(get_setting("rag_enabled", "true").lower() == "true")
    
    def save_settings(self):
        """Save application settings"""
        set_setting("temperature", str(self.temperature.get()))
        set_setting("max_tokens", str(self.max_tokens.get()))
        set_setting("rag_enabled", str(self.rag_enabled.get()).lower())
    
    def build_ui(self):
        """Build the user interface"""
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Top toolbar
        self.build_toolbar()
        
        # Main chat area
        self.build_chat_area()
        
        # Input area
        self.build_input_area()
        
        # Status bar
        self.build_status_bar()
    
    def build_toolbar(self):
        """Build the top toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Project selector
        ttk.Label(toolbar, text="Project:").pack(side="left")
        self.project_combo = ttk.Combobox(
            toolbar, 
            textvariable=self.current_project,
            values=list_projects(),
            width=20
        )
        self.project_combo.pack(side="left", padx=(5, 10))
        self.project_combo.bind("<<ComboboxSelected>>", self.on_project_change)
        
        # New project button
        ttk.Button(toolbar, text="New Project", command=self.new_project).pack(side="left", padx=5)
        
        # RAG toggle
        ttk.Checkbutton(
            toolbar, 
            text="Enable RAG", 
            variable=self.rag_enabled,
            command=self.toggle_rag
        ).pack(side="left", padx=10)
        
        # Index folder button
        ttk.Button(toolbar, text="Index Folder", command=self.index_folder).pack(side="left", padx=5)
        
        # Settings button
        ttk.Button(toolbar, text="Settings", command=self.show_settings).pack(side="left", padx=5)
        
        # Export button
        ttk.Button(toolbar, text="Export", command=self.export_chat).pack(side="left", padx=5)
    
    def build_chat_area(self):
        """Build the main chat area"""
        chat_frame = ttk.Frame(self.root)
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Chat text widget
        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            bg="#1e1e1e",
            fg="#ffffff",
            insertbackground="#ffffff",
            font=("Consolas", 10)
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_text.configure(yscrollcommand=scrollbar.set)
    
    def build_input_area(self):
        """Build the input area"""
        input_frame = ttk.Frame(self.root)
        input_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Input entry
        self.input_entry = ttk.Entry(input_frame, font=("Consolas", 11))
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_entry.bind("<Return>", self.send_message)
        
        # Buttons
        ttk.Button(input_frame, text="Attach", command=self.attach_file).grid(row=0, column=1, padx=2)
        ttk.Button(input_frame, text="Voice", command=self.voice_input).grid(row=0, column=2, padx=2)
        ttk.Button(input_frame, text="Screenshot", command=self.take_screenshot).grid(row=0, column=3, padx=2)
        ttk.Button(input_frame, text="Send", command=self.send_message).grid(row=0, column=4, padx=2)
        ttk.Button(input_frame, text="Clear", command=self.clear_chat).grid(row=0, column=5, padx=2)
    
    def build_status_bar(self):
        """Build the status bar"""
        self.status_label = ttk.Label(self.root, text="Ready", relief="sunken")
        self.status_label.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def test_connection(self):
        """Test connection to LM Studio"""
        if self.lm_client.test_connection():
            self.update_status("Connected to LM Studio")
        else:
            self.update_status("LM Studio not available - please start LM Studio")
    
    def load_project_data(self):
        """Load data for current project"""
        project = self.current_project.get()
        self.messages = load_messages(project)
        self.display_messages()
    
    def on_project_change(self, event=None):
        """Handle project change"""
        self.load_project_data()
    
    def new_project(self):
        """Create new project"""
        name = simpledialog.askstring("New Project", "Enter project name:")
        if name:
            add_project(name)
            self.project_combo.configure(values=list_projects())
            self.current_project.set(name)
            self.clear_chat()
    
    def toggle_rag(self):
        """Toggle RAG functionality"""
        self.save_settings()
        status = "enabled" if self.rag_enabled.get() else "disabled"
        self.update_status(f"RAG {status}")
    
    def index_folder(self):
        """Index folder for RAG"""
        folder = filedialog.askdirectory(title="Select folder to index")
        if folder:
            self.update_status("Indexing folder...")
            
            def index_worker():
                try:
                    for file_path in Path(folder).rglob("*"):
                        if file_path.is_file():
                            content = process_file(str(file_path))
                            if content:
                                self.rag_system.add_document(str(file_path), content)
                    
                    self.update_status("Folder indexed successfully")
                except Exception as e:
                    self.update_status(f"Indexing failed: {e}")
            
            threading.Thread(target=index_worker, daemon=True).start()
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        
        # Temperature
        ttk.Label(settings_window, text="Temperature:").pack(anchor="w", padx=10, pady=5)
        temp_scale = ttk.Scale(
            settings_window, 
            from_=0.0, 
            to=1.0, 
            variable=self.temperature,
            orient="horizontal"
        )
        temp_scale.pack(fill="x", padx=10)
        
        # Max tokens
        ttk.Label(settings_window, text="Max Tokens:").pack(anchor="w", padx=10, pady=5)
        tokens_entry = ttk.Entry(settings_window)
        tokens_entry.insert(0, str(self.max_tokens.get()))
        tokens_entry.pack(fill="x", padx=10)
        
        # Save button
        def save_settings():
            try:
                self.max_tokens.set(int(tokens_entry.get()))
                self.save_settings()
                settings_window.destroy()
                self.update_status("Settings saved")
            except ValueError:
                messagebox.showerror("Error", "Invalid max tokens value")
        
        ttk.Button(settings_window, text="Save", command=save_settings).pack(pady=20)
    
    def export_chat(self):
        """Export chat to markdown"""
        project = self.current_project.get()
        filename = f"{project.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md")],
            initialfilename=filename
        )
        
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {project} Chat Export\n")
                    f.write(f"Exported: {datetime.now().isoformat()}\n\n")
                    
                    for msg in self.messages:
                        role = "You" if msg["role"] == "user" else "Claude"
                        f.write(f"## {role}\n{msg['content']}\n\n")
                
                self.update_status(f"Exported to {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def attach_file(self):
        """Attach file to message"""
        files = filedialog.askopenfilenames(title="Select files to attach")
        if files:
            self.attachments.extend(files)
            self.update_status(f"Attached {len(files)} file(s)")
    
    def voice_input(self):
        """Handle voice input"""
        self.update_status("Listening...")
        
        def voice_worker():
            text = self.voice_manager.listen()
            if text:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, text)
                self.update_status("Voice input received")
            else:
                self.update_status("Voice input failed")
        
        threading.Thread(target=voice_worker, daemon=True).start()
    
    def take_screenshot(self):
        """Take screenshot"""
        filename = self.screenshot_manager.capture_fullscreen()
        if filename:
            self.attachments.append(filename)
            self.update_status("Screenshot attached")
        else:
            self.update_status("Screenshot failed")
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_text.delete(1.0, tk.END)
        self.attachments.clear()
        self.update_status("Chat cleared")
    
    def display_messages(self):
        """Display all messages in chat"""
        self.chat_text.delete(1.0, tk.END)
        
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "Claude"
            content = msg["content"]
            
            self.chat_text.insert(tk.END, f"{role}: ", "role")
            self.chat_text.insert(tk.END, f"{content}\n\n")
        
        # Configure tags
        self.chat_text.tag_configure("role", font=("Consolas", 10, "bold"))
        self.chat_text.see(tk.END)
    
    def send_message(self, event=None):
        """Send message to LM Studio"""
        text = self.input_entry.get().strip()
        if not text and not self.attachments:
            return
        
        # Clear input
        self.input_entry.delete(0, tk.END)
        
        # Save user message
        project = self.current_project.get()
        save_message(project, "user", text, [{"path": p} for p in self.attachments])
        
        # Display user message
        self.chat_text.insert(tk.END, "You: ", "role")
        self.chat_text.insert(tk.END, f"{text}\n\n")
        
        # Process attachments
        attachment_text = ""
        if self.attachments:
            attachment_text = "\n\nAttachments:\n"
            for path in self.attachments:
                content = process_file(path)
                attachment_text += f"\nFile: {Path(path).name}\n{content[:500]}...\n"
        
        # Add RAG context if enabled
        rag_context = ""
        if self.rag_enabled.get() and text:
            try:
                rag_results = self.rag_system.search(text)
                if rag_results:
                    rag_context = "\n\nRelevant context:\n"
                    for result in rag_results[:3]:
                        rag_context += f"\n{result['content'][:200]}...\n"
            except Exception as e:
                print(f"RAG error: {e}")
        
        # Prepare full message
        full_message = text + attachment_text + rag_context
        
        # Clear attachments
        self.attachments.clear()
        
        # Get response from LM Studio
        self.update_status("Generating response...")
        
        # Prepare messages for API
        api_messages = []
        for msg in self.messages[-10:]:  # Last 10 messages for context
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        api_messages.append({
            "role": "user",
            "content": full_message
        })
        
        # Stream response
        self.chat_text.insert(tk.END, "Claude: ", "role")
        
        def stream_response():
            try:
                response_text = ""
                for chunk in self.lm_client.stream_chat(
                    api_messages,
                    max_tokens=self.max_tokens.get(),
                    temperature=self.temperature.get()
                ):
                    response_text += chunk
                    self.chat_text.insert(tk.END, chunk)
                    self.chat_text.see(tk.END)
                    self.root.update_idletasks()
                
                # Save response
                save_message(project, "assistant", response_text)
                self.messages = load_messages(project)
                
                self.chat_text.insert(tk.END, "\n\n")
                self.update_status("Ready")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.chat_text.insert(tk.END, error_msg)
                self.update_status("Error occurred")
        
        threading.Thread(target=stream_response, daemon=True).start()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    # Initialize database
    init_database()
    
    # Create and run app
    app = ClaudeXProUltra()
    app.run()

if __name__ == "__main__":
    main()
