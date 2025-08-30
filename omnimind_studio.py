import os
import pathlib
print("RUNNING FROM:", pathlib.Path(__file__).resolve())
print("CWD:", pathlib.Path(os.getcwd()).resolve())

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
import difflib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
from abc import ABC, abstractmethod
import subprocess

# Headless check for CI environments
HEADLESS = os.getenv("OMNI_HEADLESS") == "1"

# --- Logging Setup ---
def setup_logging():
    """Sets up a rotating file logger."""
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "omnimind.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create a rotating file handler
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    logger.addHandler(handler)
    
    # Also log to console for immediate feedback, but only if not headless
    if not HEADLESS:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    logging.info("Logging initialized.")

# --- Dependency Management ---
# This application uses a requirements.txt file for dependency management.
# This is a standard and robust approach that ensures a consistent environment.
# A previous version used a self-bootstrapping mechanism, but it was removed
# in favor of this more reliable method.

# Safe, non-GUI imports
try:
    import requests
    from PIL import Image, ImageGrab
    import numpy as np
    import openai
    import keyring
    import anthropic
    from mistralai.client import MistralClient
    import google.generativeai as genai
    import apsw
    import vectorlite_py
    import tiktoken
    import fitz
    import docx
    import speech_recognition as sr
    import pyttsx3
    from sentence_transformers import SentenceTransformer
    import pytesseract
except ImportError as e:
    print("="*80)
    print(f"ERROR: A required dependency is not installed. Full error: {e}")
    print("Please install all dependencies by running:")
    print(f"    pip install -r requirements.txt")
    print("="*80)
    sys.exit(1)

# --- CORE ENGINE (SAFE TO IMPORT) ---

# App constants
APP_NAME = "OmniMind Studio"
APP_VERSION = "3.0.0"
LM_STUDIO_URL = "http://localhost:1234/v1"
OLLAMA_URL = "http://localhost:11434/v1"
ROOT_DIR = Path(__file__).resolve().parent
DB_FILE = ROOT_DIR / "claudex_ultra.db"

class SecretsManager:
    SERVICE_NAME = "OmniMindStudio"
    @staticmethod
    def set_api_key(backend_name: str, api_key: str): keyring.set_password(SecretsManager.SERVICE_NAME, backend_name, api_key)
    @staticmethod
    def get_api_key(backend_name: str) -> Optional[str]: return keyring.get_password(SecretsManager.SERVICE_NAME, backend_name)

def create_db_connection(db_file):
    conn = apsw.Connection(str(db_file)); conn.enable_load_extension(True); conn.load_extension(vectorlite_py.vectorlite_path())
    print("VectorLite extension loaded."); return conn

def init_database(conn: apsw.Connection):
    conn.execute("CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, project_id INTEGER, role TEXT NOT NULL, content TEXT NOT NULL, attachments TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (project_id) REFERENCES projects (id))")
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS rag_documents (id INTEGER PRIMARY KEY, source_name TEXT NOT NULL, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS prompt_templates (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, template TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    embedding_dim = 384
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS rag_vectors USING vectorlite(embedding float32[{embedding_dim}], hnsw(max_elements=100000))")
    print("Database initialized.")

def get_setting(conn: apsw.Connection, key: str, default: str = "") -> str:
    # Use try/except to handle case where no row is found, which is more robust
    try:
        return conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()[0]
    except (TypeError, IndexError):
        return default

def set_setting(conn: apsw.Connection, key: str, value: str):
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))

def list_projects(conn: apsw.Connection) -> List[str]:
    projects = [row[0] for row in conn.execute("SELECT name FROM projects ORDER BY name")]
    if not projects: add_project(conn, "Default"); return ["Default"]
    return projects

def add_project(conn: apsw.Connection, name: str): conn.execute("INSERT OR IGNORE INTO projects (name) VALUES (?)", (name,))
def get_project_id(conn: apsw.Connection, name: str) -> int:
    try:
        return conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()[0]
    except (TypeError, IndexError):
        return None

def save_message(conn: apsw.Connection, project_name: str, role: str, content: str, attachments: List[Dict] = None):
    project_id = get_project_id(conn, project_name) or add_project(conn, project_name) or get_project_id(conn, project_name)
    conn.execute("INSERT INTO messages (project_id, role, content, attachments) VALUES (?, ?, ?, ?)", (project_id, role, content, json.dumps(attachments or [])))

def load_messages(conn: apsw.Connection, project_name: str, limit: int = 100) -> List[Dict]:
    project_id = get_project_id(conn, project_name)
    if not project_id: return []
    rows = conn.execute("SELECT role, content, attachments, created_at FROM messages WHERE project_id = ? ORDER BY created_at ASC LIMIT ?", (project_id, limit))
    return [{"role": r[0], "content": r[1], "attachments": json.loads(r[2] or '[]'), "created_at": r[3]} for r in rows]

class ChatBackend(ABC):
    @abstractmethod
    def get_name(self) -> str: pass
    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]: pass
    @abstractmethod
    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]: pass
    def get_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> str: return "N/A"

class LMStudioBackend(ChatBackend):
    def __init__(self, base_url: str = LM_STUDIO_URL): self.base_url, self.client = base_url, openai.OpenAI(base_url=base_url, api_key="not-needed")
    def get_name(self) -> str: return "LM Studio"
    def test_connection(self) -> Tuple[bool, str]:
        try: return (True, f"Connected to {self.get_name()}") if requests.get(f"{self.base_url}/models", timeout=5).status_code == 200 else (False, f"Could not connect to {self.get_name()}")
        except Exception as e: return False, f"Error connecting to {self.get_name()}: {e}"
    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        try:
            if 'model' not in kwargs: kwargs['model'] = 'local-model'
            for chunk in self.client.chat.completions.create(messages=messages, **kwargs, stream=True):
                if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
        except Exception as e: yield f"Error: {e}"

class OllamaBackend(ChatBackend):
    def __init__(self, base_url: str = OLLAMA_URL): self.base_url, self.client = base_url, openai.OpenAI(base_url=base_url, api_key="ollama")
    def get_name(self) -> str: return "Ollama"
    def test_connection(self) -> Tuple[bool, str]:
        try: return (True, f"Connected to {self.get_name()}") if requests.get(self.base_url.replace("/v1", ""), timeout=5).status_code == 200 else (False, f"Could not connect to {self.get_name()}")
        except Exception as e: return False, f"Error connecting to {self.get_name()}: {e}"
    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        try:
            if 'model' not in kwargs: kwargs['model'] = 'local-model'
            for chunk in self.client.chat.completions.create(messages=messages, **kwargs, stream=True):
                if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
        except Exception as e: yield f"Error: {e}"

class OpenAIBackend(ChatBackend):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or SecretsManager.get_api_key("OpenAI")
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
    def get_name(self) -> str: return "OpenAI"
    def test_connection(self) -> Tuple[bool, str]:
        if not self.client: return False, "OpenAI API key not set."
        try: self.client.models.list(limit=1); return True, "Connected to OpenAI"
        except openai.AuthenticationError: return False, "OpenAI API key is invalid."
        except Exception as e: return False, f"Error connecting to OpenAI: {e}"
    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        if not self.client: yield "Error: OpenAI API key not set."; return
        try:
            for chunk in self.client.chat.completions.create(messages=messages, **kwargs, stream=True):
                if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
        except Exception as e: yield f"Error: {e}"
    def get_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> str:
        prices = {"gpt-4-turbo": {"input": 10.00, "output": 30.00}, "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}}
        model_prices = prices.get(model_name, prices["gpt-4-turbo"])
        cost = ((prompt_tokens / 1_000_000) * model_prices["input"]) + ((completion_tokens / 1_000_000) * model_prices["output"])
        return f"${cost:.6f}"

class AnthropicBackend(ChatBackend):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or SecretsManager.get_api_key("Anthropic")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
    def get_name(self) -> str: return "Anthropic"
    def test_connection(self) -> Tuple[bool, str]:
        if not self.client: return False, "Anthropic API key not set."
        try: self.client.messages.create(model="claude-3-haiku-20240307", messages=[{"role": "user", "content": "ping"}], max_tokens=10); return True, "Connected to Anthropic"
        except anthropic.AuthenticationError: return False, "Anthropic API key is invalid."
        except Exception as e: return False, f"Error connecting to Anthropic: {e}"
    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        if not self.client: yield "Error: Anthropic API key not set."; return
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), None)
        user_messages = [m for m in messages if m['role'] != 'system']
        try:
            with self.client.messages.stream(messages=user_messages, system=system_prompt, **kwargs) as stream:
                for text in stream.text_stream: yield text
        except Exception as e: yield f"Error: {e}"

class MistralBackend(ChatBackend):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or SecretsManager.get_api_key("Mistral")
        self.client = MistralClient(api_key=self.api_key) if self.api_key else None
    def get_name(self) -> str: return "Mistral"
    def test_connection(self) -> Tuple[bool, str]:
        if not self.client: return False, "Mistral API key not set."
        try: self.client.models.list(); return True, "Connected to Mistral AI"
        except Exception as e: return False, f"Error connecting to Mistral AI: {e}"
    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        if not self.client: yield "Error: Mistral API key not set."; return
        try:
            for chunk in self.client.chat_stream(messages=messages, **kwargs):
                if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
        except Exception as e: yield f"Error: {e}"

class GeminiBackend(ChatBackend):
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or SecretsManager.get_api_key("Gemini")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        else:
            self.client = None

    def get_name(self) -> str:
        return "Gemini"

    def test_connection(self) -> Tuple[bool, str]:
        if not self.client:
            return False, "Gemini API key not set."
        try:
            # Listing models is a lightweight way to test the key
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not models:
                 return False, "No text-generation models found."
            return True, "Connected to Gemini."
        except Exception as e:
            return False, f"Error connecting to Gemini: {e}"

    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        if not self.client:
            yield "Error: Gemini API key not set."
            return
        try:
            # Convert to Gemini's expected format
            gemini_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})

            # The 'model' kwarg is handled by the client being pre-configured
            # We remove it from kwargs if present to avoid conflicts
            kwargs.pop('model', None)

            response = self.client.generate_content(gemini_messages, stream=True, **kwargs)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error: {e}"


class BackendRouter:
    def __init__(self): self.backends: Dict[str, ChatBackend] = {}; self.active_backend_name: Optional[str] = None; self._discover_and_init_backends()
    def _discover_and_init_backends(self):
        self.backends.clear()
        for backend_class in [LMStudioBackend, OllamaBackend, OpenAIBackend, AnthropicBackend, MistralBackend, GeminiBackend]:
            try: self.backends[backend_class().get_name()] = backend_class()
            except Exception as e: print(f"Failed to init backend {backend_class.__name__}: {e}")
        if not self.active_backend_name and self.backends: self.set_active_backend(list(self.backends.keys())[0])
    def get_backend_names(self) -> List[str]: return list(self.backends.keys())
    def set_active_backend(self, name: str):
        if name in self.backends: self.active_backend_name = name
        else: raise ValueError(f"Backend '{name}' not found.")
    def get_active_backend(self) -> Optional[ChatBackend]: return self.backends.get(self.active_backend_name)

class RAGSystem:
    def __init__(self, conn: apsw.Connection): self.conn, self.model_name, self._model = conn, "sentence-transformers/all-MiniLM-L6-v2", None
    @property
    def model(self):
        if SentenceTransformer is None: raise RuntimeError("sentence-transformers not available")
        if self._model is None: self._model = SentenceTransformer(self.model_name)
        return self._model
    def add_document(self, source_name: str, content: str):
        if not content.strip(): return
        chunks = self._chunk_text(content); embeddings = self.model.encode(chunks, convert_to_numpy=True)
        with self.conn:
            for chunk, embedding in zip(chunks, embeddings):
                self.conn.execute("INSERT INTO rag_documents (source_name, content) VALUES (?, ?)", (source_name, chunk))
                doc_id = self.conn.last_insert_rowid()
                self.conn.execute("INSERT INTO rag_vectors (rowid, embedding) VALUES (?, ?)", (doc_id, embedding.tobytes()))

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split(); chunks, i = [], 0
        while i < len(words): chunks.append(" ".join(words[i:i + chunk_size])); i += max(1, chunk_size - overlap)
        return chunks
    def search(self, query: str, k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            sql = "SELECT t2.source_name, t2.content, t1.distance FROM rag_vectors AS t1 JOIN rag_documents AS t2 ON t1.rowid = t2.id WHERE knn_search(t1.embedding, knn_param(?, ?))"
            return [{"source": r[0], "content": r[1], "distance": r[2]} for r in self.conn.execute(sql, (query_embedding.tobytes(), k))]
        except Exception as e: print(f"RAG search error: {e}"); return []

def process_file(file_path: str) -> str:
    try:
        ext = Path(file_path).suffix.lower()
        if ext in {".txt", ".md", ".py", ".js", ".html", ".css"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
        elif ext == ".pdf" and fitz:
            with fitz.open(file_path) as doc: return "".join(page.get_text() for page in doc)
        elif ext == ".docx" and docx:
            return "\n".join(p.text for p in docx.Document(file_path).paragraphs)
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            if pytesseract:
                try:
                    text = pytesseract.image_to_string(Image.open(file_path))
                    return f"[OCR from Image: {Path(file_path).name}]\n{text}"
                except Exception as e:
                    return f"[Error during OCR on {Path(file_path).name}: {e}]"
            else:
                return f"[Image file: {Path(file_path).name} (pytesseract not installed)]"
        return f"[Unsupported file type: {ext}]"
    except Exception as e: return f"[Error processing file: {e}]"

class VoiceManager:
    def __init__(self):
        self.tts_engine, self.recognizer = None, None
        if pyttsx3:
            try: self.tts_engine = pyttsx3.init(); self.tts_engine.setProperty("rate", 150)
            except: pass
        if sr:
            try: self.recognizer = sr.Recognizer()
            except: pass
    def speak(self, text: str):
        if self.tts_engine:
            try: self.tts_engine.say(text); self.tts_engine.runAndWait()
            except: pass
    def listen(self) -> str:
        if not self.recognizer: return ""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return self.recognizer.recognize_google(audio)
        except: return ""

class ScreenshotManager:
    @staticmethod
    def capture_fullscreen() -> str:
        try:
            img = ImageGrab.grab()
            filepath = ROOT_DIR / f"screenshot_{int(time.time())}.png"
            img.save(filepath)
            return str(filepath)
        except Exception as e: print(f"Screenshot error: {e}"); return ""

# --- GUI LAYER (ONLY LOADED WHEN NOT HEADLESS) ---

def run_gui():
    """Contains all GUI-related code and is only called when not in headless mode."""
    # Lazy imports for GUI libraries
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from tkinter import font as tkFont
    from PIL import ImageTk
    from pynput import keyboard

    class HotkeyManager(threading.Thread):
        def __init__(self, hotkey, on_activate):
            super().__init__(daemon=True)
            self.hotkey = hotkey
            self.on_activate = on_activate
        def run(self):
            with keyboard.GlobalHotKeys({self.hotkey: self.on_activate}) as h:
                h.join()

    class OmniMindStudio:
        def __init__(self):
            self.root = tk.Tk(); self.root.title(f"{APP_NAME} v{APP_VERSION}"); self.root.geometry("1200x800"); self.root.minsize(1000, 600)
            self.shutdown_event = threading.Event()
            self.setup_styles()
            self.db_conn = create_db_connection(DB_FILE)
            init_database(self.db_conn)
            self.backend_router = BackendRouter(); self.rag_system = RAGSystem(self.db_conn); self.voice_manager = VoiceManager(); self.screenshot_manager = ScreenshotManager()
            self.hotkey_manager = HotkeyManager('<ctrl>+<alt>+<space>', self.toggle_main_window_visibility)
            self.current_project = tk.StringVar(value="Default"); self.active_backend = tk.StringVar(value=self.backend_router.active_backend_name); self.temperature = tk.DoubleVar(value=0.7); self.max_tokens = tk.IntVar(value=2000); self.rag_enabled = tk.BooleanVar(value=True)
            self.attachments, self.messages = [], []
            self.last_rag_results = []
            
            self.latency_var = tk.StringVar(value="Latency: N/A"); self.tokens_var = tk.StringVar(value="Tokens: N/A"); self.cost_var = tk.StringVar(value="Cost: N/A")
            try: self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except: self.tokenizer = None
            
            self.load_settings(); self.build_ui(); self.load_project_data()
            self.root.after(100, self.run_startup_health_checks) # Run after main window is drawn
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        def run_startup_health_checks(self):
            """Runs a series of checks to ensure the app is ready."""
            errors = []
            # 1. Test DB Connection
            try:
                self.db_conn.execute("SELECT 1 FROM projects LIMIT 1")
            except Exception as e:
                errors.append(f"Database connection failed: {e}")

            # 2. Test active backend connection
            active_backend = self.backend_router.get_active_backend()
            if active_backend:
                success, message = active_backend.test_connection()
                if not success:
                    errors.append(f"Could not connect to '{active_backend.get_name()}': {message}")
            else:
                errors.append("No active backend selected or initialized.")

            if errors:
                error_message = "OmniMind Studio encountered problems on startup:\n\n" + "\n- ".join(errors)
                error_message += "\n\nPlease check your settings and connections."
                messagebox.showerror("Startup Diagnostics Failed", error_message)
            else:
                self.update_status("All systems nominal.")

        def on_closing(self): self.db_conn.close(); self.root.destroy()

        def toggle_main_window_visibility(self):
            if self.root.state() == 'withdrawn' or self.root.winfo_viewable() == 0:
                self.root.deiconify(); self.root.lift(); self.root.focus_force()
            else:
                self.root.withdraw()

        def setup_styles(self):
            self.BG_COLOR = '#2b2b2b'; self.FG_COLOR = '#dcdcdc'; self.WIDGET_BG = '#3c3c3c'; self.ACCENT_COLOR = '#007acc'; self.TEXT_FONT = ("Segoe UI", 10); self.BOLD_FONT = ("Segoe UI", 10, "bold")
            self.root.configure(bg=self.BG_COLOR)
            style = ttk.Style(self.root); style.theme_use('clam')
            style.configure('.', background=self.BG_COLOR, foreground=self.FG_COLOR, font=self.TEXT_FONT, borderwidth=0, lightcolor=self.BG_COLOR, darkcolor=self.BG_COLOR)
            style.configure('TFrame', background=self.BG_COLOR)
            style.configure('TLabel', background=self.BG_COLOR, foreground=self.FG_COLOR)
            style.configure('TCheckbutton', background=self.BG_COLOR, foreground=self.FG_COLOR, indicatorbackground=self.WIDGET_BG, indicatorforeground=self.FG_COLOR)
            style.map('TCheckbutton', background=[('active', self.BG_COLOR)], indicatorbackground=[('selected', self.ACCENT_COLOR)])
            style.configure('TButton', background=self.WIDGET_BG, foreground=self.FG_COLOR, borderwidth=1, focusthickness=0, focuscolor='none')
            style.map('TButton', background=[('active', self.ACCENT_COLOR), ('pressed', self.ACCENT_COLOR)])
            style.configure('TEntry', fieldbackground=self.WIDGET_BG, foreground=self.FG_COLOR, borderwidth=1, insertcolor=self.FG_COLOR)
            style.configure('TCombobox', fieldbackground=self.WIDGET_BG, foreground=self.FG_COLOR, arrowcolor=self.FG_COLOR, borderwidth=1, selectbackground=self.WIDGET_BG, selectforeground=self.FG_COLOR)
            self.root.option_add('*TCombobox*Listbox.background', self.WIDGET_BG); self.root.option_add('*TCombobox*Listbox.foreground', self.FG_COLOR)
            style.configure('Vertical.TScrollbar', background=self.WIDGET_BG, troughcolor=self.BG_COLOR, bordercolor=self.BG_COLOR, arrowcolor=self.FG_COLOR)
            style.map('Vertical.TScrollbar', background=[('active', self.ACCENT_COLOR)])

        def get_db_setting(self, key, default=""): return get_setting(self.db_conn, key, default)
        def set_db_setting(self, key, value): set_setting(self.db_conn, key, value)
        
        def load_settings(self):
            self.temperature.set(float(self.get_db_setting("temperature", "0.7"))); self.max_tokens.set(int(self.get_db_setting("max_tokens", "2000"))); self.rag_enabled.set(self.get_db_setting("rag_enabled", "true").lower() == "true")
            if saved_backend := self.get_db_setting("active_backend"):
                if saved_backend in self.backend_router.get_backend_names(): self.active_backend.set(saved_backend); self.backend_router.set_active_backend(saved_backend)

        def save_settings(self): self.set_db_setting("temperature", str(self.temperature.get())); self.set_db_setting("max_tokens", str(self.max_tokens.get())); self.set_db_setting("rag_enabled", str(self.rag_enabled.get()).lower()); self.set_db_setting("active_backend", self.active_backend.get())
        
        def build_ui(self): self.root.grid_rowconfigure(1, weight=1); self.root.grid_columnconfigure(0, weight=1); self.build_toolbar(); self.build_chat_area(); self.build_input_area(); self.build_hud(); self.build_status_bar()
        
        def build_toolbar(self):
            toolbar = ttk.Frame(self.root, padding=(10, 5)); toolbar.grid(row=0, column=0, sticky="ew")
            ttk.Label(toolbar, text="Project:").pack(side="left"); self.project_combo = ttk.Combobox(toolbar, textvariable=self.current_project, values=list_projects(self.db_conn), width=20); self.project_combo.pack(side="left", padx=(5, 10)); self.project_combo.bind("<<ComboboxSelected>>", self.on_project_change)
            ttk.Button(toolbar, text="New", command=self.new_project).pack(side="left")
            ttk.Label(toolbar, text="Backend:", padding=(10,0)).pack(side="left"); self.backend_combo = ttk.Combobox(toolbar, textvariable=self.active_backend, values=self.backend_router.get_backend_names(), width=15); self.backend_combo.pack(side="left", padx=5); self.backend_combo.bind("<<ComboboxSelected>>", self.on_backend_change)
            ttk.Checkbutton(toolbar, text="RAG", variable=self.rag_enabled, command=self.toggle_rag).pack(side="left", padx=5)
            ttk.Button(toolbar, text="Index Folder", command=self.index_folder).pack(side="left", padx=5)
            ttk.Button(toolbar, text="Prompts", command=self.show_prompt_studio).pack(side="left", padx=5)
            ttk.Button(toolbar, text="Compare", command=self.show_compare_window).pack(side="right")
            ttk.Button(toolbar, text="Export", command=self.export_chat).pack(side="right", padx=5)
            ttk.Button(toolbar, text="Settings", command=self.show_settings).pack(side="right")
        
        def build_chat_area(self):
            chat_frame = ttk.Frame(self.root); chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,5)); chat_frame.grid_rowconfigure(0, weight=1); chat_frame.grid_columnconfigure(0, weight=1)
            self.chat_text = tk.Text(chat_frame, wrap="word", bg=self.WIDGET_BG, fg=self.FG_COLOR, insertbackground=self.FG_COLOR, font=self.TEXT_FONT, borderwidth=0, highlightthickness=0, relief="flat", state=tk.DISABLED, padding=(5,5))
            self.chat_text.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_text.yview); scrollbar.grid(row=0, column=1, sticky="ns"); self.chat_text.configure(yscrollcommand=scrollbar.set)
        
        def build_input_area(self):
            input_frame = ttk.Frame(self.root, padding=(10, 5)); input_frame.grid(row=2, column=0, sticky="ew"); input_frame.grid_columnconfigure(0, weight=1)
            self.input_entry = ttk.Entry(input_frame, font=self.TEXT_FONT); self.input_entry.grid(row=0, column=0, sticky="ew", ipady=5); self.input_entry.bind("<Return>", self.send_message)
            for i, (text, cmd) in enumerate([("Attach", self.attach_file), ("Voice", self.voice_input), ("Screenshot", self.take_screenshot), ("Send", self.send_message), ("Clear", self.clear_chat)]):
                ttk.Button(input_frame, text=text, command=cmd).grid(row=0, column=i + 1, padx=(5,0))
        
        def build_hud(self):
            hud_frame = ttk.Frame(self.root, padding=(10, 2)); hud_frame.grid(row=3, column=0, sticky="ew")
            ttk.Label(hud_frame, textvariable=self.latency_var).pack(side="left")
            ttk.Label(hud_frame, textvariable=self.tokens_var, padding=(20, 0)).pack(side="left")
            ttk.Label(hud_frame, textvariable=self.cost_var, padding=(20, 0)).pack(side="left")

        def build_status_bar(self):
            self.status_label = ttk.Label(self.root, text="Ready", anchor="w", padding=5); self.status_label.grid(row=4, column=0, sticky="ew", padx=10, pady=(0,5))
        
        def update_status(self, message: str): self.status_label.config(text=message); self.root.update_idletasks()
        def on_backend_change(self, event=None): self.backend_router.set_active_backend(self.active_backend.get()); self.save_settings(); self.test_connection()
        def test_connection(self):
            active_backend = self.backend_router.get_active_backend()
            if active_backend: self.update_status(f"Testing connection to {active_backend.get_name()}..."); success, message = active_backend.test_connection(); self.update_status(message)
            else: self.update_status("No active backend selected.")
        def load_project_data(self): self.messages = load_messages(self.db_conn, self.current_project.get()); self.display_messages()
        def on_project_change(self, event=None): self.load_project_data()
        def new_project(self):
            name = simpledialog.askstring("New Project", "Enter project name:")
            if name: add_project(self.db_conn, name); self.project_combo.configure(values=list_projects(self.db_conn)); self.current_project.set(name); self.clear_chat()
        def toggle_rag(self): self.save_settings(); self.update_status(f"RAG {'enabled' if self.rag_enabled.get() else 'disabled'}")
        def index_folder(self):
            folder = filedialog.askdirectory(title="Select folder to index")
            if not folder: return
            self.update_status("Indexing folder...")
            def index_worker():
                try:
                    for file_path in Path(folder).rglob("*"):
                        if self.shutdown_event.is_set():
                            logging.info("Shutdown event set, stopping indexing.")
                            break
                        if file_path.is_file():
                            self.update_status(f"Indexing: {file_path.name}")
                            content = process_file(str(file_path))
                            if content:
                                self.rag_system.add_document(file_path.name, content)
                    if not self.shutdown_event.is_set():
                        self.update_status("Folder indexed successfully")
                except Exception as e:
                    logging.error(f"Indexing failed: {e}", exc_info=True)
                    self.update_status(f"Indexing failed: {e}")
            threading.Thread(target=index_worker, name="IndexerThread", daemon=True).start()
        def show_settings(self):
            win = tk.Toplevel(self.root); win.title("Settings"); win.geometry("400x250"); win.transient(self.root); win.configure(bg=self.BG_COLOR)
            frame = ttk.Frame(win, padding=10); frame.pack(fill="both", expand=True)
            ttk.Label(frame, text="Temperature:").grid(row=0, column=0, sticky="w"); ttk.Scale(frame, from_=0.0, to=1.0, variable=self.temperature, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=5)
            
            ttk.Label(frame, text="Max Tokens:").grid(row=1, column=0, sticky="w", pady=10)
            vcmd = (self.root.register(lambda P: P.isdigit() or P == ""), '%P')
            tokens_entry = ttk.Entry(frame, validate='key', validatecommand=vcmd)
            tokens_entry.insert(0, str(self.max_tokens.get()))
            tokens_entry.grid(row=1, column=1, sticky="ew", padx=5)

            ttk.Button(frame, text="Manage API Keys", command=self.show_api_key_manager).grid(row=2, column=0, columnspan=2, pady=15)
            def save():
                try:
                    max_tokens_val = tokens_entry.get()
                    if max_tokens_val:
                        self.max_tokens.set(int(max_tokens_val))
                    else:
                        # or set to a default if empty
                        self.max_tokens.set(2000) 
                    self.save_settings()
                    win.destroy()
                    self.update_status("Settings saved")
                except ValueError:
                    messagebox.showerror("Error", "Invalid max tokens value. Please enter a number.", parent=win)
            ttk.Button(frame, text="Save & Close", command=save).grid(row=3, column=0, columnspan=2, pady=10)
        def show_api_key_manager(self):
            win = tk.Toplevel(self.root); win.title("API Key Management"); win.geometry("600x250"); win.transient(self.root); win.configure(bg=self.BG_COLOR)
            services = ["OpenAI", "Anthropic", "Mistral", "Gemini"]; entries = {}
            frame = ttk.Frame(win, padding=10); frame.pack(fill="both", expand=True)

            # A map to associate service names with their backend classes
            backend_map = {
                "OpenAI": OpenAIBackend,
                "Anthropic": AnthropicBackend,
                "Mistral": MistralBackend,
                "Gemini": GeminiBackend
            }

            def test_key(service_name, key_entry):
                api_key = key_entry.get()
                if not api_key or api_key == "**********":
                    messagebox.showwarning("Input Error", "Please enter an API key to test.", parent=win)
                    return
                
                backend_class = backend_map.get(service_name)
                if not backend_class:
                    messagebox.showerror("Error", f"Could not find backend for {service_name}", parent=win)
                    return

                # Create a temporary instance of the backend with the key from the entry field
                # This is now clean and generic thanks to the standardized __init__ methods.
                temp_backend = backend_class(api_key=api_key)
                
                success, message = temp_backend.test_connection()
                if success:
                    messagebox.showinfo("Connection Test", f"Successfully connected to {service_name}!", parent=win)
                else:
                    messagebox.showerror("Connection Test", f"Failed to connect to {service_name}:\n{message}", parent=win)

            for i, service in enumerate(services):
                ttk.Label(frame, text=f"{service} API Key:").grid(row=i, column=0, sticky="w", padx=5, pady=5)
                entry = ttk.Entry(frame, width=40, show="*")
                if SecretsManager.get_api_key(service):
                    entry.insert(0, "**********")
                entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
                entries[service] = entry
            
                test_button = ttk.Button(frame, text="Test", command=lambda s=service, e=entry: test_key(s, e))
                test_button.grid(row=i, column=2, padx=(5,0))

            def save_keys():
                keys_changed = False
                for service, entry in entries.items():
                    key = entry.get()
                    if key and key != "**********":
                        SecretsManager.set_api_key(service, key)
                        keys_changed = True
                
                if keys_changed:
                    messagebox.showinfo("API Keys", "API keys saved securely.", parent=win)
                    # Re-discover backends only if keys have actually changed
                    self.backend_router._discover_and_init_backends()
                    self.backend_combo.configure(values=self.backend_router.get_backend_names())
                    self.test_connection()
                win.destroy()
        
            save_button = ttk.Button(frame, text="Save & Close", command=save_keys)
            save_button.grid(row=len(services), column=0, columnspan=3, pady=20)
        
        def show_prompt_studio(self):
            win = tk.Toplevel(self.root); win.title("Prompt Studio"); win.geometry("800x600"); win.transient(self.root); win.configure(bg=self.BG_COLOR)
            main_frame = ttk.Frame(win, padding=10); main_frame.pack(fill="both", expand=True); main_frame.grid_columnconfigure(1, weight=1); main_frame.grid_rowconfigure(0, weight=1)
            
            left_pane = ttk.Frame(main_frame); left_pane.grid(row=0, column=0, sticky="ns", padx=(0, 10))
            listbox = tk.Listbox(left_pane, bg=self.WIDGET_BG, fg=self.FG_COLOR, highlightthickness=0, borderwidth=0); listbox.pack(side="left", fill="y", expand=True)
            
            right_pane = ttk.Frame(main_frame); right_pane.grid(row=0, column=1, sticky="nsew")
            right_pane.grid_rowconfigure(1, weight=1); right_pane.grid_columnconfigure(0, weight=1)
            ttk.Label(right_pane, text="Template Name:").grid(row=0, column=0, sticky="w")
            name_entry = ttk.Entry(right_pane); name_entry.grid(row=0, column=1, sticky="ew", padx=5)
            text_editor = tk.Text(right_pane, wrap="word", bg=self.WIDGET_BG, fg=self.FG_COLOR, font=self.TEXT_FONT); text_editor.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)

            def load_templates():
                listbox.delete(0, tk.END)
                for row in self.db_conn.execute("SELECT name FROM prompt_templates ORDER BY name"):
                    listbox.insert(tk.END, row[0])

            def on_select(evt):
                if not listbox.curselection(): return
                name = listbox.get(listbox.curselection())
                try:
                    template = self.db_conn.execute("SELECT template FROM prompt_templates WHERE name = ?", (name,)).fetchone()[0]
                    name_entry.delete(0, tk.END)
                    name_entry.insert(0, name)
                    text_editor.delete("1.0", tk.END)
                    text_editor.insert("1.0", template)
                except (TypeError, IndexError):
                    pass # Template not found
            listbox.bind('<<ListboxSelect>>', on_select)

            def save_template():
                name, template = name_entry.get().strip(), text_editor.get("1.0", tk.END).strip()
                if not name or not template: messagebox.showerror("Error", "Name and template cannot be empty.", parent=win); return
                self.db_conn.execute("INSERT OR REPLACE INTO prompt_templates (name, template) VALUES (?, ?)", (name, template))
                load_templates()

            def delete_template():
                if not listbox.curselection(): return
                name = listbox.get(listbox.curselection())
                if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name}'?", parent=win):
                    self.db_conn.execute("DELETE FROM prompt_templates WHERE name = ?", (name,))
                    name_entry.delete(0, tk.END)
                    text_editor.delete("1.0", tk.END)
                    load_templates()

            def use_template():
                if not listbox.curselection(): return
                name = listbox.get(listbox.curselection())
                try:
                    template = self.db_conn.execute("SELECT template FROM prompt_templates WHERE name = ?", (name,)).fetchone()[0]
                    self.input_entry.delete(0, tk.END)
                    self.input_entry.insert(0, template)
                    win.destroy()
                except (TypeError, IndexError):
                    pass # Template not found

            button_frame = ttk.Frame(right_pane); button_frame.grid(row=2, column=0, columnspan=2, pady=5)
            for text, cmd in [("Save", save_template), ("Delete", delete_template), ("New", lambda: (name_entry.delete(0, tk.END), text_editor.delete("1.0", tk.END)))]:
                ttk.Button(button_frame, text=text, command=cmd).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Use in Chat", command=use_template).pack(side="right", padx=5)
            
            load_templates()
        
        def show_compare_window(self):
            win = tk.Toplevel(self.root); win.title("Compare & Diff Models"); win.geometry("1200x800"); win.transient(self.root); win.configure(bg=self.BG_COLOR)
            main_frame = ttk.Frame(win, padding=10); main_frame.pack(fill="both", expand=True); main_frame.grid_rowconfigure(2, weight=1); main_frame.grid_columnconfigure(0, weight=1); main_frame.grid_columnconfigure(1, weight=1)
            controls_frame = ttk.Frame(main_frame); controls_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5); backend_names = self.backend_router.get_backend_names()
            ttk.Label(controls_frame, text="Backend 1:").pack(side="left", padx=(0,5)); combo1 = ttk.Combobox(controls_frame, values=backend_names, width=15); combo1.pack(side="left", padx=(0, 20)); combo1.set(backend_names[0] if backend_names else "")
            ttk.Label(controls_frame, text="Backend 2:").pack(side="left", padx=(0,5)); combo2 = ttk.Combobox(controls_frame, values=backend_names, width=15); combo2.pack(side="left"); combo2.set(backend_names[1] if len(backend_names) > 1 else "")
            prompt_frame = ttk.Frame(main_frame); prompt_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5); prompt_frame.grid_columnconfigure(0, weight=1)
            ttk.Label(prompt_frame, text="Prompt:").pack(side="left", padx=(0,5)); prompt_entry = ttk.Entry(prompt_frame); prompt_entry.pack(side="left", fill="x", expand=True)
            text_widget1 = tk.Text(main_frame, wrap="word", bg=self.WIDGET_BG, fg=self.FG_COLOR, font=self.TEXT_FONT, state=tk.DISABLED); text_widget1.grid(row=2, column=0, sticky="nsew", padx=(0,5))
            text_widget2 = tk.Text(main_frame, wrap="word", bg=self.WIDGET_BG, fg=self.FG_COLOR, font=self.TEXT_FONT, state=tk.DISABLED); text_widget2.grid(row=2, column=1, sticky="nsew", padx=(5,0))
            diff_frame = ttk.Frame(main_frame, padding=(0,10,0,0)); diff_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5); diff_frame.grid_rowconfigure(0, weight=1); diff_frame.grid_columnconfigure(0, weight=1)
            diff_widget = tk.Text(diff_frame, wrap="word", bg=self.WIDGET_BG, fg=self.FG_COLOR, font=self.TEXT_FONT, state=tk.DISABLED); diff_widget.pack(fill="both", expand=True)
            diff_widget.tag_configure("addition", foreground="lightgreen"); diff_widget.tag_configure("deletion", foreground="#ff8888")
            def run_comparison_thread():
                b1_name, b2_name, prompt = combo1.get(), combo2.get(), prompt_entry.get()
                if not all([b1_name, b2_name, prompt]): messagebox.showerror("Input Error", "Please select two backends and enter a prompt.", parent=win); return
                for w in [text_widget1, text_widget2, diff_widget]: w.config(state=tk.NORMAL); w.delete("1.0", tk.END); w.config(state=tk.DISABLED)
                b1, b2 = self.backend_router.backends.get(b1_name), self.backend_router.backends.get(b2_name)
                q1, q2 = queue.Queue(), queue.Queue()
                def target(b, q):
                    full_response = "".join(b.stream_chat([{"role": "user", "content": prompt}])); q.put(full_response)
                threading.Thread(target=target, args=(b1, q1), daemon=True).start(); threading.Thread(target=target, args=(b2, q2), daemon=True).start()
                full_resp1, full_resp2 = q1.get(), q2.get()
                for w, t in [(text_widget1, full_resp1), (text_widget2, full_resp2)]: w.config(state=tk.NORMAL); w.insert("1.0", t); w.config(state=tk.DISABLED)
                self.display_diff(full_resp1, full_resp2, diff_widget)
            run_button = ttk.Button(controls_frame, text="Run", command=run_comparison_thread); run_button.pack(side="left", padx=10)

        def display_diff(self, text1, text2, diff_widget):
            diff_widget.config(state=tk.NORMAL); diff_widget.delete("1.0", tk.END)
            diff = list(difflib.Differ().compare(text1.splitlines(keepends=True), text2.splitlines(keepends=True)))
            for line in diff:
                if line.startswith('+ '): diff_widget.insert(tk.END, line, 'addition')
                elif line.startswith('- '): diff_widget.insert(tk.END, line, 'deletion')
                elif not line.startswith('? '): diff_widget.insert(tk.END, line)
            diff_widget.config(state=tk.DISABLED)

        def export_chat(self):
            project = self.current_project.get()
            if not (filepath := filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")], initialfilename=f"{project.replace(' ', '_')}.md")): return
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {project} Chat Export\n\n"); [f.write(f"## {'You' if m['role'] == 'user' else 'Assistant'}\n{m['content']}\n\n") for m in self.messages]
                self.update_status(f"Exported to {Path(filepath).name}")
            except Exception as e: messagebox.showerror("Export Error", f"Failed to export: {e}")
        def attach_file(self):
            if files := filedialog.askopenfilenames(title="Select files"): self.attachments.extend(files); self.update_status(f"Attached {len(files)} file(s)")
        def voice_input(self):
            self.update_status("Listening...");
            def voice_worker():
                if text := self.voice_manager.listen(): self.input_entry.delete(0, tk.END); self.input_entry.insert(0, text); self.update_status("Voice input received")
                else: self.update_status("Voice input failed")
            threading.Thread(target=voice_worker, daemon=True).start()
        def take_screenshot(self):
            if filename := ScreenshotManager.capture_fullscreen(): self.attachments.append(filename); self.update_status("Screenshot attached")
            else: self.update_status("Screenshot failed")
        def clear_chat(self): self.chat_text.config(state=tk.NORMAL); self.chat_text.delete(1.0, tk.END); self.chat_text.config(state=tk.DISABLED); self.attachments.clear(); self.update_status("Chat cleared")
        def display_messages(self):
            self.chat_text.config(state=tk.NORMAL); self.chat_text.delete(1.0, tk.END)
            for msg in self.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                self.chat_text.insert(tk.END, f"{role}: ", ("role_user" if role == "You" else "role_assistant")); self.chat_text.insert(tk.END, f"{msg['content']}\n\n")
            self.chat_text.tag_configure("role_user", font=self.BOLD_FONT, foreground="#90ee90"); self.chat_text.tag_configure("role_assistant", font=self.BOLD_FONT, foreground="#add8e6")
            self.chat_text.config(state=tk.DISABLED); self.chat_text.see(tk.END)
        
        def handle_command(self, command_text: str):
            parts = command_text.strip().split()
            command = parts[0].lower()
            args = parts[1:]
            
            command_map = {
                "/models": self.handle_command_models,
                "/bench": self.handle_command_bench,
                "/citations": self.handle_command_doc_citations,
                "/memory": self.handle_command_memory,
                "/vision.ask": self.handle_command_vision_ask,
                "/export": self.handle_command_export,
            }
            
            if command in command_map:
                command_map[command](args)
            else:
                self.add_system_message(f"Unknown command: {command}")

        def handle_command_memory(self, args):
            if not args:
                self.add_system_message("Usage: /memory [add|list|clear] [text...]")
                return
            
            sub_command = args[0].lower()
            if sub_command == "add":
                text_to_add = " ".join(args[1:])
                if not text_to_add:
                    self.add_system_message("Usage: /memory add [text to remember]")
                    return
                self.db_conn.execute("INSERT INTO memories (content) VALUES (?)", (text_to_add,))
                self.add_system_message(f"Memory added: '{text_to_add}'")
            elif sub_command == "list":
                memories = self.db_conn.execute("SELECT id, content FROM memories ORDER BY created_at DESC LIMIT 10").fetchall()
                if not memories:
                    self.add_system_message("No memories found.")
                    return
                message = "Recent memories:\n" + "\n".join(f"- (ID: {m[0]}) {m[1]}" for m in memories)
                self.add_system_message(message)
            elif sub_command == "clear":
                self.db_conn.execute("DELETE FROM memories")
                self.add_system_message("All memories cleared.")
            else:
                self.add_system_message(f"Unknown memory command: {sub_command}")

        def handle_command_vision_ask(self, args):
            if not self.attachments:
                self.add_system_message("Please attach an image first to use /vision.ask")
                return
            
            question = " ".join(args)
            if not question:
                self.add_system_message("Usage: /vision.ask [question about the attached image]")
                return
            
            # Use the last attached file for the vision query
            last_attachment = self.attachments[-1]
            self.add_system_message(f"Asking vision model about '{Path(last_attachment).name}'...")
            
            # Prepend the question to the user message for the next send
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, f"[Vision Question: {question}]")
            self.send_message()

        def handle_command_export(self, args):
            if not args or args[0].lower() not in ["md", "json"]:
                self.add_system_message("Usage: /export [md|json]")
                return
            
            format_type = args[0].lower()
            project = self.current_project.get()
            initial_filename = f"{project.replace(' ', '_')}.{format_type}"
            
            if format_type == "md":
                filepath = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")], initialfilename=initial_filename)
                if not filepath: return
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {project} Chat Export\n\n")
                        for m in self.messages:
                            f.write(f"## {'You' if m['role'] == 'user' else 'Assistant'}\n{m['content']}\n\n")
                    self.update_status(f"Exported to {Path(filepath).name}")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to export: {e}")
            
            elif format_type == "json":
                filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfilename=initial_filename)
                if not filepath: return
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(self.messages, f, indent=2)
                    self.update_status(f"Exported to {Path(filepath).name}")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to export: {e}")

        def add_system_message(self, message: str):
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.insert(tk.END, f"System: {message}\n\n", "role_system")
            self.chat_text.tag_configure("role_system", font=self.BOLD_FONT, foreground="orange")
            self.chat_text.config(state=tk.DISABLED)
            self.chat_text.see(tk.END)

        def handle_command_models(self, args):
            backend_names = self.backend_router.get_backend_names()
            message = "Available backends:\n- " + "\n- ".join(backend_names)
            self.add_system_message(message)

        def handle_command_bench(self, args):
            active_backend = self.backend_router.get_active_backend()
            if not active_backend:
                self.add_system_message("No active backend selected.")
                return

            def bench_worker():
                self.add_system_message(f"Running benchmark on {active_backend.get_name()}...")
                start_time = time.monotonic()
                try:
                    # A simple, non-streaming test
                    response = "".join(list(active_backend.stream_chat([{"role": "user", "content": "Hello!"}], max_tokens=5)))
                    duration = time.monotonic() - start_time
                    if "Error:" in response:
                        self.add_system_message(f"Benchmark failed: {response}")
                    else:
                        self.add_system_message(f"Benchmark complete. Response time: {duration:.2f}s")
                except Exception as e:
                    self.add_system_message(f"Benchmark failed with an error: {e}")
            
            threading.Thread(target=bench_worker, daemon=True).start()

        def handle_command_doc_citations(self, args):
            if not hasattr(self, 'last_rag_results') or not self.last_rag_results:
                self.add_system_message("No document citations found. Please ask a question that uses the RAG system first.")
                return
            
            message = "Citations from last RAG query:\n"
            for i, res in enumerate(self.last_rag_results):
                message += f"{i+1}. Source: {res['source']} (Distance: {res['distance']:.4f})\n"
            self.add_system_message(message)

        def send_message(self, event=None):
            text = self.input_entry.get().strip()
            if not text and not self.attachments: return
            
            if text.startswith("/"):
                self.handle_command(text)
                self.input_entry.delete(0, tk.END)
                return

            self.input_entry.delete(0, tk.END)
            project = self.current_project.get()
            save_message(self.db_conn, project, "user", text, [{"path": p} for p in self.attachments]); self.messages.append({"role": "user", "content": text}); self.display_messages()
            attachment_text = "\n\nAttachments:\n" + "\n".join(f"File: {Path(p).name}\n{process_file(p)[:500]}..." for p in self.attachments) if self.attachments else ""
            
            rag_context = ""
            self.last_rag_results = [] # Clear previous results
            if self.rag_enabled.get() and text:
                try:
                    if results := self.rag_system.search(text):
                        self.last_rag_results = results
                        rag_context = "\n\nRelevant context:\n" + "\n".join(f"Source: {res['source']}\n{res['content'][:200]}..." for res in results[:3])
                except Exception as e: print(f"RAG error: {e}")
            full_message = text + attachment_text + rag_context
            self.attachments.clear()
            active_backend = self.backend_router.get_active_backend()
            if not active_backend: self.update_status("Error: No backend selected."); return
            self.update_status(f"Generating response from {active_backend.get_name()}...")
            api_messages = [{"role": m["role"], "content": m["content"]} for m in self.messages[-10:] if m['role'] != 'system']; api_messages.append({"role": "user", "content": full_message})
            
            self.chat_text.config(state=tk.NORMAL); self.chat_text.insert(tk.END, "Assistant: ", "role_assistant"); self.chat_text.config(state=tk.DISABLED)
            
            def stream_response():
                start_time = time.monotonic()
                prompt_tokens, completion_tokens, response_text = 0, 0, ""
                if self.tokenizer:
                    try:
                        prompt_tokens = len(self.tokenizer.encode(full_message))
                    except Exception as e:
                        logging.warning(f"Could not encode prompt for token counting: {e}")

                try:
                    kwargs = {"model": "gpt-4-turbo", "max_tokens": self.max_tokens.get(), "temperature": self.temperature.get()}
                    if isinstance(active_backend, OllamaBackend): kwargs['model'] = 'llama3'
                    elif isinstance(active_backend, AnthropicBackend): kwargs['model'] = 'claude-3-opus-20240229'
                    elif isinstance(active_backend, MistralBackend): kwargs['model'] = 'mistral-large-latest'
                    # Gemini model is set in its __init__
                    
                    for chunk in active_backend.stream_chat(api_messages, **kwargs):
                        if self.shutdown_event.is_set():
                            logging.info("Shutdown event set, stopping chat stream.")
                            break
                        response_text += chunk
                        self.chat_text.config(state=tk.NORMAL); self.chat_text.insert(tk.END, chunk); self.chat_text.see(tk.END); self.chat_text.config(state=tk.DISABLED); self.root.update_idletasks()
                    
                    if self.shutdown_event.is_set():
                        return

                    latency = time.monotonic() - start_time
                    if self.tokenizer:
                        try:
                            completion_tokens = len(self.tokenizer.encode(response_text))
                        except Exception as e:
                            logging.warning(f"Could not encode response for token counting: {e}")
                    
                    total_tokens = prompt_tokens + completion_tokens
                    
                    self.latency_var.set(f"Latency: {latency:.2f}s")
                    self.tokens_var.set(f"Tokens: {total_tokens} (P: {prompt_tokens}, C: {completion_tokens})")
                    self.cost_var.set(f"Cost: {active_backend.get_cost(kwargs.get('model', 'default'), prompt_tokens, completion_tokens)}")

                    save_message(self.db_conn, project, "assistant", response_text)
                    self.messages.append({"role": "assistant", "content": response_text})
                    self.chat_text.config(state=tk.NORMAL); self.chat_text.insert(tk.END, "\n\n"); self.chat_text.config(state=tk.DISABLED)
                    self.update_status("Ready")
                except Exception as e:
                    logging.error(f"Error during chat stream: {e}", exc_info=True)
                    error_text = f"\n\n--- An error occurred: {e} ---"
                    self.chat_text.config(state=tk.NORMAL); self.chat_text.insert(tk.END, error_text); self.chat_text.config(state=tk.DISABLED)
                    self.update_status(f"Error: {e}")
            threading.Thread(target=stream_response, name="ChatStreamThread", daemon=True).start()
        def run(self): self.hotkey_manager.start(); self.root.mainloop()

    app = OmniMindStudio()
    app.run()

# --- MAIN EXECUTION BLOCK ---

def main():
    """Headless entry point for tasks like testing or CLI operations."""
    print("Running in headless mode. No GUI will be started.")
    # In a real application, this could run CLI tasks, etc.
    pass

if __name__ == "__main__":
    setup_logging()
    if HEADLESS:
        main()
    else:
        run_gui()
