# OmniMind Studio

*The Indie-Pro AI Desktop That Beats Claude for Windows*

---

## Overview

OmniMind Studio is a **one-pager Python desktop app** that integrates local and commercial LLMs, advanced RAG, voice and screenshots, and project management into a clean Windows application. It is self-bootstrapping (installs its own dependencies), works offline with LM Studio, and scales to cloud APIs when needed.

This repository contains:

* **`claudex_pro.py`** → the main app (Tk desktop, RAG, voice, screenshots, projects).
* **`claudex_ultra.db`** → the official SQLite database (projects, messages, settings, indexed docs).

---

## Key Features (Already Implemented)

### 🔌 Local Backend Integration

* **LM Studio backend** with streaming responses
* Self-contained client class (`LMStudioClient`) with connection testing

### 🧠 RAG (Retrieval-Augmented Generation)

* Document ingestion (TXT, MD, PY, JS, HTML, CSS, PDF, DOCX, images as placeholders)
* Embeddings via **sentence-transformers/all-MiniLM-L6-v2**
* Search with cosine similarity on embeddings
* Results injected into prompt context

### 📁 Project & Message Management

* Projects stored in SQLite (`projects`, `messages`, `settings`, `rag_documents`)
* Add, switch, and manage projects from toolbar
* Persisted chat history, attachments, and settings per project

### 🎤 Voice & Audio

* Text-to-speech via `pyttsx3`
* Speech recognition via Google STT (`speech_recognition` + `pyaudio`)
* Hotkey voice input

### 🖼️ Screenshots & Files

* Capture full screen (via `PIL.ImageGrab`)
* Attach files (PDF, DOCX, TXT, MD, images, etc.)
* Inline content preview from attachments

### 💾 Export & Sharing

* Export chats to **Markdown** with timestamps and role formatting

### 🛠️ UI & UX

* Tkinter desktop app with dark theme
* Resizable window, toolbar, chat pane, input box, status bar
* Hotkeys: Enter to send, Ctrl+Alt+Space overlay planned
* Status updates (connection, RAG indexing, errors)

### ⚙️ Settings

* Configurable **temperature**, **max tokens**, and RAG enable/disable
* Stored in SQLite `settings` table

---

## Installation

### Requirements

* **Windows 10/11**
* **Python 3.10+**

### First Run

```bash
python claudex_pro.py
```

On launch the app will:

1. Create a `.venv` virtual environment if missing.
2. Auto-install all required dependencies.
3. Relaunch itself inside the venv.

---

## Dependencies (auto-installed)

* `requests`
* `tkinter-tooltip`
* `Pillow`
* `numpy`
* `scikit-learn`
* `sentence-transformers`
* `pyttsx3`
* `SpeechRecognition`
* `pyaudio`
* `python-docx`
* `PyMuPDF`
* `openai`

---

## Database

* The app uses **`claudex_ultra.db`** as its **canonical database**.
* Legacy `claudex.db` is deprecated; delete or archive it.
* Schema includes:

  * `projects`
  * `messages`
  * `settings`
  * `rag_documents`

---

## Usage

* Select or create a **project** from the toolbar.
* Type a message, attach files or screenshots, and send.
* Toggle **RAG** to inject relevant indexed content.
* Export a chat to **Markdown** anytime.
* Use **voice input** to dictate a query.

---

## Current Limitations

* Only LM Studio backend implemented (no Ollama, OpenAI, Anthropic, etc. yet).
* Region screenshot not yet supported (fullscreen only).
* No global overlay hotkey yet.
* RAG uses SQLite with byte embeddings only (no FAISS/Qdrant adapter).

---

## Roadmap (for Jules AI)

* 🔗 Add full backend abstraction (Ollama, llama.cpp, Anthropic, OpenAI, Mistral, Gemini, etc.)
* 📚 Upgrade RAG with SQLite-VSS + FAISS/Qdrant, citations, re-ranking
* 🖼️ Vision + OCR (Tesseract, PaddleOCR, Gemini Vision)
* 🎤 Faster-Whisper STT + Piper TTS
* 🔥 Global overlay hotkey, quick capture HUD
* 📊 Cost/latency HUD + per-backend health checks
* 🛡️ Privacy (local-only mode, secrets vault, audit logs)
* ⚡ Packaging (PyInstaller EXE, MSIX installer, auto-update)

---

## License

MIT (placeholder — adjust before release)

---

## Author

Guillaume Lessard – iD01t Productions



