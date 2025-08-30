<img width="1024" height="1024" alt="OmniMind Studio Logo" src="https://github.com/user-attachments/assets/46d3536b-f7f3-424a-b4db-b4d82d30b23c" />

# OmniMind Studio

*Your Local-First, Cloud-Ready, AI Powerhouse*

---

## Overview

OmniMind Studio is a powerful, single-file Python desktop application for Windows that serves as a comprehensive interface for a wide range of local and commercial Large Language Models (LLMs). It features an advanced Retrieval-Augmented Generation (RAG) system, robust multi-backend support, and a suite of productivity tools designed for developers, writers, and AI enthusiasts.

The application is self-bootstrapping, automatically managing its own virtual environment and dependencies, making setup a breeze.

## Key Features

### 🔌 Universal Backend Connectivity

*   **Backend Abstraction Layer:** Connect to any supported AI model through a unified interface.
*   **Local Models:** Full support for **LM Studio** and **Ollama**, allowing you to run powerful models entirely offline.
*   **Commercial Models:** Seamless integration with major API providers:
    *   **OpenAI** (e.g., GPT-4, GPT-3.5)
    *   **Anthropic** (e.g., Claude 3 Opus, Sonnet, Haiku)
    *   **Mistral** (e.g., Mistral Large)
*   **Secure API Key Management:** API keys are stored securely in the native **Windows Credential Manager** using the `keyring` library.

### 🧠 RAG 2.0: Supercharged Memory

*   **High-Speed Vector Search:** The RAG system is powered by **`vectorlite`**, a high-performance SQLite extension that provides incredibly fast and efficient local vector similarity searches.
*   **Flexible Document Indexing:** Index entire folders of documents (`.txt`, `.md`, `.pdf`, `.docx`, and more) to create a custom knowledge base for your projects.
*   **Source-Grounded Responses:** The groundwork for full RAG citations is in place, with the system retrieving the source of information used in its responses.

### 🛠️ Advanced UI & UX

*   **Modern Dark Theme:** A clean, professional, and visually appealing dark theme applied across the entire application for comfortable use.
*   **Compare & Diff View:** Open a dedicated window to send the same prompt to two different models simultaneously and view their responses side-by-side, with a color-coded diff to highlight differences.
*   **Performance HUD:** A real-time Heads-Up Display shows the **latency**, **token count** (prompt and completion), and estimated **cost** for each interaction with a commercial model.
*   **Global Hotkey:** Bring OmniMind Studio to the foreground from anywhere in Windows using the system-wide hotkey: **`Ctrl+Alt+Space`**.
*   **Prompt Studio:** Create, save, and manage a library of your favorite and most-used prompt templates for quick reuse.

### 👁️ Multimodal Capabilities

*   **Image-to-Text (OCR):** The application can "read" text from attached images (`.png`, `.jpg`, etc.) using the Tesseract OCR engine and incorporate that text into your prompt context.

### ⚡ Commands & Productivity

*   **Slash Commands:** Use powerful slash commands directly in the input box for advanced actions.
    *   `/models`: Lists all available and configured backends.
    *   `/bench`: (Placeholder) For benchmarking model performance.
    *   `/doc.citations`: (Placeholder) For advanced citation-based responses.

---

## Installation

### Requirements

*   **Windows 10/11**
*   **Python 3.10+**
*   **(Optional) Tesseract OCR:** To enable the OCR feature for images, you must have Google's Tesseract OCR engine installed and available in your system's PATH. You can find installers on the official [Tesseract GitHub page](https://github.com/tesseract-ocr/tessdoc).

### First Run

No complex setup is required. Simply run the script from your terminal:

```bash
python omnimind_studio.py
```

On the first launch, the application will automatically:
1.  Create a local `.venv` virtual environment in its directory.
2.  Install all required Python dependencies into the venv.
3.  Relaunch itself inside the fully configured environment.

---

## Backend Setup

To use commercial models from OpenAI, Anthropic, or Mistral, you need to add your API keys.

1.  Click the **"Settings"** button in the toolbar.
2.  In the Settings window, click **"Manage API Keys"**.
3.  Enter your API keys into the respective fields.
4.  Click **"Save Keys"**. Your keys will be stored securely in the Windows Credential Manager. They are **not** stored in plain text.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
