feature/omnimind-studio-final
# OmniMind Studio

**Version 3.0.0**

OmniMind Studio is a feature-rich, all-in-one desktop AI powerhouse for Windows. It provides a unified interface for interacting with multiple local and commercial AI models, a professional-grade RAG (Retrieval-Augmented Generation) system, and a suite of productivity tools designed for developers, writers, and AI enthusiasts.

![OmniMind Studio Screenshot](assets/screenshot.png) *(Note: Placeholder image)*

## Features

-   **Multi-Backend Support**: Seamlessly switch between different AI providers:
    -   **Local**: LM Studio, Ollama
    -   **Commercial**: OpenAI, Anthropic, Mistral, Google Gemini
-   **RAG 2.0 System**: A powerful, local-first RAG system built on `vectorlite` for high-speed, scalable document indexing and retrieval. Index entire folders of documents and have their content automatically included as context in your prompts.
-   **Secure API Key Management**: API keys are stored securely in the native Windows Credential Manager using the `keyring` library. No more plaintext keys in config files.
-   **Polished & Productive UI**:
    -   A sleek, professional dark theme.
    -   **Performance HUD**: Real-time display of latency, token counts, and estimated cost for each API call.
    -   **Compare & Diff View**: Run the same prompt against two different models and see a color-coded diff of their responses side-by-side.
    -   **Prompt Studio**: Create, save, and manage a library of your favorite prompt templates for quick reuse.
-   **Multimodal Capabilities**:
    -   **OCR/Vision**: Attach images (`.png`, `.jpg`, etc.) to your prompt, and the app will use `pytesseract` to perform OCR and include the text content.
    -   **Voice Input**: Use your microphone to dictate prompts.
-   **Productivity Boosters**:
    -   **Global Hotkey**: Press `Ctrl+Alt+Space` from anywhere in Windows to instantly bring OmniMind Studio to the foreground.
    -   **Screenshot Tool**: Instantly capture your screen and attach it as a file for OCR.
    -   **Slash Commands**: Use commands like `/models` in the input box for quick actions.
    -   **Chat Export**: Export your conversations to Markdown files.

## Setup and Installation

### Prerequisites

-   Windows 10 or 11
-   Python 3.9+ (make sure it's added to your PATH)
-   Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd omnimind-studio
    ```

2.  **Create and Activate a Virtual Environment:**
    Using a virtual environment is strongly recommended to avoid conflicts with other Python projects.
    ```powershell
    # Create the virtual environment
    python -m venv .venv
    # Activate it (for PowerShell)
    .\.venv\Scripts\Activate.ps1
    ```
    *(For Command Prompt, use `.venv\Scripts\activate.bat`)*

3.  **Install Dependencies:**
    This step installs all the necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Tesseract for OCR:**
    The "Vision" feature, which reads text from images, requires Google's Tesseract OCR engine.
    -   Download the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page.
    -   During installation, **make sure to note the installation path**.
    -   Add the Tesseract installation directory to your system's `PATH` environment variable. For example, if you installed it to `C:\Program Files\Tesseract-OCR`, add that exact path to your `PATH`.

## How to Run

Once the setup is complete, you can run the application directly from the script:
=======
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
main

No complex setup is required. Simply run the script from your terminal:

```bash
python omnimind_studio.py
```
feature/omnimind-studio-final
The first time you run the app, it will perform a series of health checks. If any issues are found (like a missing local AI server or an invalid API key), a diagnostic message will appear.

### Offline and Local-Only Mode

OmniMind Studio works perfectly well without an internet connection, provided you are using local backends like **LM Studio** or **Ollama**.
-   Ensure your local AI server (LM Studio or Ollama) is running before you start the application.
-   The application will detect if it can't connect to a commercial service and will inform you, but it will not prevent you from using the local services that are available.

## Troubleshooting
On the first launch, the application will automatically:
1.  Create a local `.venv` virtual environment in its directory.
2.  Install all required Python dependencies into the venv.
3.  Relaunch itself inside the fully configured environment.
main

-   **`IndentationError` or other syntax errors on startup:** Ensure you are running a compatible version of Python (3.9+).
-   **`pytesseract.TesseractNotFoundError`:** This means the application can't find the Tesseract OCR engine. Make sure it's installed and that its directory is correctly added to your system's `PATH`.
-   **API Key Errors:** Use the "Test" button in `Settings -> Manage API Keys` to verify your keys are correct.
-   **Log Files:** For more detailed error information, check the log files located in the `logs/` directory.

feature/omnimind-studio-final
## How to Build an Executable

The `build_win.ps1` PowerShell script is provided to bundle the application into a standalone `.exe` file using PyInstaller.

1.  Make sure you have all dependencies installed, including `pyinstaller`.
2.  Run the build script from a PowerShell terminal:
    ```powershell
    .\build_win.ps1
    ```
3.  The final executable will be located in the `dist/` directory.

**Note on MSIX Packaging:** The build script also contains commented-out logic for creating a modern MSIX installer. This is an advanced feature that requires the Windows SDK to be installed and a valid code-signing certificate. Please see the comments within the `build_win.ps1` script for more details if you wish to pursue this distribution method.

## Backend Setup

To use commercial models from OpenAI, Anthropic, or Mistral, you need to add your API keys.

1.  Click the **"Settings"** button in the toolbar.
2.  In the Settings window, click **"Manage API Keys"**.
3.  Enter your API keys into the respective fields.
4.  Click **"Save Keys"**. Your keys will be stored securely in the Windows Credential Manager. They are **not** stored in plain text.

---
main

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
