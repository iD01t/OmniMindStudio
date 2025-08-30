
<img width="1024" height="1024" alt="ChatGPT Image Aug 29, 2025, 10_52_13 PM" src="https://github.com/user-attachments/assets/ce530457-1b14-43e2-984b-b6868fad58d5" />

# OmniMind Studio

**Version 3.0.0**

OmniMind Studio is a feature-rich, all-in-one desktop AI powerhouse for Windows. It provides a unified interface for interacting with multiple local and commercial AI models, a professional-grade RAG (Retrieval-Augmented Generation) system, and a suite of productivity tools designed for developers, writers, and AI enthusiasts.


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

```bash
python omnimind_studio.py
```

The first time you run the app, it will perform a series of health checks. If any issues are found (like a missing local AI server or an invalid API key), a diagnostic message will appear.

### Offline and Local-Only Mode

OmniMind Studio works perfectly well without an internet connection, provided you are using local backends like **LM Studio** or **Ollama**.
-   Ensure your local AI server (LM Studio or Ollama) is running before you start the application.
-   The application will detect if it can't connect to a commercial service and will inform you, but it will not prevent you from using the local services that are available.

## Troubleshooting

-   **`IndentationError` or other syntax errors on startup:** Ensure you are running a compatible version of Python (3.9+).
-   **`pytesseract.TesseractNotFoundError`:** This means the application can't find the Tesseract OCR engine. Make sure it's installed and that its directory is correctly added to your system's `PATH`.
-   **API Key Errors:** Use the "Test" button in `Settings -> Manage API Keys` to verify your keys are correct.
-   **Log Files:** For more detailed error information, check the log files located in the `logs/` directory.

## How to Build an Executable

The `build_win.ps1` PowerShell script is provided to bundle the application into a standalone `.exe` file using PyInstaller.

1.  Make sure you have all dependencies installed, including `pyinstaller`.
2.  Run the build script from a PowerShell terminal:
    ```powershell
    .\build_win.ps1
    ```
3.  The final executable will be located in the `dist/` directory.

**Note on MSIX Packaging:** The build script also contains commented-out logic for creating a modern MSIX installer. This is an advanced feature that requires the Windows SDK to be installed and a valid code-signing certificate. Please see the comments within the `build_win.ps1` script for more details if you wish to pursue this distribution method.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
