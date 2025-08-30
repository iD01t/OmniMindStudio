# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-08-30

### Added
- **Multi-Backend AI Support**: Added a flexible backend router to support multiple AI providers, including LM Studio, Ollama, OpenAI, Anthropic, Mistral, and Google Gemini.
- **RAG 2.0 System**: Implemented a high-performance, local-first RAG system using `vectorlite`.
- **Secure API Key Management**: API keys are now stored securely in the native Windows Credential Manager using `keyring`.
- **New UI Features**:
    - "Compare & Diff" view to compare responses from two models.
    - "Prompt Studio" for creating, saving, and managing prompt templates.
    - Performance HUD to display latency, token counts, and estimated cost.
- **Multimodal & Productivity Tools**:
    - OCR support for images attached to prompts.
    - Voice input for dictating prompts.
    - Global hotkey (`Ctrl+Alt+Space`) to activate the app.
    - Screenshot tool for quick image capture and OCR.
- **Slash Commands**: Implemented `/models`, `/bench`, `/citations`, `/memory`, `/vision.ask`, and `/export` commands.
- **Startup Health Checks**: The application now performs diagnostic checks on startup to ensure database and backend connectivity.
- **Structured Logging**: Added rotating file-based logging to the `logs/` directory.

### Changed
- **Complete Architectural Refactor**: The application was refactored to separate the core "engine" logic from the UI, enabling headless testing and improving maintainability.
- **Dependency Management**: Switched from a self-bootstrapping script to a standard `requirements.txt` file.
- **Database Backend**: Migrated from standard `sqlite3` to `apsw` to support the `vectorlite` extension.

### Fixed
- **Headless Testing Crash**: Resolved a critical bug where GUI libraries were imported in headless environments, causing tests to fail.
- **Numerous UI and Logic Bugs**: Fixed bugs in the Prompt Studio, API key testing, database API calls, and Gemini chat history handling.
- **Project Structure**: Created a complete and robust project structure with all necessary files (`README.md`, build scripts, tests, etc.).
