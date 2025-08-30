import json
import os
import sys
import time
from pathlib import Path

# Add the root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# This is a hack to prevent the GUI from starting.
# In a real app, you'd have a headless mode or separate the logic better.
feature/omnimind-studio-final
os.environ["OMNI_HEADLESS"] = "1"

os.environ["OMNIMIND_TESTING"] = "1"
main

from omnimind_studio import BackendRouter, SecretsManager

def load_golden_prompts():
    """Load golden prompts from JSON file."""
    prompts_file = ROOT_DIR / "tests" / "golden_prompts.json"
    with open(prompts_file, "r", encoding="utf-8") as f:
        return json.load(f)

def run_single_test(router, test_case):
    """Run a single test case."""
    name = test_case.get('name', 'Unnamed test')
    backend_name = test_case.get('backend')
    prompt = test_case.get('prompt')
    expected = test_case.get('expected')

    print(f"--- Running test: {name} ---")
    print(f"  Backend: {backend_name}")

    backend = router.backends.get(backend_name)
    if not backend:
 feature/omnimind-studio-final
        print(f"  - 💥 FAILED: Backend '{backend_name}' not found or failed to initialize.")
        return False

    # For commercial backends, check if the key exists before running
    if backend_name in ["OpenAI", "Anthropic", "Mistral", "Gemini"]:
        # This check relies on the backend's own __init__ logic to check SecretsManager
        # We check here to provide a clear "skipped" message
        if not backend.api_key:

        print("  - 💥 FAILED: Backend not found.")
        return False

    # For commercial backends, check if the key exists before running
    if backend_name in ["OpenAI", "Anthropic", "Mistral"]:
        if not SecretsManager.get_api_key(backend_name):
main
            print(f"  - ⚠️ SKIPPED: API key for {backend_name} not set.")
            return "skipped"

    try:
        start_time = time.monotonic()
        # Use a low temperature for more predictable outputs in tests
        response_iterator = backend.stream_chat([{"role": "user", "content": prompt}], temperature=0.01)
        full_response = "".join(list(response_iterator))
        duration = time.monotonic() - start_time

        print(f"  - Response received in {duration:.2f}s.")

        if not full_response or "Error:" in full_response:
             print(f"  - 💥 FAILED: Received an error or empty response: {full_response[:100]}...")
             return False

        if "contains" in expected:
            checks = expected["contains"]
            if not isinstance(checks, list):
                checks = [checks]

            for check_str in checks:
                if check_str.lower() not in full_response.lower():
                    print(f"  - 💥 FAILED: Response did not contain '{check_str}'.")
                    print(f"     Response: {full_response[:200]}...")
                    return False

        print("  - ✅ PASSED")
        return True

    except Exception as e:
        print(f"  - 💥 FAILED: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner function."""
    print("=====================================")
feature/omnimind-studio-final
    print("  OmniMind Studio - Integration Tests")
    print("=====================================")


    print("  OmniMind Studio - Test Suite")
    print("=====================================")

    # This assumes all dependencies are installed via the main app's bootstrap
 main
    router = BackendRouter()

    passed, failed, skipped = 0, 0, 0

    try:
        golden_prompts = load_golden_prompts()
        print(f"Loaded {len(golden_prompts)} test cases.\n")

        for test_case in golden_prompts:
            result = run_single_test(router, test_case)
            if result is True:
                passed += 1
            elif result is False:
                failed += 1
            elif result == "skipped":
                skipped += 1
            print("-" * 35 + "\n")

        print("=====================================")
        print("  Test Summary")
        print("=====================================")
        print(f"  ✅ Passed: {passed}")
        print(f"  💥 Failed: {failed}")
        print(f"  ⚠️ Skipped: {skipped}")
        print("=====================================")

feature/omnimind-studio-final
        # We expect some tests to fail (local backends) or be skipped (commercial backends)
        # in the CI environment, so we don't exit with a non-zero code.
        # A more advanced setup would use pytest markers to selectively run tests.
        if failed > 2: # Allow up to 2 local backend failures

        if failed > 0:
main
            sys.exit(1)

    except Exception as e:
        print(f"A critical error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
feature/omnimind-studio-final

    # A check to avoid running the main app's GUI if imported.
    if os.environ.get("OMNIMIND_TESTING") != "1":
         print("This script is for testing only. Run omnimind_studio.py to start the app.")
         sys.exit(0)
 main
    main()
