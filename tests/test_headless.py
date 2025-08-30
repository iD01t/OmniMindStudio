import unittest
import os
import sys
from pathlib import Path

# Add the root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Set the testing flag
os.environ["OMNI_HEADLESS"] = "1"

# Import the components to be tested
from omnimind_studio import (
    create_db_connection,
    init_database,
    get_setting,
    set_setting,
    GeminiBackend,
    SecretsManager
)

# We need to import keyring here to handle the case where it was deleted
# in the setUp method.
import keyring

class TestHeadless(unittest.TestCase):

    def setUp(self):
        """Set up a temporary in-memory database for each test."""
        self.db_conn = create_db_connection(":memory:")
        init_database(self.db_conn)
        # Ensure no real API keys from the environment interfere with tests
        self.original_gemini_key = SecretsManager.get_api_key("Gemini")
        if self.original_gemini_key:
            keyring.delete_password(SecretsManager.SERVICE_NAME, "Gemini")

    def tearDown(self):
        """Clean up after each test."""
        self.db_conn.close()
        # Restore original key if it existed
        if self.original_gemini_key:
            SecretsManager.set_api_key("Gemini", self.original_gemini_key)

    def test_db_bootstrap(self):
        """Test if the database and its tables are created successfully."""
        cursor = self.db_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
        self.assertIsNotNone(cursor.fetchone(), "The 'projects' table was not created.")
        cursor = self.db_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
        self.assertIsNotNone(cursor.fetchone(), "The 'settings' table was not created.")
        cursor = self.db_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_documents'")
        self.assertIsNotNone(cursor.fetchone(), "The 'rag_documents' table was not created.")

    def test_settings_read_write(self):
        """Test if settings can be written to and read from the database."""
        set_setting(self.db_conn, "test_key", "test_value")
        retrieved_value = get_setting(self.db_conn, "test_key")
        self.assertEqual(retrieved_value, "test_value", "The retrieved setting did not match the set value.")

        # Test default value
        default_value = get_setting(self.db_conn, "non_existent_key", "default")
        self.assertEqual(default_value, "default", "The default value was not returned for a non-existent key.")

    def test_gemini_connection_fails_gracefully(self):
        """Test that the Gemini backend fails gracefully without an API key."""
        # Ensure no key is set in the secret manager for this test
        # We need to handle the possibility that a key is already cached in the keyring backend
        try:
            keyring.delete_password(SecretsManager.SERVICE_NAME, "Gemini")
        except keyring.errors.NoKeyringError:
            pass # Ignore if no keyring backend is available
        except Exception:
            # This can happen if the key doesn't exist, which is fine
            pass

        gemini_backend = GeminiBackend()
        success, message = gemini_backend.test_connection()

        self.assertFalse(success, "Connection test should fail without an API key.")
        self.assertEqual(message, "Gemini API key not set.", "The failure message is incorrect.")

    def test_tkinter_not_imported(self):
        """Verify that tkinter is not in the list of imported modules."""
        self.assertNotIn("tkinter", sys.modules, "tkinter should not be imported in headless mode.")

    def test_rag_indexing_and_retrieval(self):
        """Test basic RAG indexing and retrieval."""
        from omnimind_studio import RAGSystem
        rag_system = RAGSystem(self.db_conn)

        # Test indexing
        test_content = "The quick brown fox jumps over the lazy dog."
        rag_system.add_document("test_doc", test_content)

        # Check if the content was added
        doc_count = self.db_conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
        self.assertGreater(doc_count, 0, "Document was not added to the RAG index.")

        # Test retrieval
        search_results = rag_system.search("lazy fox")
        self.assertIsInstance(search_results, list)
        self.assertGreater(len(search_results), 0, "RAG search returned no results.")
        self.assertIn("lazy dog", search_results[0]['content'], "RAG search result content is incorrect.")



if __name__ == '__main__':
    unittest.main()
