import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import rag


class DocumentTests(unittest.TestCase):
    def test_text_file_is_read_as_utf8(self):
        self.assertEqual(rag.read_document("notes.txt", b"hello AskCode"), "hello AskCode")

    def test_python_file_is_read_as_utf8(self):
        self.assertEqual(rag.read_document("notes.py", b"def hello(): pass"), "def hello(): pass")

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(ValueError):
            rag.read_document("notes.docx", b"not supported")

    def test_chunks_overlap_and_keep_the_full_text(self):
        text = "a" * 1000
        chunks = rag.split_into_chunks(text)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][-150:], chunks[1][:150])
        self.assertEqual("".join(chunks[0] + chunks[1][150:]), text)

    def test_python_chunks_are_functions_with_line_citations(self):
        source = (
            "import os\n"
            "\n"
            "def first():\n"
            "    return os.getcwd()\n"
            "\n"
            "async def second():\n"
            "    return 2\n"
        )

        chunks = rag.make_document_chunks("sample.py", source)

        self.assertEqual([chunk["kind"] for chunk in chunks], ["function", "function"])
        self.assertEqual(
            [(chunk["start_line"], chunk["end_line"]) for chunk in chunks],
            [(3, 4), (6, 7)],
        )


class RetrievalTests(unittest.TestCase):
    def test_hybrid_search_includes_lexical_match(self):
        class FakeCollection:
            def count(self):
                return 2

            def get(self, include):
                return {
                    "ids": ["semantic", "keyword"],
                    "documents": ["Unrelated code about requests.", "Bearer tokens expire after one hour."],
                    "metadatas": [
                        {"source": "other.py", "start_line": 1, "end_line": 1, "kind": "function"},
                        {"source": "auth.py", "start_line": 2, "end_line": 4, "kind": "function"},
                    ],
                }

            def query(self, **kwargs):
                return {
                    "ids": [["semantic", "keyword"]],
                    "documents": [[
                        "Unrelated code about requests.",
                        "Bearer tokens expire after one hour.",
                    ]],
                    "metadatas": [[
                        {"source": "other.py", "start_line": 1, "end_line": 1, "kind": "function"},
                        {"source": "auth.py", "start_line": 2, "end_line": 4, "kind": "function"},
                    ]],
                    "distances": [[0.2, 0.3]],
                }

        with patch.object(rag, "get_provider", return_value="ollama"):
            with patch.object(
                rag, "get_collection", return_value=FakeCollection()
            ):
                with patch.object(rag, "get_llm_client") as llm_factory:
                    llm_factory.return_value.embed.return_value = [[0.1, 0.2]]
                    results = rag.search_documents(
                        "When do bearer tokens expire?"
                    )

        self.assertTrue(any(item["source"] == "auth.py" for item in results))

    def test_no_relevant_results_returns_not_found_without_calling_llm(self):
        with patch.object(rag, "search_documents", return_value=[]):
            with patch.object(rag, "get_llm_client") as llm_factory:
                answer, sources = rag.answer_question("Which database is used?")

        self.assertIn("couldn't find", answer)
        self.assertEqual(sources, [])
        llm_factory.assert_not_called()

    def test_hybrid_search_filters_unrelated_passages(self):
        class FakeCollection:
            def count(self):
                return 1

            def get(self, include):
                return {
                    "ids": ["unrelated"],
                    "documents": ["The weather is sunny."],
                    "metadatas": [
                        {
                            "source": "weather.txt",
                            "start_line": 1,
                            "end_line": 1,
                            "kind": "text",
                        }
                    ],
                }

            def query(self, **kwargs):
                return {
                    "ids": [["unrelated"]],
                    "documents": [["The weather is sunny."]],
                    "metadatas": [[
                        {
                            "source": "weather.txt",
                            "start_line": 1,
                            "end_line": 1,
                            "kind": "text",
                        }
                    ]],
                    "distances": [[1.4]],
                }

        with patch.object(rag, "get_provider", return_value="ollama"):
            with patch.object(
                rag, "get_collection", return_value=FakeCollection()
            ):
                with patch.object(rag, "get_llm_client") as llm_factory:
                    llm_factory.return_value.embed.return_value = [[0.1, 0.2]]
                    results = rag.search_documents("What database stores the users?")

        self.assertEqual(results, [])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_health_returns_selected_provider(self):
        with patch.object(api, "get_provider", return_value="gemini"):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "provider": "gemini"})

    def test_ask_returns_answer_and_source_lines(self):
        sources = [
            {
                "source": "auth.py",
                "start_line": 2,
                "end_line": 4,
                "kind": "function",
            }
        ]
        with patch.object(api, "answer_question", return_value=("Tokens last one hour.", sources)):
            response = self.client.post("/ask", json={"question": "How long do tokens last?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], sources)

    def test_ask_returns_client_error_for_blank_question(self):
        with patch.object(api, "answer_question", side_effect=ValueError("Type a question first.")):
            response = self.client.post("/ask", json={"question": " "})

        self.assertEqual(response.status_code, 400)

    def test_index_upload_calls_the_document_indexer(self):
        with patch.object(api, "index_document", return_value=2) as indexer:
            response = self.client.post(
                "/index",
                files={"files": ("sample.py", b"def hello():\n    return 'hi'", "text/x-python")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"indexed": [{"filename": "sample.py", "sections": 2}]},
        )
        indexer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
