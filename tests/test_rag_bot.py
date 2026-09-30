import unittest
from types import SimpleNamespace

from rag_bot import SourceText, build_context, embed_texts, generate_grounded_answer, make_chunks, split_text


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_size_and_overlap(self):
        text = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80
        chunks = split_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertIn("B", "".join(chunks))
        self.assertTrue(any(set(chunks[index]) & set(chunks[index + 1]) for index in range(len(chunks) - 1)))

    def test_invalid_chunk_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=10, overlap=10)

    def test_chunks_keep_source_and_page_metadata(self):
        chunks = make_chunks([SourceText("guide.pdf", "Grounded source text.", page=3)])
        self.assertEqual(chunks[0].metadata["source"], "guide.pdf")
        self.assertEqual(chunks[0].metadata["page"], 3)


class CitationTests(unittest.TestCase):
    def test_context_and_citation_metadata_share_labels(self):
        context, sources = build_context(
            ["A source passage."],
            [{"source": "guide.pdf", "page": 2, "chunk": 1}],
        )
        self.assertIn("[S1] Source: guide.pdf, page 2", context)
        self.assertEqual(sources, [{"reference": "S1", "source": "guide.pdf", "location": "page 2"}])

    def test_text_citation_uses_section_metadata(self):
        _, sources = build_context(["Text passage."], [{"source": "guide.txt", "page": -1, "section": "Section 4"}])
        self.assertEqual(sources[0]["location"], "Section 4")


class EmbeddingBatchTests(unittest.TestCase):
    def test_embedding_requests_are_batched_to_the_configured_limit(self):
        class FakeModels:
            def __init__(self):
                self.batch_sizes = []
                self.task_types = []

            def embed_content(self, model, contents, config):
                self.batch_sizes.append(len(contents))
                self.task_types.append(config["task_type"])
                embeddings = [SimpleNamespace(values=[float(index)]) for index in range(len(contents))]
                return SimpleNamespace(embeddings=embeddings)

        models = FakeModels()
        client = SimpleNamespace(models=models)
        vectors = embed_texts([f"text {index}" for index in range(130)], client)
        self.assertEqual(models.batch_sizes, [64, 64, 2])
        self.assertEqual(models.task_types, ["RETRIEVAL_DOCUMENT"] * 3)
        self.assertEqual(len(vectors), 130)

    def test_query_embeddings_use_the_retrieval_query_task(self):
        class FakeModels:
            def embed_content(self, model, contents, config):
                self.task_type = config["task_type"]
                return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

        models = FakeModels()
        vectors = embed_texts(["question"], SimpleNamespace(models=models), task_type="RETRIEVAL_QUERY")
        self.assertEqual(models.task_type, "RETRIEVAL_QUERY")
        self.assertEqual(vectors, [[0.1, 0.2]])


class GroundedAnswerTests(unittest.TestCase):
    def test_answer_uses_chat_send_message(self):
        class FakeChat:
            def send_message(self, message):
                self.message = message
                return SimpleNamespace(text="Supported answer [S1].")

        class FakeChats:
            def create(self, model, config):
                self.model = model
                self.config = config
                self.chat = FakeChat()
                return self.chat

        chats = FakeChats()
        answer = generate_grounded_answer("Question?", "[S1] Source passage", SimpleNamespace(chats=chats))
        self.assertEqual(answer, "Supported answer [S1].")
        self.assertIn("Question: Question?", chats.chat.message)
        self.assertEqual(chats.config["temperature"], 0)
        self.assertIn("supplied source excerpts", chats.config["system_instruction"])


if __name__ == "__main__":
    unittest.main()
