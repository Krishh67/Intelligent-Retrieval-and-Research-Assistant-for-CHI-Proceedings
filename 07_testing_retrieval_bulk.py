

















import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path("embedding")

FAISS_PATH = BASE_DIR / "faiss.index"
METADATA_PATH = BASE_DIR / "metadata.pkl"

# IMPORTANT:
# point this to your chunks.jsonl location
CHUNKS_PATH = Path(
    "chunks.jsonl"
)

MODEL_NAME = "BAAI/bge-large-en-v1.5"

TOP_K = 5

# ==========================================================
# TEST QUERIES
# ==========================================================

TEST_QUERIES = [

    # LLMs / GenAI
    "What CHI papers discuss the use of large language models for human-computer interaction?",
    "How are researchers using GPT and other language models to support interactive systems?",
    "What work has been done on conversational AI assistants in CHI 2024?",
    "Which CHI papers explore integrating LLMs into user interfaces?",

    # UX / Usability
    "How can AI help automate usability evaluation of user interfaces?",
    "What research exists on AI-generated usability feedback for designers?",
    "Are there CHI papers that use language models for heuristic evaluation?",
    "How are conversational agents being used to support UX evaluation?",

    # Design Tools
    "What tools help designers prototype applications using AI foundation models?",
    "How can designers combine multiple AI models into a single workflow?",
    "What systems support multimodal AI prototyping for designers?",
    "Are there visual programming approaches for working with foundation models?",

    # Human-AI Collaboration
    "How do humans and AI systems collaborate during creative work?",
    "What are effective approaches for human-AI cooperation and shared decision making?",
    "Which CHI papers study trust and collaboration between humans and AI?",
    "How can AI act as a collaborative partner instead of just a tool?",

    # Creativity
    "How can generative AI support creativity and ideation?",
    "What research explores AI-assisted creative design workflows?",
    "How are designers using generative models during the creative process?",
    "Which CHI papers investigate co-creation with AI systems?",

    # Accessibility
    "How are AI systems being used to improve accessibility?",
    "What CHI research focuses on assistive technologies powered by AI?",
    "How can machine learning help people with disabilities?",
    "What accessibility applications of large language models have been studied?",

    # Education
    "How are large language models being used in educational settings?",
    "What CHI papers discuss AI tutors and learning support systems?",
    "How can AI improve student learning experiences?",
    "What research explores interactive educational tools using generative AI?",

    # Healthcare
    "How is AI being applied to healthcare and medical decision support?",
    "What CHI papers investigate human-AI interaction in healthcare?",
    "How can conversational AI assist patients or healthcare providers?",
    "What are the challenges of deploying AI systems in medical contexts?",

    # Mobile / UI
    "What research has been done on mobile user interfaces enhanced by AI?",
    "How are language models being integrated into mobile applications?",
    "Which CHI papers explore intelligent user interfaces?",
    "What AI techniques are being used to improve mobile UX?",

    # Trust / Ethics
    "How do users develop trust in AI systems?",
    "What factors influence trust and reliance on AI assistants?",
    "Which CHI papers discuss responsible AI and ethical concerns?",
    "How can AI systems be made more transparent and explainable?",

    # Multimodal
    "What multimodal AI systems were presented at CHI 2024?",
    "How are text, image, audio, and video models being combined?",
    "What research explores multimodal interaction with foundation models?",
    "How can users build workflows that span multiple AI modalities?",

    # Retrieval Stress Tests
    "I am a UX designer and want an AI tool that can automatically review my interface and suggest improvements. What CHI papers should I read?",
    "I want to build a system that chains GPT, image generation, and speech models together. Has CHI published anything similar?",
    "What research would help me design a collaborative AI assistant for creative work?",
    "I am looking for papers about using generative AI as a design partner rather than a content generator.",
    "How can AI help designers rapidly prototype multimodal applications?",
    "What are the most relevant CHI papers on foundation models and design tools?"
]

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded")

# ==========================================================
# LOAD FAISS
# ==========================================================

print("Loading FAISS...")
index = faiss.read_index(str(FAISS_PATH))
print("Vectors:", index.ntotal)

# ==========================================================
# LOAD METADATA
# ==========================================================

print("Loading metadata...")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Metadata:", len(metadata))

# ==========================================================
# LOAD CHUNKS
# ==========================================================

print("Loading chunks...")

chunks = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print("Chunks:", len(chunks))

# ==========================================================
# RUN BENCHMARK
# ==========================================================

REPORT_FILE = BASE_DIR / "retrieval_report_2.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    for q_num, query in enumerate(TEST_QUERIES, start=1):

        print(f"Running query {q_num}/{len(TEST_QUERIES)}")

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        report.write("=" * 100 + "\n")
        report.write(f"QUERY {q_num}\n")
        report.write(query + "\n")
        report.write("=" * 100 + "\n\n")

        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1
        ):

            meta = metadata[idx]
            chunk = chunks[idx]

            report.write(f"TOP {rank}\n")
            report.write(f"Score: {score:.4f}\n")
            report.write(f"Paper ID: {meta['paper_id']}\n")
            report.write(f"Year: {meta['year']}\n")
            report.write(f"Title: {meta['title']}\n")
            report.write(f"Section: {meta['section']}\n\n")

            report.write("TEXT PREVIEW:\n")
            report.write(chunk["text"][:1200])

            report.write("\n\n")
            report.write("-" * 100)
            report.write("\n\n")

print()
print("DONE")
print("Saved:")
print(REPORT_FILE)