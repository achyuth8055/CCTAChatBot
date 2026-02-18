import os
import re
import sys
import io
import contextlib
import logging
import warnings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "true")

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "cookcounty_tax_faqs"

def _clean_document_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if re.search(
            r"(©|copyright|all rights reserved|privacy policy|terms of service|disclaimer|policies & agreements|^\d{1,2}/\d{1,2}/\d{4}|^https?://|^Page \d+|^\d+$)",
            line.strip(),
            re.I
        ):
            continue
        if re.match(r"^(Company|Services|Pricing|Blogs|Contact Us|Deadlines)$", line.strip(), re.I):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def compute_embeddings_batch(texts):
    import os
    import io
    import contextlib
    sys.stderr = io.StringIO() 
    
    from chromadb.utils import embedding_functions
    ef = embedding_functions.DefaultEmbeddingFunction()
    return ef(texts)

if __name__ == '__main__':
    _stderr_suppressor = io.StringIO()
    
    if os.path.exists(CHROMA_PATH):
        try:
            import shutil
            shutil.rmtree(CHROMA_PATH)
            print(f"[Chroma] Cleared existing database at {CHROMA_PATH}")
        except Exception as e:
            print(f"[Chroma] Warning: Could not clear database: {e}")

    with contextlib.redirect_stderr(_stderr_suppressor):
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"[Chroma] Initialized empty collection.")

    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"[Loader] Loaded {len(raw_documents)} documents from '{DATA_PATH}'.")

    print("[Preprocessor] Cleaning documents...")
    for doc in raw_documents:
        doc.page_content = _clean_document_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(raw_documents)
    print(f"[Splitter] Produced {len(chunks)} chunks (chunk_size=500, overlap=150).")

    documents = []
    metadata = []
    ids = []

    i = 0
    for chunk in chunks:
        if not chunk.page_content.strip():
            continue
        documents.append(chunk.page_content)
        ids.append(f"doc_{i}")
        metadata.append(chunk.metadata)
        i += 1

    print(f"[Preprocessor] Prepared {len(documents)} non-empty chunks for indexing.")

    if len(documents) > 0:
        num_workers = min(multiprocessing.cpu_count(), 8)
        print(f"[Parallel] Computing embeddings with {num_workers} workers...")
        
        chunk_size = (len(documents) + num_workers - 1) // num_workers
        batches = [documents[j:j + chunk_size] for j in range(0, len(documents), chunk_size)]
        
        all_embeddings = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            valid_batches = [b for b in batches if b]
            if not valid_batches:
                 print("[Parallel] No valid batches.")
            else:
                results = list(executor.map(compute_embeddings_batch, valid_batches))
                for res in results:
                    all_embeddings.extend(res)
                
        print(f"[Parallel] Computed {len(all_embeddings)} embeddings. Upserting to ChromaDB...")
        
        try:
            collection.upsert(
                documents=documents,
                embeddings=all_embeddings,
                metadatas=metadata,
                ids=ids
            )
            print(f"[Chroma] Successfully upserted {len(documents)} documents.")
        except Exception as e:
            print(f"[Chroma] Error during upsert: {e}")

    try:
        print(f"[Chroma] After upsert, document count: {collection.count()}")
    except Exception:
        pass
