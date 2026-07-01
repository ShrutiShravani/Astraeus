import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.constants.constant import MAX_CONCURRENT_JOBS,chunk_overlap,chunk_size,REQUEST_TIMEOUT
from dotenv import load_dotenv
from qdrant_client.http import models
import re
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from datetime import datetime
from src.utils.dlq import send_to_dlq
from custom_logging import logger
import time

load_dotenv()
MAX_RETRIES = 3

BASE_DIR= Path(os.getenv("DATA_DIR"))

MASKED_DIR= BASE_DIR/"masked"


COLLECTION_NAME= "financial_reports"


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

#intialzie Qdrant
client = QdrantClient(url="http://localhost:6333")

# Initialize Embeddings (Cost-optimized)
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
total_failed=0

total_retries=0

def ensure_collection():
    collections=client.get_collections().collections
    exists=any(c.name==COLLECTION_NAME for c in collections)
    if exists:
        collection_info = client.get_collection(COLLECTION_NAME)
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config,dict):
            current_size= vectors_config["dense"].size
        else:
            current_size= vectors_config.size
        if current_size != 384:
            print(f"Dimension mismatch! Existing: {current_size}, Required: 384. Recreating...")
            client.delete_collection(COLLECTION_NAME)
            exists = False  # Set to false so the next block creates it fresh
        else:
            print(f"Collection {COLLECTION_NAME} is already configured correctly (384).")

    if not exists:
        print(f"Creating collection {COLLECTION_NAME} with 384 dimensions...")
        client.create_collection(
        collection_name=COLLECTION_NAME,
        # NOTICE THE CHANGE HERE: We use a dictionary to NAME our vectors
        vectors_config={
            "dense": models.VectorParams(
                size=384, 
                distance=models.Distance.COSINE
            )
        },
        # ADD THIS: Enable the sparse index for SPLADE
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams()
            )
        }
    )

def process_and_upload(md_files: list[Path]):
    uploaded_chunks = 0
    total_failed = 0
    total_retries = 0
    
    ensure_collection()
   
    
    # 1. Initialize Vectorstore ONCE
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        vector_name="dense"
    )

    # 2. Splitters
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")], strip_headers=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print(f"Found {len(md_files)} masked files to process...")

    # 3. Process Files
    for md_path in md_files:
        filename = md_path.name
        document_id = md_path.stem
        
        company, year = robust_metadata_extractor(filename)
        if not company or not year:
            print(f"Skipping {filename}: Metadata not found.")
            continue
        
        if "TRANSCRIPT" in filename.upper() or "transcripts" in str(md_path):
            doc_type = "Transcript"
        else:
            doc_type = "10K"

        # SENIOR CHECK: Skip if company is already in DB (Optional but saves time)
        # You can add a filter check here if you want to be truly professional.
        content = md_path.read_text(encoding="utf-8")

        
        page_blocks = re.split(r"## PAGE (\d+).*?\n", content)
        all_docs = []

        # Process all pages first
        start_idx = 1 if len(page_blocks[0].strip()) < 5 else 0

        for i in range(start_idx, len(page_blocks), 2):
            current_page_number = int(page_blocks[i])
            current_page_content = page_blocks[i+1]
            chunk_no=1
            current_page_content = re.sub(r"^={10,}\n+", "", current_page_content)
            
            md_header_splits = markdown_splitter.split_text(current_page_content)

            for section in md_header_splits:
                # Chunking logic
                if "|" in section.page_content or len(section.page_content) < chunk_size:
                    sub_chunks = [section]
                else:
                    sub_chunks = text_splitter.split_documents([section])
                
                for chunk in sub_chunks:
                    if len(chunk.page_content.strip()) < 10:
                        continue
                    chunk.metadata={
                        "document_id": md_path.stem,
                        "company": company,
                        "year": int(year),
                        "source": filename,
                        "page_num": current_page_number,
                        "doc_type": doc_type
                    }
                    all_docs.append(chunk)
                    chunk_no += 1  
                
        # 4. UPLOAD ONCE PER FILE (Correct Indentation)
        print(f"Uploading {len(all_docs)} total chunks for {company}...")
        batch_size = 50

        for j in range(0, len(all_docs), batch_size):
            batch_start = time.time()

            batch = all_docs[j : j + batch_size]
            pages=sorted({doc.metadata['page_num'] for doc in batch})
            page_range=(str(pages[0]) if len(pages)==1 else f"{pages[0]}-{pages[-1]}")
            if len(batch)==0:
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    ids= [doc.metadata["chunk_id"] for doc in batch]
                    vectorstore.add_documents(documents=batch,id=ids)
                    print(f"Done: {company} {year} | Batch {j}-{j+len(batch)}")
                    uploaded_chunks+=len(batch)
                    latency = time.time() - batch_start
                    logger.info({
                        "document":document_id,
                        "page_range": page_range,
                        "latency":latency,
                        "chunks": len(batch)
                    })
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        total_retries+=1
                        continue
                    
                    else:
                        print(f"Error uploading {company}: {e}")
                        send_to_dlq(
                                    document_id=document_id,
                                    stage="masking", 
                                    page= page_range,
                                    reason="Batch failed", 
                                    engine= None,
                                    error=e
                                )
                        
                        logger.error({
                            "document": document_id,
                            "chunks": len(batch),
                            "page_range": page_range,
                            "error": str(e)
                        })
                        total_failed+=len(batch)
    
    with open ("data/vector_store_status.txt", "w") as f:
        f.write(f"Indexed {len(all_docs)} chunks on {datetime.now()}")

    print("Ingestion Complete: All files processed exactly once.")

def robust_metadata_extractor(filename):
    """Senior Logic: Extracts metadata regardless of filename structure."""
    clean_name = filename.replace(".md", "").upper()
    year_match = re.search(r'(19|20)\d{2}', clean_name)
    if not year_match:
        return None, None
    
    year = year_match.group(0)
    year_start_index = clean_name.find(year)
    company_raw = clean_name[:year_start_index]
    company_name = company_raw.strip('_')
    
    return company_name, year

if __name__ == "__main__":
    # LOCAL BATCH MODE
    files = list(MASKED_DIR.rglob("*.md"))
    process_and_upload(files)