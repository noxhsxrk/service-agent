from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import shutil
import os
from indexing_manager import IndexingManager
import time
from typing import List, Tuple, Any

console = Console()

async def process_document_chunk(documents_chunk, repo_dir, agent, index_path, progress_callback):
    try:
        # Split documents with enhanced metadata
        text_splitter = RecursiveCharacterTextSplitter(
            **agent.splitter_settings
        )
        splits = text_splitter.split_documents(documents_chunk)

        # Add chunk context to metadata
        for i, split in enumerate(splits):
            chunk_size = agent.splitter_settings['chunk_size']
            chunk_overlap = agent.splitter_settings['chunk_overlap']
            
            content_lines = split.page_content.split('\n')
            start_line = i * (chunk_size - chunk_overlap)
            end_line = start_line + len(content_lines)
            
            split.metadata.update({
                'chunk_number': i + 1,
                'total_chunks': len(splits),
                'start_line': start_line,
                'end_line': end_line,
                'preview': next((line.strip() for line in content_lines if line.strip()), '')[:100],
                'repo_name': repo_dir
            })

        return splits
    except Exception as e:
        console.print(f"[red]Error processing document chunk: {str(e)}")
        return None

async def process_documents_parallel(documents: List[Any], repo_dir: str, agent: Any, index_path: str) -> Tuple[Chroma, List[Any]]:
    """Process documents in parallel with improved performance and checkpointing"""
    try:
        # Print initial document stats
        total_content_size = sum(len(doc.page_content) for doc in documents)
        console.print(f"\n[blue]Processing {len(documents)} documents for {repo_dir}")
        console.print(f"[blue]Total content size: {total_content_size / 1024 / 1024:.2f} MB")

        # Clean up existing index with proper error handling
        if os.path.exists(index_path):
            try:
                # Ensure all files are writable before removal
                for root, dirs, files in os.walk(index_path):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        os.chmod(dir_path, 0o755)  # rwxr-xr-x
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        os.chmod(file_path, 0o644)  # rw-r--r--
                
                shutil.rmtree(index_path)
                console.print(f"[yellow]Cleaned up existing index: {index_path}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not clean up index: {str(e)}")
                # Try to remove individual files
                try:
                    for root, dirs, files in os.walk(index_path):
                        for file_name in files:
                            file_path = os.path.join(root, file_name)
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass
                except Exception:
                    pass

        # Create fresh index directory with proper permissions
        os.makedirs(index_path, mode=0o755, exist_ok=True)

        # Determine chunk size based on content
        if total_content_size > 10 * 1024 * 1024:  # 10MB
            chunk_size = 1000
            overlap = 100
        else:
            chunk_size = 2000
            overlap = 200

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Split documents
        splits = []
        for doc in documents:
            try:
                doc_splits = text_splitter.split_documents([doc])
                splits.extend(doc_splits)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to split document {doc.metadata.get('file_path', 'unknown')}: {str(e)}")
                continue

        if not splits:
            console.print(f"[red]Error: No valid splits generated for {repo_dir}")
            return None, []

        console.print(f"[blue]Generated {len(splits)} splits")

        # Create vector store with retries and proper error handling
        max_retries = 3
        retry_delay = 2
        
        for retry in range(max_retries):
            try:
                # Create vector store with explicit persist
                vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=agent.embeddings,
                    persist_directory=index_path,
                    collection_metadata=agent.collection_metadata
                )
                
                # Force persist to ensure data is written
                vector_store.persist()
                
                # Verify vector store was created
                if vector_store._collection is None:
                    raise ValueError("Vector store collection is None")
                
                # Verify documents were added
                collection = vector_store._collection
                stored_docs = collection.get()
                if not stored_docs['ids']:
                    raise ValueError("No documents found in vector store after creation")
                
                console.print(f"[green]Successfully created vector store with {len(stored_docs['ids'])} documents")
                return vector_store, splits

            except Exception as e:
                if retry < max_retries - 1:
                    console.print(f"[yellow]Retry {retry + 1}/{max_retries}: Vector store creation failed: {str(e)}")
                    # Clean up failed attempt
                    try:
                        if os.path.exists(index_path):
                            shutil.rmtree(index_path)
                        os.makedirs(index_path, mode=0o755, exist_ok=True)
                    except Exception as cleanup_error:
                        console.print(f"[yellow]Warning: Cleanup between retries failed: {str(cleanup_error)}")
                    await asyncio.sleep(retry_delay)
                else:
                    console.print(f"[red]Failed to create vector store after {max_retries} attempts: {str(e)}")
                    return None, []

    except Exception as e:
        console.print(f"[red]Error in document processing: {str(e)}")
        return None, [] 