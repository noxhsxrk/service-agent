from fastapi import FastAPI, Request, Form, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn
import os
from pathlib import Path
from repo_agent import RepoAgent
from dotenv import load_dotenv, set_key
import asyncio
from typing import List, Set
import json
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from rich.console import Console
from datetime import datetime
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from pydantic import BaseModel
from tkinter import filedialog, Tk
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from document_processor import process_documents_parallel, process_document_chunk
import shutil
import time

app = FastAPI()
load_dotenv()

# Initialize Rich console
console = Console()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize RepoAgent
REPOS_PATH = os.getenv("REPOS_PATH", ".")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

agent = None
active_websockets: Set[WebSocket] = set()

async def broadcast_progress(message: dict):
    """Broadcast progress to all connected WebSocket clients"""
    if not active_websockets:
        console.print("[yellow]Warning: No active WebSocket connections")
        return
        
    disconnected = set()
    for websocket in active_websockets:
        try:
            await websocket.send_json(message)
        except Exception as e:
            console.print(f"[red]Error broadcasting to WebSocket: {str(e)}")
            disconnected.add(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        active_websockets.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    console.print(f"[green]WebSocket client connected. Total clients: {len(active_websockets)}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "stage": "connected",
            "current": 0,
            "total": 100,
            "message": "WebSocket connected successfully"
        })
        
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        console.print(f"[yellow]WebSocket client disconnected: {str(e)}")
    finally:
        active_websockets.remove(websocket)
        console.print(f"[yellow]WebSocket client removed. Total clients: {len(active_websockets)}")

async def init_agent(force_reindex: bool = False):
    global agent
    if agent is None:
        agent = RepoAgent(
            repos_path=REPOS_PATH,
            ollama_base_url=OLLAMA_BASE_URL,
            ollama_model=OLLAMA_MODEL,
            progress_callback=broadcast_progress
        )
    if agent is not None:
        # Initialize index if needed
        await agent.index_repositories(force_reindex=force_reindex)
    return agent

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "repos_path": REPOS_PATH}
    )

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global agent
    
    try:
        # Initialize agent if not already initialized
        if agent is None:
            agent = RepoAgent(
                repos_path=REPOS_PATH,
                ollama_base_url=OLLAMA_BASE_URL,
                ollama_model=OLLAMA_MODEL,
                progress_callback=broadcast_progress
            )
            
            # Only check for index existence, don't create it
            if not os.path.exists("./repo_index"):
                return {
                    "answer": "Repository index not found. Please initialize the index first using the 'Check Changes' or 'Reindex' button in the Vector Store page.",
                    "sources": []
                }
        
        # Parse the query to check if it's repository-specific
        repo_name, actual_question = agent._parse_query(question)
        
        # Broadcast progress update for embedding generation
        await broadcast_progress({
            "stage": "embedding",
            "current": 0,
            "total": 100,
            "message": "Generating question embedding..."
        })
        
        await broadcast_progress({
            "stage": "search",
            "current": 50,
            "total": 100,
            "message": "Searching for relevant documents..."
        })

        # Get the answer using the agent's ask method
        answer = agent.ask(question)
        
        await broadcast_progress({
            "stage": "complete",
            "current": 100,
            "total": 100,
            "message": "Answer ready!"
        })

        # Extract sources from the answer if they're included
        sources = []
        if "[Sources:" in answer:
            answer_parts = answer.split("[Sources:")
            main_answer = answer_parts[0].strip()
            sources_text = answer_parts[1].strip("]").strip()
            sources = [s.strip().strip('`') for s in sources_text.split('\n') if s.strip()]
            answer = main_answer
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        await broadcast_progress({
            "stage": "error",
            "current": 0,
            "total": 100,
            "message": str(e)
        })
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

@app.get("/reindex")
async def reindex():
    try:
        # Create a queue for progress updates
        progress_queue = asyncio.Queue()
        
        # Create a regular function for progress callback
        def progress_callback(data: dict):
            # Create a task to put the data in the queue
            loop = asyncio.get_event_loop()
            loop.create_task(progress_queue.put(data))
        
        async def progress_generator():
            try:
                # Send initial progress
                yield f"data: {json.dumps({'stage': 'start', 'current': 0, 'total': 100, 'message': 'Starting indexing process...'})}\n\n"
                
                # Initialize agent with the progress callback
                global agent
                if agent is None:
                    agent = RepoAgent(
                        repos_path=REPOS_PATH,
                        ollama_base_url=OLLAMA_BASE_URL,
                        ollama_model=OLLAMA_MODEL,
                        progress_callback=progress_callback
                    )
                else:
                    agent.progress_callback = progress_callback
                
                # Get list of repositories
                repo_dirs = agent._get_repo_dirs()
                if not repo_dirs:
                    yield f"data: {json.dumps({'stage': 'error', 'current': 0, 'total': 100, 'message': 'No Git repositories found'})}\n\n"
                    return
                
                # Clean up existing indexes
                repo_index_root = Path("./repo_index")
                if repo_index_root.exists():
                    try:
                        shutil.rmtree(repo_index_root)
                        console.print("[yellow]Cleaned up existing indexes")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not clean up indexes: {str(e)}")
                
                # Create fresh index directory
                os.makedirs(repo_index_root, exist_ok=True)
                
                # Clear existing vector stores
                agent.vector_stores = {}
                
                # Start reindexing in a separate task
                reindex_task = asyncio.create_task(agent.index_repositories(force_reindex=True))
                
                try:
                    # Keep yielding progress updates until reindexing is complete
                    while not reindex_task.done():
                        try:
                            # Wait for a progress update with timeout
                            data = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                            yield f"data: {json.dumps(data)}\n\n"
                        except asyncio.TimeoutError:
                            # No progress update received, check if reindex is done
                            continue
                        except asyncio.CancelledError:
                            reindex_task.cancel()
                            break
                    
                    # Wait for reindex task to complete and propagate any exceptions
                    await reindex_task
                    
                    # Send completion message
                    yield f"data: {json.dumps({'stage': 'complete', 'current': 100, 'total': 100, 'message': 'Indexing completed successfully'})}\n\n"
                
                except Exception as e:
                    # Cancel reindex task if there's an error
                    if not reindex_task.done():
                        reindex_task.cancel()
                    yield f"data: {json.dumps({'stage': 'error', 'current': 0, 'total': 100, 'message': str(e)})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'stage': 'error', 'current': 0, 'total': 100, 'message': str(e)})}\n\n"

        return StreamingResponse(
            progress_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error during reindexing: {str(e)}"}
        )

@app.get("/vector-store", response_class=HTMLResponse)
async def vector_store_ui(request: Request):
    return templates.TemplateResponse(
        "vector_store.html",
        {"request": request, "repos_path": REPOS_PATH}
    )

@app.get("/vector-store-data")
async def get_vector_store_data():
    global agent
    
    try:
        # Initialize agent without forcing reindex
        if agent is None:
            agent = RepoAgent(
                repos_path=REPOS_PATH,
                ollama_base_url=OLLAMA_BASE_URL,
                ollama_model=OLLAMA_MODEL,
                progress_callback=broadcast_progress
            )
        
        # Check if index exists
        if not os.path.exists("./repo_index"):
            return {
                "total_repositories": 0,
                "total_documents": 0,
                "total_chunks": 0,
                "last_updated": None,
                "repositories": [],
                "needs_indexing": True
            }
            
        # Get all repositories data
        repo_data = []
        total_chunks = 0
        total_documents = 0
        
        for repo_name, vector_store in agent.vector_stores.items():
            if vector_store is None:
                continue
                
            try:
                collection = vector_store._collection
                if collection is None:
                    console.print(f"[yellow]Warning: Null collection for {repo_name}")
                    continue
                    
                documents = collection.get()
                if not documents or not documents.get('metadatas'):
                    console.print(f"[yellow]Warning: No documents found for {repo_name}")
                    continue
                
                # Group documents by source file for this repository
                doc_groups = {}
                git_log_entries = []
                
                for i, metadata in enumerate(documents['metadatas']):
                    if not metadata:
                        continue
                        
                    file_path = metadata.get('file_path')
                    if not file_path:
                        continue
                        
                    doc_type = metadata.get('type', 'file')
                    
                    if doc_type == 'git_log':
                        commit_hash = metadata.get('commit_hash')
                        author = metadata.get('author')
                        date = metadata.get('date')
                        if commit_hash and author and date:
                            git_log_entries.append({
                                'commit_hash': commit_hash,
                                'author': author,
                                'date': date,
                                'chunks': 1
                            })
                    else:
                        if file_path not in doc_groups:
                            doc_groups[file_path] = {
                                'file_path': file_path,
                                'chunks': 0,
                                'last_modified': metadata.get('last_modified') or metadata.get('mtime') or 'Unknown'
                            }
                        doc_groups[file_path]['chunks'] += 1
                
                # Add repository data
                repo_metadata = agent._load_repo_metadata(repo_name)
                repo_entry = {
                    'name': repo_name,
                    'total_files': len(doc_groups),
                    'total_chunks': len(documents['ids']) if documents.get('ids') else 0,
                    'documents': list(doc_groups.values()),
                    'git_log': git_log_entries,
                    'last_indexed': repo_metadata.get('last_indexed', 'Never')
                }
                
                repo_data.append(repo_entry)
                total_chunks += repo_entry['total_chunks']
                total_documents += repo_entry['total_files']
                
            except Exception as e:
                console.print(f"[yellow]Warning: Error processing repository {repo_name}: {str(e)}")
                continue
        
        # Get metadata file info for the last update time
        try:
            metadata_file = Path("./repo_index/file_metadata.json")
            last_updated = datetime.fromtimestamp(metadata_file.stat().st_mtime).isoformat() if metadata_file.exists() else None
        except Exception:
            last_updated = None
        
        return {
            "total_repositories": len(repo_data),
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "last_updated": last_updated,
            "repositories": repo_data,
            "needs_indexing": False
        }
    except Exception as e:
        console.print(f"[red]Error fetching vector store data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching vector store data: {str(e)}"}
        )

@app.get("/vector-store-document")
async def get_vector_store_document(path: str):
    global agent
    
    try:
        # Initialize agent if not already initialized
        if agent is None:
            if os.path.exists("./repo_index"):
                agent = RepoAgent(REPOS_PATH, progress_callback=broadcast_progress)
                await init_agent()
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Vector store not initialized. Please reindex first."}
                )

        if agent is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Failed to initialize agent. Please check your configuration."}
            )

        # Find which repository contains this document
        repo_found = None
        collection = None
        for repo_name, vector_store in agent.vector_stores.items():
            if vector_store is None:
                continue
            # Get the collection's documents
            docs = vector_store._collection.get()
            # Check if the document exists in this repository
            if any(metadata.get('file_path') == path for metadata in docs['metadatas']):
                repo_found = repo_name
                collection = vector_store._collection
                break
        
        if not collection:
            return JSONResponse(
                status_code=404,
                content={"error": "Document not found in any repository"}
            )
        
        # Get all chunks for the specified document
        documents = collection.get()
        
        chunks = []
        for i, metadata in enumerate(documents['metadatas']):
            if metadata.get('file_path') == path:
                chunks.append({
                    'id': documents['ids'][i],
                    'content': documents['documents'][i],
                    'metadata': metadata
                })
        
        if not chunks:
            return JSONResponse(
                status_code=404,
                content={"error": "Document not found in vector store"}
            )
        
        return {
            "file_path": path,
            "repository": repo_found,
            "chunks": chunks
        }
    except Exception as e:
        console.print(f"[red]Error in get_vector_store_document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching document details: {str(e)}"}
        )

@app.delete("/vector-store-document")
async def delete_vector_store_document(path: str):
    global agent
    
    if agent is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Vector store not initialized"}
        )
    
    try:
        # Find which repository contains this document
        repo_found = None
        collection = None
        for repo_name, vector_store in agent.vector_stores.items():
            if vector_store is None:
                continue
            # Get the collection's documents
            docs = vector_store._collection.get()
            # Check if the document exists in this repository
            if any(metadata.get('file_path') == path for metadata in docs['metadatas']):
                repo_found = repo_name
                collection = vector_store._collection
                break
        
        if not collection:
            return JSONResponse(
                status_code=404,
                content={"error": "Document not found in any repository"}
            )
        
        # Get all document IDs for the specified path
        documents = collection.get()
        
        ids_to_delete = []
        for i, metadata in enumerate(documents['metadatas']):
            if metadata.get('file_path') == path:
                ids_to_delete.append(documents['ids'][i])
        
        if not ids_to_delete:
            return JSONResponse(
                status_code=404,
                content={"error": "Document not found in vector store"}
            )
        
        # Delete the documents
        collection.delete(ids_to_delete)
        
        return {
            "status": "success", 
            "message": f"Deleted {len(ids_to_delete)} chunks from {path} in repository {repo_found}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error deleting document: {str(e)}"}
        )

@app.get("/repositories")
async def get_repositories():
    try:
        # Create a temporary RepoAgent just for listing repositories
        temp_agent = RepoAgent(
            repos_path=REPOS_PATH,
            ollama_base_url=OLLAMA_BASE_URL,
            ollama_model=OLLAMA_MODEL,
            progress_callback=None,  # No need for progress callback
            load_embeddings=False  # Skip loading embeddings and vector stores
        )
        
        # Get list of repository directories without loading vector stores
        repo_dirs = temp_agent._get_repo_dirs()
        
        return {
            "repositories": repo_dirs
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching repositories: {str(e)}"}
        )

@app.post("/check-changes")
async def check_changes():
    """Check for changes in repositories without forcing reindex"""
    try:
        # Initialize agent if needed
        global agent
        if agent is None:
            agent = RepoAgent(REPOS_PATH, progress_callback=broadcast_progress)
        
        changes = []
        repo_dirs = agent._get_repo_dirs()
        
        for repo_dir in repo_dirs:
            repo_path = os.path.join(REPOS_PATH, repo_dir)
            current_commit = agent._get_current_commit_hash(repo_path)
            metadata = agent._load_repo_metadata(repo_dir)
            last_commit = metadata.get('last_commit_hash')
            last_indexed = metadata.get('last_indexed', 'Never')
            
            status = {
                'repo': repo_dir,
                'current_commit': current_commit[:8] if current_commit else None,
                'last_commit': last_commit[:8] if last_commit else None,
                'last_indexed': last_indexed,
                'status': 'unknown'
            }
            
            if not current_commit:
                status['status'] = 'error'
                status['message'] = 'Could not get current commit hash'
            elif not last_commit:
                status['status'] = 'new'
                status['message'] = 'Not indexed yet'
            elif current_commit != last_commit:
                status['status'] = 'changed'
                status['message'] = 'Repository has changed'
            else:
                status['status'] = 'current'
                status['message'] = 'Up to date'
            
            changes.append(status)
        
        return {
            "status": "success",
            "changes": changes
        }
    except Exception as e:
        console.print(f"[red]Error in check_changes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex-all")
async def reindex_all():
    """Force reindex all repositories"""
    try:
        global agent
        if agent is None:
            agent = RepoAgent(
                repos_path=REPOS_PATH,
                ollama_base_url=OLLAMA_BASE_URL,
                ollama_model=OLLAMA_MODEL,
                progress_callback=broadcast_progress
            )
        
        await agent.index_repositories(force_reindex=True)
        return {"status": "success"}
    except Exception as e:
        console.print(f"[red]Error in reindex_all: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ReindexRequest(BaseModel):
    repo_dir: str

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
                'preview': next((line.strip() for line in content_lines if line.strip()), '')[:100]
            })

        # Create vector store for this chunk
        chunk_store = Chroma.from_documents(
            documents=splits,
            persist_directory=f"{index_path}_temp_{multiprocessing.current_process().name}",
            embedding=agent.embeddings,
            collection_metadata=agent.collection_metadata
        )
        
        return splits, chunk_store
    except Exception as e:
        console.print(f"[red]Error processing document chunk: {str(e)}")
        return None, None

async def process_documents_parallel(documents, repo_dir, agent, index_path):
    # Determine number of chunks based on CPU cores
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(documents) // num_cores)
    document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
    
    all_splits = []
    chunk_stores = []
    
    # Process chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Create tasks for parallel processing
        tasks = []
        for chunk in document_chunks:
            task = asyncio.create_task(process_document_chunk(chunk, repo_dir, agent, index_path, agent.progress_callback))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        for splits, chunk_store in results:
            if splits and chunk_store:
                all_splits.extend(splits)
                chunk_stores.append(chunk_store)
    
    # Merge all chunk stores into final vector store
    final_store = Chroma.from_documents(
        documents=all_splits,
        persist_directory=index_path,
        embedding=agent.embeddings,
        collection_metadata=agent.collection_metadata
    )
    
    # Clean up temporary stores
    for chunk_store in chunk_stores:
        chunk_store.delete_collection()
    
    return final_store, all_splits

@app.post("/reindex-repo")
async def reindex_repo(request: ReindexRequest):
    """Reindex a specific repository"""
    try:
        global agent
        if agent is None:
            agent = RepoAgent(
                repos_path=REPOS_PATH,
                ollama_base_url=OLLAMA_BASE_URL,
                ollama_model=OLLAMA_MODEL,
                progress_callback=broadcast_progress
            )
        
        # Decode the repository path
        repo_dir = request.repo_dir.replace('---', '/')
        
        # Check if repository exists
        repo_path = os.path.join(REPOS_PATH, repo_dir)
        if not os.path.exists(repo_path):
            raise HTTPException(status_code=404, detail=f"Repository path not found: {repo_dir}")
        
        if not agent._is_git_repo(repo_path):
            raise HTTPException(status_code=400, detail=f"Not a valid Git repository: {repo_dir}")
        
        # Force reindex just this repository
        index_path = agent._get_repo_index_path(repo_dir)
        
        # Clean up any existing index
        if os.path.exists(index_path):
            try:
                # Ensure all files are writable before removal
                for root, dirs, files in os.walk(index_path):
                    for dir_name in dirs:
                        try:
                            dir_path = os.path.join(root, dir_name)
                            os.chmod(dir_path, 0o755)  # rwxr-xr-x
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not change directory permissions: {str(e)}")
                    for file_name in files:
                        try:
                            file_path = os.path.join(root, file_name)
                            os.chmod(file_path, 0o644)  # rw-r--r--
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not change file permissions: {str(e)}")
                
                shutil.rmtree(index_path)
                console.print(f"[yellow]Cleaned up existing index for {repo_dir}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not clean up existing index: {str(e)}")
                # Try to remove individual files
                try:
                    for root, dirs, files in os.walk(index_path):
                        for file_name in files:
                            try:
                                file_path = os.path.join(root, file_name)
                                os.remove(file_path)
                            except Exception:
                                pass
                except Exception:
                    pass
        
        # Remove from vector stores if exists
        if repo_dir in agent.vector_stores:
            del agent.vector_stores[repo_dir]

        async def generate_progress():
            start_time = time.time()
            try:
                # Initial progress
                data = {
                    "stage": "start",
                    "current": 0,
                    "total": 100,
                    "message": f"Starting {'re' if os.path.exists(index_path) else ''}index of {repo_dir}",
                    "elapsed": 0
                }
                yield f"data: {json.dumps(data)}\n\n"
                
                # Count total files
                total_files = 0
                files_to_process = []
                
                # First pass: count files and build list
                for root, _, files in os.walk(repo_path):
                    if not any(excluded in root.split(os.sep) for excluded in agent.exclude_dirs):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if agent._is_allowed_file(file_path):
                                rel_path = os.path.relpath(file_path, repo_path)
                                total_files += 1
                                files_to_process.append((file_path, rel_path))
                
                if total_files == 0:
                    data = {
                        "stage": "error",
                        "current": 0,
                        "total": 100,
                        "message": "No files to process",
                        "elapsed": time.time() - start_time
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    return

                # Process files
                file_count = 0
                
                # Process the files we found
                for file_path, rel_path in files_to_process:
                    try:
                        file_count += 1
                        elapsed = time.time() - start_time
                        
                        # Progress update
                        progress = int((file_count / total_files) * 40) + 10
                        data = {
                            "stage": "processing",
                            "current": progress,
                            "total": 100,
                            "message": f"Processing file {file_count}/{total_files}: {rel_path}",
                            "elapsed": elapsed
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                        # Add metadata
                        for doc in file_docs:
                            doc.metadata.update({
                                'file_path': rel_path,
                                'file_extension': os.path.splitext(file_path)[1],
                                'repo_name': repo_dir,
                                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                                'file_size': os.path.getsize(file_path)
                            })
                        
                        documents.extend(file_docs)
                        
                    except Exception as e:
                        elapsed = time.time() - start_time
                        data = {
                            "stage": "warning",
                            "current": progress,
                            "total": 100,
                            "message": f"Warning: Could not load {rel_path}: {str(e)}",
                            "elapsed": elapsed
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                if documents:
                    data = {
                        "stage": "embedding",
                        "current": 50,
                        "total": 100,
                        "message": "Creating embeddings and vector store..."
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    try:
                        # Create the index directory if it doesn't exist
                        os.makedirs(index_path, exist_ok=True, mode=0o755)
                        
                        # Process documents in parallel
                        vector_store, splits = await process_documents_parallel(documents, repo_dir, agent, index_path)
                        
                        if vector_store and splits:
                            # Verify vector store is valid
                            try:
                                collection = vector_store._collection
                                if collection is None:
                                    raise ValueError("Vector store collection is None")
                                
                                stored_docs = collection.get()
                                if not stored_docs['ids']:
                                    raise ValueError("No documents found in vector store after creation")
                                
                                # Store in agent's vector stores
                                agent.vector_stores[repo_dir] = vector_store

                                # Save metadata
                                current_commit = agent._get_current_commit_hash(repo_path)
                                if current_commit:
                                    metadata = {
                                        'last_commit_hash': current_commit,
                                        'last_indexed': datetime.now().isoformat(),
                                        'total_documents': len(documents),
                                        'total_chunks': len(splits)
                                    }
                                    agent._save_repo_metadata(repo_dir, metadata)

                                data = {
                                    "stage": "complete",
                                    "current": 100,
                                    "total": 100,
                                    "message": f"Repository {repo_dir} indexed successfully with {len(stored_docs['ids'])} chunks"
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                            except Exception as e:
                                data = {
                                    "stage": "error",
                                    "current": 0,
                                    "total": 100,
                                    "message": f"Error verifying vector store: {str(e)}"
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                        else:
                            data = {
                                "stage": "error",
                                "current": 0,
                                "total": 100,
                                "message": f"Failed to create vector store for {repo_dir}"
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                    except Exception as e:
                        data = {
                            "stage": "error",
                            "current": 0,
                            "total": 100,
                            "message": f"Error creating vector store: {str(e)}"
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                else:
                    data = {
                        "stage": "error",
                        "current": 0,
                        "total": 100,
                        "message": f"No documents found in {repo_dir}"
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    
            except Exception as e:
                data = {
                    "stage": "error",
                    "current": 0,
                    "total": 100,
                    "message": str(e),
                    "elapsed": time.time() - start_time
                }
                yield f"data: {json.dumps(data)}\n\n"

        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        console.print(f"[red]Error in reindex_repo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ReposPathUpdate(BaseModel):
    repos_path: str

@app.post("/api/pick-folder")
async def pick_folder():
    try:
        # Create and hide the Tkinter root window
        root = Tk()
        root.withdraw()
        
        # Open the folder picker dialog
        selected_path = filedialog.askdirectory()
        root.destroy()
        
        if selected_path:
            return {"success": True, "selected_path": selected_path}
        else:
            return {"success": False, "error": "No folder selected"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/update-repos-path")
async def update_repos_path(path_update: ReposPathUpdate):
    try:
        if not path_update.repos_path:
            return {"success": False, "error": "No path provided"}
        
        # Update the .env file
        env_path = '.env'
        set_key(env_path, 'REPOS_PATH', path_update.repos_path)
        
        # Update environment variables directly
        global REPOS_PATH
        REPOS_PATH = path_update.repos_path
        os.environ["REPOS_PATH"] = path_update.repos_path
        
        # Clear existing agent but don't reinitialize
        global agent
        agent = None
        
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8001, reload=True) 