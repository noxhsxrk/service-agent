from fastapi import FastAPI, Request, Form, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn
import os
from pathlib import Path
from repo_agent import RepoAgent
from dotenv import load_dotenv
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

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
    
    if agent is None:
        await init_agent()
        
    if agent is None:
        return {
            "answer": "Error: Failed to initialize Ollama agent. Please check your Ollama configuration.",
            "sources": []
        }
    
    try:
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
                yield f"data: {json.dumps({'stage': 'start', 'current': 0, 'total': 100, 'message': 'Starting reindexing process...'})}\n\n"
                
                # Initialize agent with the progress callback
                global agent
                if agent is not None:
                    agent.progress_callback = progress_callback
                
                # Start reindexing in a separate task
                reindex_task = asyncio.create_task(init_agent(force_reindex=True))
                
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
                    yield f"data: {json.dumps({'stage': 'complete', 'current': 100, 'total': 100, 'message': 'Reindexing completed successfully'})}\n\n"
                
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
    
    if agent is None:
        if os.path.exists("./repo_index"):
            agent = RepoAgent(REPOS_PATH, progress_callback=broadcast_progress)
            await init_agent()
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Vector store not initialized. Please reindex first."}
            )
    
    try:
        # Get all repositories data
        repo_data = []
        total_chunks = 0
        total_documents = 0
        
        for repo_name, vector_store in agent.vector_stores.items():
            if vector_store is None:
                continue
                
            collection = vector_store._collection
            documents = collection.get()
            
            # Group documents by source file for this repository
            doc_groups = {}
            for i, doc in enumerate(documents['metadatas']):
                file_path = doc.get('source', 'unknown')
                if file_path not in doc_groups:
                    doc_groups[file_path] = {
                        'file_path': file_path,
                        'chunks': 0,
                        'last_modified': doc.get('last_modified', None) or doc.get('mtime', None)
                    }
                doc_groups[file_path]['chunks'] += 1
            
            # Add repository data
            repo_data.append({
                'name': repo_name,
                'total_files': len(doc_groups),
                'total_chunks': len(documents['ids']),
                'documents': list(doc_groups.values()),
                'last_indexed': agent._load_repo_metadata(repo_name).get('last_indexed', 'Never')
            })
            
            total_chunks += len(documents['ids'])
            total_documents += len(doc_groups)
        
        # Get metadata file info for the last update time
        metadata_file = Path("./repo_index/file_metadata.json")
        last_updated = datetime.fromtimestamp(metadata_file.stat().st_mtime).isoformat() if metadata_file.exists() else None
        
        return {
            "total_repositories": len(repo_data),
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "last_updated": last_updated,
            "repositories": repo_data
        }
    except Exception as e:
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
            if any(metadata.get('source') == path for metadata in docs['metadatas']):
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
            if metadata.get('source') == path:
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
            if any(metadata.get('source') == path for metadata in docs['metadatas']):
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
            if metadata.get('source') == path:
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
    global agent
    
    if agent is None:
        if os.path.exists("./repo_index"):
            agent = RepoAgent(REPOS_PATH, progress_callback=broadcast_progress)
            await init_agent()
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Vector store not initialized. Please reindex first."}
            )
    
    try:
        # Get list of repository names
        repo_names = list(agent.vector_stores.keys())
        return {
            "repositories": repo_names
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
        
        # Decode the repository path by replacing the separator with slashes
        repo_dir = request.repo_dir.replace('---', '/')
        
        # Check if repository exists
        repo_path = os.path.join(REPOS_PATH, repo_dir)
        if not os.path.exists(repo_path):
            raise HTTPException(status_code=404, detail=f"Repository path not found: {repo_dir}")
        
        if not agent._is_git_repo(repo_path):
            raise HTTPException(status_code=400, detail=f"Not a valid Git repository: {repo_dir}")
        
        # Force reindex just this repository
        index_path = agent._get_repo_index_path(repo_dir)
        if repo_dir in agent.vector_stores:
            del agent.vector_stores[repo_dir]
        
        await broadcast_progress({
            'stage': 'scanning',
            'current': 10,
            'total': 100,
            'message': 'Scanning repository files...'
        })
        
        # Process repository files
        documents = []
        file_count = 0
        total_files = sum(1 for root, _, files in os.walk(repo_path) 
                         if not any(excluded in root.split(os.sep) for excluded in agent.exclude_dirs)
                         for file in files if agent._is_allowed_file(os.path.join(root, file)))
        
        for root, _, files in os.walk(repo_path):
            if any(excluded in root.split(os.sep) for excluded in agent.exclude_dirs):
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if agent._is_allowed_file(file_path):
                        loader = TextLoader(file_path)
                        file_docs = loader.load()
                        file_count += 1
                        
                        # Report progress for file processing
                        progress = int((file_count / total_files) * 40) + 10  # 10-50%
                        await broadcast_progress({
                            'stage': 'processing',
                            'current': progress,
                            'total': 100,
                            'message': f'Processing file {file_count}/{total_files}: {os.path.relpath(file_path, repo_path)}'
                        })
                        
                        # Add file path and extension to metadata
                        for doc in file_docs:
                            doc.metadata['file_path'] = os.path.relpath(file_path, repo_path)
                            doc.metadata['file_extension'] = os.path.splitext(file_path)[1]
                            doc.metadata['repo_name'] = repo_dir
                        
                        documents.extend(file_docs)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not load {file_path}: {str(e)}")

        if documents:
            await broadcast_progress({
                'stage': 'splitting',
                'current': 50,
                'total': 100,
                'message': 'Splitting documents into chunks...'
            })
            
            # Split documents with enhanced metadata
            text_splitter = RecursiveCharacterTextSplitter(
                **agent.splitter_settings
            )
            splits = text_splitter.split_documents(documents)

            # Add chunk context to metadata
            for i, split in enumerate(splits):
                chunk_size = agent.splitter_settings['chunk_size']
                chunk_overlap = agent.splitter_settings['chunk_overlap']
                
                # Calculate line numbers based on content
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
                
                # Report progress for chunk processing
                if i % 10 == 0:  # Update every 10 chunks to avoid too frequent updates
                    progress = int((i / len(splits) * 30)) + 50  # 50-80%
                    await broadcast_progress({
                        'stage': 'embedding',
                        'current': progress,
                        'total': 100,
                        'message': f'Processing chunk {i + 1}/{len(splits)}'
                    })

            await broadcast_progress({
                'stage': 'saving',
                'current': 80,
                'total': 100,
                'message': 'Creating vector store...'
            })

            # Create vector store for repository
            agent.vector_stores[repo_dir] = Chroma.from_documents(
                documents=splits,
                persist_directory=index_path,
                embedding=agent.embeddings,
                collection_metadata=agent.collection_metadata
            )

            # Save repository metadata with current commit hash
            current_commit = agent._get_current_commit_hash(repo_path)
            if current_commit:
                metadata = {
                    'last_commit_hash': current_commit,
                    'last_indexed': datetime.now().isoformat(),
                    'total_documents': len(documents),
                    'total_chunks': len(splits)
                }
                agent._save_repo_metadata(repo_dir, metadata)

            await broadcast_progress({
                'stage': 'complete',
                'current': 100,
                'total': 100,
                'message': f'Repository {repo_dir} reindexed successfully'
            })

        return {"status": "success", "message": f"Repository {repo_dir} reindexed successfully"}
    except Exception as e:
        console.print(f"[red]Error in reindex_repo: {str(e)}")
        await broadcast_progress({
            'stage': 'error',
            'current': 0,
            'total': 100,
            'message': str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True) 