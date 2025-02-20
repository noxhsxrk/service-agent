import os
import json
import hashlib
from datetime import datetime
import typer
from typing import Optional, Callable, Dict, Any, Literal
from pathlib import Path
from rich.console import Console
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import asyncio
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from document_processor import process_documents_parallel, process_document_chunk
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from indexing_manager import IndexingManager

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# Initialize rich console for better output
console = Console()
app = typer.Typer()

class RepoAgent:
    # Common directories to exclude
    exclude_dirs = {
        'node_modules',
        'venv',
        'env',
        '.env',
        'lib',
        'libs',
        'vendor',
        'dist',
        'build',
        '.git',
        '__pycache__',
        '.pytest_cache',
        '.next',
        'target'  # for Rust projects
    }

    # File extensions to include (text-based files only)
    allowed_extensions = {
        # Programming languages
        '.py',    # Python
        '.js',    # JavaScript
        '.jsx',   # React JSX
        '.ts',    # TypeScript
        '.tsx',   # React TypeScript
        '.java',  # Java
        '.go',    # Go
        '.rs',    # Rust
        '.cpp',   # C++
        '.c',     # C
        '.h',     # C/C++ headers
        '.hpp',   # C++ headers
        '.cs',    # C#
        '.rb',    # Ruby
        '.php',   # PHP
        '.scala', # Scala
        '.swift', # Swift
        '.kt',    # Kotlin
        '.r',     # R
        '.sh',    # Shell scripts
        '.bash',  # Bash scripts
        
        # Web development
        '.html',  # HTML
        '.htm',   # HTML
        '.css',   # CSS
        '.scss',  # SASS
        '.less',  # LESS
        '.vue',   # Vue.js
        '.svelte',# Svelte
        
        # Configuration files
        '.json',  # JSON
        '.yml',   # YAML
        '.yaml',  # YAML
        '.xml',   # XML
        '.toml',  # TOML
        '.ini',   # INI
        '.conf',  # Config files
        '.env',   # Environment files
        
        # Documentation
        '.md',    # Markdown
        '.rst',   # reStructuredText
        '.txt',   # Plain text
        '.tex',   # LaTeX
        
        # Other text-based files
        '.sql',   # SQL
        '.graphql',# GraphQL
        '.proto', # Protocol Bufferst
    }

    # Binary and non-text files to explicitly exclude
    exclude_extensions = {
        # Binary files
        '.exe', '.dll', '.so', '.dylib',  # Executables and libraries
        '.zip', '.tar', '.gz', '.7z', '.rar',  # Archives
        '.pyc', '.pyo', '.pyd',  # Python bytecode
        '.class', '.jar',  # Java bytecode
        
        # Media files
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',  # Images
        '.svg', '.ico',  # Vector images and icons
        '.mp3', '.wav', '.ogg', '.m4a',  # Audio
        '.mp4', '.avi', '.mov', '.wmv',  # Video
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',  # Documents
        '.ppt', '.pptx',  # Presentations
        
        # Font files
        '.ttf', '.otf', '.woff', '.woff2', '.eot',
        
        # Database files
        '.db', '.sqlite', '.sqlite3',
        
        # Other binary files
        '.bin', '.dat', '.pkl', '.model'  # Generic binary files
    }

    def __init__(self, repos_path: str,
                 ollama_base_url: Optional[str] = None,
                 ollama_model: str = OLLAMA_MODEL,
                 progress_callback: Optional[Callable] = None,
                 force_reindex: bool = False,
                 load_embeddings: bool = True):
        self.repos_path = Path(repos_path)
        self.ollama_base_url = ollama_base_url or "http://localhost:11434"
        self.ollama_model = ollama_model
        self.force_reindex = force_reindex
        
        # Initialize Rich progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            transient=False
        )
        
        # Time tracking variables
        self.processing_start_time = None
        self.files_processed = 0
        self.total_processing_time = 0
        self.average_speed = 0
        
        # HNSW index settings for better search performance
        self.collection_metadata = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
            "hnsw:M": 32,
        }
        
        # Initialize embeddings and vector stores
        if load_embeddings:
            try:
                self.embeddings = OllamaEmbeddings(
                    base_url=self.ollama_base_url,
                    model=self.ollama_model
                )
                # Test embeddings
                test_result = self.embeddings.embed_query("test")
                if not test_result or len(test_result) == 0:
                    raise ValueError("Embeddings initialization failed - empty result")
                
                # Initialize vector stores dict
                self.vector_stores = {}
                
                # Load existing vector stores if available
                if os.path.exists("./repo_index"):
                    console.print("[yellow]Loading existing vector stores...")
                    for repo_dir in self._get_repo_dirs():
                        index_path = self._get_repo_index_path(repo_dir)
                        if os.path.exists(index_path):
                            try:
                                vector_store = Chroma(
                                    persist_directory=index_path,
                                    embedding_function=self.embeddings,
                                    collection_metadata=self.collection_metadata
                                )
                                # Verify vector store
                                if vector_store._collection is not None:
                                    stored_docs = vector_store._collection.get()
                                    if stored_docs['ids']:
                                        self.vector_stores[repo_dir] = vector_store
                                        console.print(f"[green]Loaded vector store for {repo_dir} with {len(stored_docs['ids'])} documents")
                                    else:
                                        console.print(f"[yellow]Warning: Empty vector store found for {repo_dir}")
                            except Exception as e:
                                console.print(f"[yellow]Warning: Could not load vector store for {repo_dir}: {str(e)}")
                
                self.qa_chain = None
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            except Exception as e:
                console.print(f"[red]Error initializing embeddings: {str(e)}")
                self.embeddings = None
                self.vector_stores = None
                self.qa_chain = None
                self.memory = None
        else:
            self.embeddings = None
            self.vector_stores = None
            self.qa_chain = None
            self.memory = None
        
        self.index_metadata_file = Path("./repo_index/file_metadata.json")
        self.progress_callback = progress_callback
        
        # Enhanced search parameters for better coverage
        self.search_params = {
            "k": 100,
            "score_threshold": 0.3,
            "min_sources": 20,
            "max_tokens_per_source": 1000,
            "max_combined_tokens": 6000,
        }

        # Document splitting settings optimized for code files
        self.splitter_settings = {
            "chunk_size": 1500,
            "chunk_overlap": 500,
            "length_function": len,
            "add_start_index": True,
            "separators": [
                "\n\nclass ", "\n\ninterface ", "\n\ntype ", "\n\nfunction ", "\n\ndef ",
                "\n  def ", "\n  async def ", "\n    def ", "\n    async def ",
                "\n\nconst ", "\n\nexport const ", "\n\nexport default ", "\n\nfunction ",
                "\n\n@", "\n@RestController", "\n@RequestMapping", "\n@GetMapping", "\n@PostMapping",
                "\n\nrouter.", "\n\napp.", "\n\nserver.",
                "\n\n}", "}\n\n",
                "\n\n"
            ]
        }

    async def send_progress(self, stage: str, current: int, total: int, message: str = "", extra_info: Optional[Dict] = None):
        """Enhanced progress reporting with terminal progress bar"""
        if self.progress_callback:
            try:
                data = {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": message
                }
                if extra_info:
                    data.update(extra_info)
                # Ensure we await the callback if it's a coroutine
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(data)
                else:
                    self.progress_callback(data)
                await asyncio.sleep(0)  # Give other tasks a chance to run
            except Exception as e:
                console.print(f"[yellow]Warning: Progress callback failed: {str(e)}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_index_metadata(self) -> dict:
        """Load the index metadata from file"""
        if self.index_metadata_file.exists():
            with open(self.index_metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index_metadata(self, metadata: dict):
        """Save the index metadata to file"""
        self.index_metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_file_metadata(self, file_path: str) -> dict:
        """Get file metadata including size and modification time"""
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'last_indexed': datetime.now().isoformat()
        }

    def _is_allowed_file(self, file_path: str) -> bool:
        """Check if a file should be included in the index"""
        # Get the file extension (lowercase)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Check if the file is in an excluded directory
        if any(excluded in file_path.split(os.sep) for excluded in self.exclude_dirs):
            return False
            
        # Check if the file has an allowed extension
        if ext in self.allowed_extensions:
            try:
                # Try to read the first few bytes to confirm it's text
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
                return True
            except UnicodeDecodeError:
                # If we can't read it as text, exclude it
                return False
                
        # Explicitly exclude known binary files
        if ext in self.exclude_extensions:
            return False
            
        return False

    def _is_git_repo(self, path: str) -> bool:
        """Check if a directory is a git repository"""
        git_dir = os.path.join(path, '.git')
        return os.path.exists(git_dir) and os.path.isdir(git_dir)

    def _get_repo_dirs(self) -> list[str]:
        """Get list of repository directories by recursively checking for .git folders"""
        repo_dirs = []
        
        # Walk through all subdirectories
        for root, dirs, _ in os.walk(self.repos_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            # Check if current directory is a git repo
            if self._is_git_repo(root):
                # Get relative path from workspace root
                rel_path = os.path.relpath(root, self.repos_path)
                if rel_path == '.':
                    continue  # Skip the root directory itself
                
                repo_dirs.append(rel_path)
                # Continue searching subdirectories
        
        console.print(f"[yellow]Found Git repositories: {repo_dirs}")
        return repo_dirs

    def _get_repo_index_path(self, repo_name: str) -> str:
        """Get the index directory path for a specific repository"""
        # Convert repo path to a safe directory name
        safe_name = repo_name.replace('/', '_').replace('\\', '_')
        return f"./repo_index/{safe_name}"

    def _get_current_commit_hash(self, repo_path: str) -> str:
        """Get the current commit hash of a git repository"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get commit hash for {repo_path}: {str(e)}")
            return ""

    def _load_repo_metadata(self, repo_dir: str) -> dict:
        """Load repository-specific metadata"""
        metadata_path = Path(self._get_repo_index_path(repo_dir)) / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load metadata for {repo_dir}: {str(e)}")
        return {}

    def _save_repo_metadata(self, repo_dir: str, metadata: dict):
        """Save repository-specific metadata"""
        metadata_path = Path(self._get_repo_index_path(repo_dir)) / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save metadata for {repo_dir}: {str(e)}")

    async def _check_repo_status(self, repo_dir: str, repo_path: str) -> tuple[bool, str]:
        """Check repository status and return detailed progress message"""
        # Get current commit hash
        current_commit = self._get_current_commit_hash(repo_path)
        if not current_commit:
            return True, f"⚠️  Could not get commit hash for {repo_dir}, will reindex"
            
        # Load stored metadata
        metadata = self._load_repo_metadata(repo_dir)
        last_commit = metadata.get('last_commit_hash')
        last_indexed = metadata.get('last_indexed', 'Never')
        
        # Format status message
        if not last_commit:
            return True, f"🆕 First time indexing {repo_dir}"
            
        needs_reindex = current_commit != last_commit
        if needs_reindex:
            message = (
                f"📦 Repository {repo_dir} has changed:\n"
                f"   Previous: {last_commit[:8]} ({last_indexed})\n"
                f"   Current:  {current_commit[:8]} (now)"
            )
        else:
            message = f"✅ {repo_dir} is up to date (commit: {current_commit[:8]})"
            
        return needs_reindex, message

    def _get_git_log(self, repo_path: str, max_entries: int = 1000) -> list[dict]:
        """Get Git log data for a repository"""
        try:
            import subprocess
            
            # Get git log with structured output
            cmd = [
                'git', 'log',
                '--pretty=format:{%n  "commit": "%H",%n  "author": "%an",%n  "date": "%ai",%n  "message": "%s"%n}',
                '-n', str(max_entries)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the JSON-like output into a list of dictionaries
            log_entries = []
            for entry in result.stdout.strip().split('\n{'):
                if entry:
                    # Clean up the entry to make it valid JSON
                    entry = '{' + entry if not entry.startswith('{') else entry
                    try:
                        log_entries.append(json.loads(entry))
                    except json.JSONDecodeError:
                        console.print(f"[yellow]Warning: Could not parse git log entry: {entry}")
                        continue
            
            return log_entries
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get git log for {repo_path}: {str(e)}")
            return []

    async def process_single_repository(self, repo_dir: str, repo_idx: int, total_repos: int, force_reindex: bool = False):
        """Process a single repository with terminal progress bar"""
        try:
            repo_path = os.path.join(self.repos_path, repo_dir)
            index_path = self._get_repo_index_path(repo_dir)
            
            # Initialize indexing manager for checkpointing
            indexing_manager = IndexingManager(index_path)
            
            # Reset time tracking variables
            self.processing_start_time = datetime.now()
            self.files_processed = 0
            self.total_processing_time = 0
            self.average_speed = 0
            
            # Create progress tasks
            with self.progress:
                repo_task = self.progress.add_task(
                    f"[cyan]Processing {repo_dir}",
                    total=100
                )
                
                # Check repository status
                needs_update, status_message = await self._check_repo_status(repo_dir, repo_path)
                
                if not needs_update and not (force_reindex or self.force_reindex):
                    self.progress.update(repo_task, completed=100, description=f"[green]✓ {repo_dir} is up to date")
                    return
                
                self.progress.console.print(f"[yellow]⚡ {status_message}")
                
                # Process repository files
                documents = []
                file_count = 0
                
                # Get checkpoint if exists
                processed_files, current_chunk = indexing_manager._get_checkpoint(repo_dir)
                if processed_files and not force_reindex:
                    self.progress.console.print(f"[yellow]📋 Found checkpoint - Resuming from {len(processed_files)} processed files")
                
                # Count total files first
                total_files = 0
                files_to_process = []
                for root, _, files in os.walk(repo_path):
                    if not any(excluded in root.split(os.sep) for excluded in self.exclude_dirs):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if self._is_allowed_file(file_path):
                                rel_path = os.path.relpath(file_path, repo_path)
                                # Only count files that haven't been processed or if force reindex
                                if force_reindex or rel_path not in processed_files:
                                    total_files += 1
                                    files_to_process.append((file_path, rel_path))
                
                if total_files == 0:
                    self.progress.update(repo_task, completed=100, description=f"[green]✓ {repo_dir} is already up to date")
                    return
                
                files_task = self.progress.add_task(
                    f"[blue]Processing files",
                    total=total_files
                )
                
                # Process files
                for file_path, rel_path in files_to_process:
                    try:
                        loader = TextLoader(file_path)
                        file_docs = loader.load()
                        file_count += 1
                        self.files_processed += 1
                        
                        # Calculate time estimates
                        elapsed_time = (datetime.now() - self.processing_start_time).total_seconds()
                        if elapsed_time > 0:
                            self.average_speed = self.files_processed / elapsed_time
                            estimated_remaining_seconds = (total_files - file_count) / self.average_speed if self.average_speed > 0 else 0
                            estimated_completion = self._format_time_estimate(estimated_remaining_seconds)
                            
                            # Update progress
                            self.progress.update(
                                files_task,
                                advance=1,
                                description=f"[blue]Processing files ({file_count}/{total_files}) - {estimated_completion} remaining"
                            )
                            self.progress.update(repo_task, completed=int((file_count / total_files) * 50))
                        
                        # Add file metadata
                        for doc in file_docs:
                            doc.metadata.update({
                                'file_path': rel_path,
                                'file_extension': os.path.splitext(file_path)[1],
                                'repo_name': repo_dir,
                                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                                'file_size': os.path.getsize(file_path)
                            })
                        
                        documents.extend(file_docs)
                        
                        # Save checkpoint after each file
                        processed_files.append(rel_path)
                        indexing_manager._save_checkpoint(repo_dir, processed_files, file_count)
                        
                    except Exception as e:
                        self.progress.console.print(f"[yellow]⚠️  Could not load {file_path}: {str(e)}")

                if documents:
                    doc_count = len(documents)
                    embedding_task = self.progress.add_task(
                        f"[magenta]Creating embeddings",
                        total=doc_count
                    )
                    
                    try:
                        # Try to load existing vector store first
                        existing_vector_store = None
                        try:
                            if os.path.exists(index_path):
                                existing_vector_store = Chroma(
                                    persist_directory=index_path,
                                    embedding_function=self.embeddings,
                                    collection_metadata=self.collection_metadata
                                )
                                self.progress.console.print(f"[yellow]📚 Found existing vector store for {repo_dir}")
                        except Exception as e:
                            self.progress.console.print(f"[yellow]⚠️  Could not load existing vector store: {str(e)}")
                        
                        # Process documents in parallel with improved performance
                        vector_store, splits = await process_documents_parallel(documents, repo_dir, self, index_path)
                        
                        if vector_store:
                            self.vector_stores[repo_dir] = vector_store

                            # Save repository metadata
                            current_commit = self._get_current_commit_hash(repo_path)
                            if current_commit:
                                metadata = {
                                    'last_commit_hash': current_commit,
                                    'last_indexed': datetime.now().isoformat(),
                                    'total_documents': len(documents),
                                    'total_chunks': len(splits),
                                    'index_version': '2.0'
                                }
                                self._save_repo_metadata(repo_dir, metadata)

                            self.progress.update(repo_task, completed=100, description=f"[green]✓ {repo_dir} indexed successfully")
                            self.progress.update(embedding_task, completed=doc_count)
                            
                            # Only clear checkpoint after successful vector store creation
                            indexing_manager._save_checkpoint(repo_dir, [], 0)
                        else:
                            # If vector store creation failed but we have an existing one, use it
                            if existing_vector_store:
                                self.vector_stores[repo_dir] = existing_vector_store
                                self.progress.console.print(f"[yellow]⚠️  Using existing vector store for {repo_dir}")
                            else:
                                self.progress.console.print(f"[red]❌ Failed to create vector store for {repo_dir}")
                                # Keep the checkpoint for retry
                    except Exception as e:
                        self.progress.console.print(f"[red]❌ Error creating vector store for {repo_dir}: {str(e)}")
                        # If we have an existing vector store, use it
                        if existing_vector_store:
                            self.vector_stores[repo_dir] = existing_vector_store
                            self.progress.console.print(f"[yellow]⚠️  Using existing vector store for {repo_dir}")
                else:
                    self.progress.console.print(f"[yellow]⚠️  No documents found in {repo_dir}")
            
        except Exception as e:
            self.progress.console.print(f"[red]❌ Error processing repository {repo_dir}: {str(e)}")
            raise

    async def index_repositories(self, force_reindex: bool = False):
        """Index all repositories in parallel"""
        repo_dirs = self._get_repo_dirs()
        if not repo_dirs:
            message = "❌ No Git repositories found in the specified path"
            console.print(f"[red]{message}")
            await self.send_progress("error", 0, 0, message)
            return
        
        total_repos = len(repo_dirs)
        message = f"🔍 Found {total_repos} Git repositories to process"
        console.print(f"[yellow]{message}")
        await self.send_progress("start", 0, total_repos, message)

        # Initialize vector_stores if None
        if self.vector_stores is None:
            self.vector_stores = {}
        
        # Process repositories sequentially for better stability
        for repo_idx, repo_dir in enumerate(repo_dirs, 1):
            try:
                await self.process_single_repository(repo_dir, repo_idx, total_repos, force_reindex)
                console.print(f"[green]✓ Successfully processed {repo_dir} ({repo_idx}/{total_repos})")
            except Exception as e:
                console.print(f"[red]❌ Error processing {repo_dir}: {str(e)}")
                # Continue with next repository instead of failing completely
                continue
        
        # Verify vector stores were created
        indexed_repos = list(self.vector_stores.keys())
        console.print(f"\n[blue]Indexing complete. Indexed repositories: {indexed_repos}")
        if not indexed_repos:
            console.print("[red]Warning: No repositories were successfully indexed")

    def viewDocument(self, file_path: str) -> dict:
        """Get chunks for a specific document with enhanced context"""
        chunks = []
        for repo_name, vector_store in self.vector_stores.items():
            results = vector_store.get(
                where={"file_path": file_path}
            )
            if results and results['documents']:
                # Sort chunks by their position in the file
                chunks.extend([{
                    'content': doc,
                    'metadata': meta,
                    'repo_name': repo_name
                } for doc, meta in zip(results['documents'], results['metadatas'])])
        
        # Sort chunks by their position in the file
        chunks.sort(key=lambda x: x['metadata']['chunk_number'])
        
        # Add context about surrounding chunks
        for i, chunk in enumerate(chunks):
            chunk['metadata']['has_previous'] = i > 0
            chunk['metadata']['has_next'] = i < len(chunks) - 1
            chunk['metadata']['file_context'] = f"Chunk {chunk['metadata']['chunk_number']} of {chunk['metadata']['total_chunks']} from {chunk['metadata']['file_path']}"
        
        return {
            'chunks': chunks,
            'total_chunks': len(chunks),
            'file_path': file_path
        }

    def _parse_query(self, question: str) -> tuple[Optional[list[str]], str]:
        """Parse the query to extract repository names and actual question"""
        # Debug: Print input question
        console.print(f"\n[yellow]Parsing query: '{question}'")
        
        question = question.strip()
        words = question.split()
        repo_names = []
        question_parts = []
        
        for word in words:
            if word.startswith('@'):
                repo_name = word[1:]  # Remove @ symbol
                repo_names.append(repo_name)
            else:
                question_parts.append(word)
        
        if repo_names:
            # Debug: Print parsed components
            console.print(f"[yellow]Found repo prefixes: repos={repo_names}, question='{' '.join(question_parts)}'")
            return repo_names, ' '.join(question_parts)
            
        # Debug: Print no repo case
        console.print("[yellow]No repo prefix found, searching all repositories")
        return None, question.strip()

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to approximately max_tokens"""
        # Rough approximation: 1 token ≈ 4 characters
        char_limit = max_tokens * 4
        if len(content) > char_limit:
            return content[:char_limit] + "..."
        return content

    def _prepare_filtered_results(self, filtered_results: list, max_combined_tokens: int) -> list:
        """Prepare and truncate filtered results to fit within token limits"""
        prepared_results = []
        total_chars = 0
        char_limit = max_combined_tokens * 4  # Rough approximation: 1 token ≈ 4 characters
        
        for doc, score, repo in filtered_results:
            # Truncate content if needed
            truncated_content = self._truncate_content(
                doc.page_content, 
                self.search_params["max_tokens_per_source"]
            )
            
            # Check if adding this content would exceed the limit
            content_chars = len(truncated_content)
            if total_chars + content_chars > char_limit:
                break
                
            # Add truncated content
            doc.page_content = truncated_content
            prepared_results.append((doc, score, repo))
            total_chars += content_chars
            
        return prepared_results

    def ask(self, question: str) -> str:
        """Ask a question about the repositories"""
        # Check if vector store directory exists
        if not os.path.exists("./repo_index"):
            return "Repository index not found. Please initialize the index first using the 'Check Changes' or 'Reindex' button in the Vector Store page."

        # Check if embeddings are loaded
        if not self.embeddings:
            return "Error: Embeddings not initialized. Please ensure the agent is properly initialized with load_embeddings=True."

        # Check if vector stores are loaded
        if not self.vector_stores:
            return "No repositories are currently indexed. Please visit the Vector Store page to initialize or check the status of your repositories."

        try:
            # Debug: Print available repositories
            console.print(f"\n[yellow]Available repositories: {list(self.vector_stores.keys())}")
            
            repo_names, actual_question = self._parse_query(question)
            console.print(f"\n[yellow]Parsed query - repo_names: {repo_names}, question: {actual_question}")
            
            all_results = []
            total_k = self.search_params["k"]
            
            if repo_names is not None:
                # Search in specific repositories
                invalid_repos = [repo for repo in repo_names if repo not in self.vector_stores]
                if invalid_repos:
                    return f"Error: Repositories not found: {', '.join(invalid_repos)}. Available repositories: {', '.join(self.vector_stores.keys())}"
                
                # Calculate k per repository for specified repos
                k_per_repo = max(total_k // len(repo_names), self.search_params["min_sources"])
                
                for repo_name in repo_names:
                    vector_store = self.vector_stores[repo_name]
                    console.print(f"\n[yellow]Searching in repository: {repo_name}")
                    
                    try:
                        results = vector_store.similarity_search_with_score(
                            actual_question or question,
                            k=k_per_repo
                        )
                        all_results.extend([(doc, score, repo_name) for doc, score in results])
                        console.print(f"[cyan]Found {len(results)} results in {repo_name}")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error searching in {repo_name}: {str(e)}")
            else:
                # Search across all repositories
                console.print(f"\n[yellow]Searching across all repositories")
                k_per_repo = max(total_k // len(self.vector_stores), self.search_params["min_sources"])
                
                for current_repo, vector_store in self.vector_stores.items():
                    try:
                        results = vector_store.similarity_search_with_score(
                            actual_question or question,
                            k=k_per_repo
                        )
                        repo_results = [(doc, score, current_repo) for doc, score in results]
                        all_results.extend(repo_results)
                        console.print(f"[cyan]Found {len(repo_results)} results in {current_repo}")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error searching in {current_repo}: {str(e)}")
            
            if not all_results:
                return "No results found in the repositories."
            
            # Sort by similarity score (lower score means more similar)
            all_results.sort(key=lambda x: x[1])
            
            # Filter results by similarity threshold
            filtered_results = [
                result for result in all_results 
                if result[1] <= (1 - self.search_params["score_threshold"])
            ]
            
            # Ensure minimum number of sources if available
            if len(filtered_results) < self.search_params["min_sources"] and len(all_results) >= self.search_params["min_sources"]:
                filtered_results = all_results[:self.search_params["min_sources"]]
            
            # Prepare results to fit within token limits
            filtered_results = self._prepare_filtered_results(
                filtered_results,
                self.search_params["max_combined_tokens"]
            )
            
            if not filtered_results:
                return "No relevant results found that meet the similarity threshold."
            
            # Print detailed results for debugging
            console.print("\n[blue]Search results across repositories:")
            for doc, score, repo in filtered_results[:10]:
                similarity = 1 - score
                console.print(f"[cyan]- [{repo}] (similarity: {similarity:.4f})")
                console.print(f"  File: {doc.metadata.get('file_path', 'unknown')}")
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                console.print(f"  Content preview: {preview}")
            
            # Create a temporary vector store with combined results
            combined_docs = [result[0] for result in filtered_results]
            vector_store = Chroma.from_documents(
                documents=combined_docs,
                embedding=self.embeddings,
                collection_metadata=self.collection_metadata
            )

            # Initialize QA chain with the appropriate vector store and model
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": min(len(filtered_results), 10)}
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOllama(
                    temperature=0,
                    model=self.ollama_model,
                    base_url=self.ollama_base_url
                ),
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": self._get_prompt_template()}
            )
            
            result = qa_chain({"question": actual_question or question})
            
            # Add repository context and search statistics to the answer
            if repo_names is not None:
                result["answer"] = f"[Searching in repositories: {', '.join(repo_names)}]\n\n{result['answer']}"
            else:
                repos_searched = sorted(self.vector_stores.keys())
                total_results = len(filtered_results)
                sources_by_repo = {}
                
                # Count sources from each repository
                for _, _, repo in filtered_results:
                    sources_by_repo[repo] = sources_by_repo.get(repo, 0) + 1
                
                # Format repository statistics
                repo_stats = []
                for repo in repos_searched:
                    source_count = sources_by_repo.get(repo, 0)
                    if source_count > 0:
                        repo_stats.append(f"{repo} ({source_count} sources)")
                
                # Add source documents to the answer
                source_files = []
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        if "file_path" in doc.metadata:
                            source_files.append(f"{doc.metadata['repo_name']}/{doc.metadata['file_path']}")
                
                result["answer"] = (
                    f"[Search Statistics]\n"
                    f"- Repositories with matches: {', '.join(repo_stats)}\n"
                    f"- Total relevant sources: {total_results}\n"
                    f"- Similarity threshold: {self.search_params['score_threshold']}\n"
                    f"- Max sources per repository: {k_per_repo}\n\n"
                    f"==============================================\n\n"
                    f"{result['answer']}\n\n"
                    f"[Sources:\n{chr(10).join(source_files)}]" if source_files else result["answer"]
                )
            
            return result["answer"]
            
        except Exception as e:
            console.print(f"[red]Error during search: {str(e)}")
            return f"Error: {str(e)}"

    def _get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for the QA chain"""
        prompt_template = """
        You are a highly knowledgeable code analysis assistant with expertise in understanding codebases. Your role is to provide accurate and concise answers about any aspect of the code, including but not limited to:

        1. Configuration values and environment variables
        2. API endpoints, timeouts, and configurations
        3. Function implementations and their authors
        4. Code structure and organization
        5. Dependencies and external services
        6. Database schemas and queries
        7. Authentication and security measures
        8. Git history and code changes
        9. Documentation and comments
        10. Testing and deployment configurations

        When answering questions:
        - For configuration values: Provide the exact value and where it's defined
        - For API endpoints: Include the full path, method, and relevant configurations
        - For function details: Mention the author, location, and key implementation details
        - For listings (like API endpoints): Present them in a clear, structured format
        - For Git history: Include relevant commit information and authors
        - Always cite the specific files and line numbers where information was found

        Use the following context to provide accurate and specific answers.
        If information is spread across multiple files, combine them to give a complete picture.
        If you're not completely certain about something, explain what you know and what might need verification.
        
        I need you to answer as a bullet point list.

        Context: {context}

        Question: {question}

        Answer: Let me provide you with the specific information from the codebase:
        """

        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    def _format_time_estimate(self, seconds: float) -> str:
        """Format time estimate in a human-readable way"""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            if minutes > 0:
                return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
            return f"{hours} hour{'s' if hours != 1 else ''}"

@app.command()
def setup(repos_path: str = typer.Option(..., help="Path to your repositories folder"),
          ollama_base_url: Optional[str] = typer.Option(None, help="Ollama base URL (default: http://localhost:11434)"),
          ollama_model: str = typer.Option(OLLAMA_MODEL, help="Ollama model to use"),
          force_reindex: bool = typer.Option(False, help="Force reindexing of all files")):
    """Setup and index your repositories"""
    load_dotenv()

    agent = RepoAgent(
        repos_path=repos_path,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        force_reindex=force_reindex
    )
    asyncio.run(agent.index_repositories(force_reindex=force_reindex))
    console.print("[green]Setup complete! You can now ask questions about your repositories.")

@app.command()
def ask(question: str,
        repos_path: str = typer.Option(..., help="Path to your repositories folder"),
        ollama_base_url: Optional[str] = typer.Option(None, help="Ollama base URL"),
        ollama_model: str = typer.Option(OLLAMA_MODEL, help="Ollama model to use")):
    """Ask a question about your repositories"""
    load_dotenv()

    agent = RepoAgent(
        repos_path=repos_path,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model
    )
    
    # Load existing index if available
    if os.path.exists("./repo_index"):
        # This will automatically check for changes and reindex if needed
        asyncio.run(agent.index_repositories(force_reindex=False))
    else:
        console.print("[yellow]No index found. Creating new index...")
        asyncio.run(agent.index_repositories(force_reindex=True))

    answer = agent.ask(question)
    console.print(f"\n[green]Answer:[/green] {answer}")

if __name__ == "__main__":
    app() 