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
                 progress_callback: Optional[Callable] = None):
        self.repos_path = Path(repos_path)
        self.ollama_base_url = ollama_base_url or "http://localhost:11434"
        self.ollama_model = ollama_model
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            base_url=self.ollama_base_url,
            model=self.ollama_model
        )
        
        self.vector_stores = {}  # Dictionary to store vector stores for each repo
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.index_metadata_file = Path("./repo_index/file_metadata.json")
        self.progress_callback = progress_callback
        
        # HNSW index settings for better search performance
        self.collection_metadata = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
            "hnsw:M": 32,
        }
        
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

    async def send_progress(self, stage: str, current: int, total: int, message: str = ""):
        """Send progress updates through callback"""
        if self.progress_callback:
            try:
                data = {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": message
                }
                # Call the callback directly - it will handle creating the task
                self.progress_callback(data)
                # Small delay to allow the event loop to process the callback
                await asyncio.sleep(0)
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
        """Process a single repository and create its vector store"""
        try:
            repo_path = os.path.join(self.repos_path, repo_dir)
            index_path = self._get_repo_index_path(repo_dir)
            
            # Check repository status
            needs_update, status_message = await self._check_repo_status(repo_dir, repo_path)
            
            if not needs_update and not force_reindex:
                console.print(f"[green]✓ Repository {repo_dir} is up to date")
                return
            
            console.print(f"[yellow]⚡ {status_message}")
            
            # Process repository files
            documents = []
            file_count = 0
            total_files = sum(1 for root, _, files in os.walk(repo_path) 
                             if not any(excluded in root.split(os.sep) for excluded in self.exclude_dirs)
                             for file in files if self._is_allowed_file(os.path.join(root, file)))
            
            for root, _, files in os.walk(repo_path):
                if any(excluded in root.split(os.sep) for excluded in self.exclude_dirs):
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if self._is_allowed_file(file_path):
                            loader = TextLoader(file_path)
                            file_docs = loader.load()
                            file_count += 1
                            
                            # Report progress for file processing
                            progress = int((file_count / total_files) * 40) + 10  # 10-50%
                            await self.send_progress(
                                'processing',
                                progress,
                                100,
                                f'Processing file {file_count}/{total_files}: {os.path.relpath(file_path, repo_path)}'
                            )
                            
                            # Add file path and extension to metadata
                            for doc in file_docs:
                                doc.metadata['file_path'] = os.path.relpath(file_path, repo_path)
                                doc.metadata['file_extension'] = os.path.splitext(file_path)[1]
                                doc.metadata['repo_name'] = repo_dir
                            
                            documents.extend(file_docs)
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Could not load {file_path}: {str(e)}")

            if documents:
                console.print(f"✂️  Splitting {len(documents)} documents from {repo_dir}")
                await self.send_progress(
                    'splitting',
                    50,
                    100,
                    f'Processing {len(documents)} documents from {repo_dir}'
                )
                
                # Process documents in parallel
                vector_store, splits = await process_documents_parallel(documents, repo_dir, self, index_path)
                self.vector_stores[repo_dir] = vector_store

                # Save repository metadata with current commit hash
                current_commit = self._get_current_commit_hash(repo_path)
                if current_commit:
                    metadata = {
                        'last_commit_hash': current_commit,
                        'last_indexed': datetime.now().isoformat(),
                        'total_documents': len(documents),
                        'total_chunks': len(splits)
                    }
                    self._save_repo_metadata(repo_dir, metadata)

                # Process git log if available
                try:
                    git_log = self._get_git_log(repo_path)
                    if git_log:
                        git_docs = []
                        for entry in git_log:
                            doc = Document(
                                page_content=f"{entry['message']}\n\nFiles changed:\n{entry['files']}",
                                metadata={
                                    'type': 'git_log',
                                    'commit_hash': entry['hash'],
                                    'author': entry['author'],
                                    'date': entry['date'],
                                    'repo_name': repo_dir
                                }
                            )
                            git_docs.append(doc)
                        
                        if git_docs:
                            git_store = Chroma.from_documents(
                                documents=git_docs,
                                persist_directory=f"{index_path}_git",
                                embedding=self.embeddings,
                                collection_metadata=self.collection_metadata
                            )
                            # Merge git store into main store
                            self.vector_stores[repo_dir] = vector_store
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not process git log for {repo_dir}: {str(e)}")

                await self.send_progress(
                    'complete',
                    100,
                    100,
                    f'Repository {repo_dir} indexed successfully'
                )
                
                console.print(f"[green]✓ Repository {repo_dir} indexed successfully")
            else:
                console.print(f"[yellow]⚠️  No documents found in {repo_dir}")
            
        except Exception as e:
            console.print(f"[red]❌ Error processing repository {repo_dir}: {str(e)}")
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

        # Create tasks for parallel processing
        tasks = []
        for repo_idx, repo_dir in enumerate(repo_dirs, 1):
            task = asyncio.create_task(
                self.process_single_repository(repo_dir, repo_idx, total_repos, force_reindex)
            )
            tasks.append(task)
        
        # Wait for all repositories to be processed
        try:
            await asyncio.gather(*tasks)
            final_message = f"✨ All {total_repos} Git repositories processed in parallel"
            console.print(f"[green]{final_message}")
            await self.send_progress("complete", total_repos, total_repos, final_message)
        except Exception as e:
            error_message = f"❌ Error during parallel repository processing: {str(e)}"
            console.print(f"[red]{error_message}")
            await self.send_progress("error", 0, total_repos, error_message)
            raise

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
        if not self.vector_stores:
            console.print("[red]Error: No repositories indexed. Please run index_repositories first.")
            return "Error: No repositories indexed. Please run index_repositories first."

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
        ollama_model=ollama_model
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