import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import shutil

console = Console()

class IndexingManager:
    def __init__(self, index_root: str = "./repo_index"):
        self.index_root = Path(index_root)
        self.checkpoint_file = self.index_root / "checkpoint.json"
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, Any] = {
            "total_chunks_processed": 0,
            "chunks_per_second": [],
            "start_time": None,
            "last_checkpoint": None
        }
        
    def _load_checkpoint(self) -> Dict:
        """Load the last checkpoint if it exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load checkpoint: {str(e)}")
        return {}
        
    def _save_checkpoint(self, repo_dir: str, processed_files: List[str], current_chunk: int):
        """Save current progress to checkpoint"""
        checkpoint = self._load_checkpoint()
        checkpoint[repo_dir] = {
            "processed_files": processed_files,
            "current_chunk": current_chunk,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            self.stats["last_checkpoint"] = datetime.now().isoformat()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save checkpoint: {str(e)}")

    def _get_checkpoint(self, repo_dir: str) -> Tuple[List[str], int]:
        """Get the last checkpoint for a repository"""
        checkpoint = self._load_checkpoint()
        if repo_dir in checkpoint:
            return (
                checkpoint[repo_dir]["processed_files"],
                checkpoint[repo_dir]["current_chunk"]
            )
        return [], 0

    def _update_stats(self, chunks_processed: int, duration: float):
        """Update processing statistics"""
        if duration > 0:
            chunks_per_second = chunks_processed / duration
            self.stats["chunks_per_second"].append(chunks_per_second)
            self.stats["total_chunks_processed"] += chunks_processed
            
            # Calculate moving average
            recent_speeds = self.stats["chunks_per_second"][-10:]
            avg_speed = sum(recent_speeds) / len(recent_speeds)
            
            console.print(f"[cyan]Processing speed: {avg_speed:.2f} chunks/second")
            console.print(f"[cyan]Total chunks processed: {self.stats['total_chunks_processed']}")

    async def process_document_batch(
        self,
        documents: List[Document],
        embeddings: Any,
        collection_metadata: Dict,
        batch_size: int = 100
    ) -> Tuple[List[Document], List[np.ndarray]]:
        """Process a batch of documents in parallel"""
        start_time = time.time()
        
        # Split documents into smaller batches for embedding
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        all_embeddings = []
        
        # Process each batch
        for batch in batches:
            try:
                # Get texts from the batch
                texts = [doc.page_content for doc in batch]
                
                # Generate embeddings for the batch
                batch_embeddings = embeddings.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                duration = time.time() - start_time
                self._update_stats(len(batch), duration)
                
                # Small delay to prevent overwhelming the Ollama server
                await asyncio.sleep(0.1)
                
            except Exception as e:
                console.print(f"[red]Error processing batch: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
        
        return documents, all_embeddings

    async def create_incremental_store(
        self,
        documents: List[Document],
        index_path: str,
        embeddings: Any,
        collection_metadata: Dict[str, Any],
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Optional[Chroma]:
        """Create a vector store incrementally with batching and better error handling"""
        try:
            # Ensure the index directory exists and is clean
            os.makedirs(index_path, exist_ok=True)
            
            # Split documents into batches
            total_docs = len(documents)
            batches = [documents[i:i + batch_size] for i in range(0, total_docs, batch_size)]
            console.print(f"[blue]Creating vector store with {total_docs} documents in {len(batches)} batches")
            
            vector_store = None
            docs_processed = 0
            
            # Process each batch
            for batch_idx, batch in enumerate(batches):
                try:
                    # Print batch statistics
                    batch_size_mb = sum(len(doc.page_content) for doc in batch) / 1024 / 1024
                    console.print(f"[blue]Processing batch {batch_idx + 1}/{len(batches)} ({batch_size_mb:.2f} MB)")
                    
                    if batch_idx == 0:
                        # Create initial vector store with first batch
                        try:
                            vector_store = Chroma.from_documents(
                                documents=batch,
                                embedding=embeddings,
                                persist_directory=index_path,
                                collection_metadata=collection_metadata
                            )
                            console.print(f"[green]Successfully created initial vector store")
                        except Exception as e:
                            console.print(f"[red]Failed to create initial vector store: {str(e)}")
                            raise
                    else:
                        # Add subsequent batches to existing store
                        try:
                            vector_store.add_documents(batch)
                            console.print(f"[green]Successfully added batch {batch_idx + 1}")
                        except Exception as e:
                            console.print(f"[red]Failed to add batch {batch_idx + 1}: {str(e)}")
                            raise
                    
                    # Update progress after each batch
                    docs_processed += len(batch)
                    if progress_callback:
                        progress_callback(docs_processed)
                    
                    # Persist after each batch
                    if vector_store:
                        try:
                            vector_store.persist()
                            console.print(f"[green]Successfully persisted after batch {batch_idx + 1}")
                        except Exception as e:
                            console.print(f"[red]Failed to persist after batch {batch_idx + 1}: {str(e)}")
                            raise
                        
                except Exception as e:
                    console.print(f"[yellow]Error processing batch {batch_idx + 1}/{len(batches)}: {str(e)}")
                    # If this is not the first batch and we have a vector store, try to continue
                    if batch_idx > 0 and vector_store:
                        console.print(f"[yellow]Attempting to continue with next batch...")
                        continue
                    else:
                        raise  # Re-raise if it's the first batch or we have no vector store
            
            if vector_store:
                console.print(f"[green]Successfully created vector store with {docs_processed} documents")
            return vector_store
            
        except Exception as e:
            console.print(f"[red]Error creating incremental store: {str(e)}")
            # Clean up failed index
            if os.path.exists(index_path):
                try:
                    shutil.rmtree(index_path)
                    console.print(f"[yellow]Cleaned up failed index directory: {index_path}")
                except Exception as cleanup_error:
                    console.print(f"[yellow]Warning: Could not clean up failed index: {str(cleanup_error)}")
            raise

    def estimate_completion_time(self) -> str:
        """Estimate time to completion based on current processing speed"""
        if not self.stats["chunks_per_second"]:
            return "Calculating..."
            
        recent_speeds = self.stats["chunks_per_second"][-10:]
        if not recent_speeds:
            return "Calculating..."
            
        avg_speed = sum(recent_speeds) / len(recent_speeds)
        if avg_speed <= 0:
            return "Cannot estimate completion time"
            
        remaining_chunks = self.stats.get("total_chunks", 0) - self.stats["total_chunks_processed"]
        if remaining_chunks <= 0:
            return "Almost done"
            
        seconds_remaining = remaining_chunks / avg_speed
        minutes_remaining = int(seconds_remaining / 60)
        
        if minutes_remaining < 1:
            return f"Less than a minute remaining"
        elif minutes_remaining == 1:
            return "About 1 minute remaining"
        else:
            return f"About {minutes_remaining} minutes remaining"

    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        return {
            "total_chunks_processed": self.stats["total_chunks_processed"],
            "average_speed": sum(self.stats["chunks_per_second"][-10:]) / len(self.stats["chunks_per_second"][-10:]) if self.stats["chunks_per_second"] else 0,
            "estimated_completion": self.estimate_completion_time(),
            "last_checkpoint": self.stats["last_checkpoint"],
            "start_time": self.stats["start_time"]
        } 