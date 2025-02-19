from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from rich.console import Console
import shutil
import os

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
                'preview': next((line.strip() for line in content_lines if line.strip()), '')[:100]
            })

        # Create a unique temporary directory for this chunk
        temp_dir = f"{index_path}_temp_{multiprocessing.current_process().name}"
        os.makedirs(temp_dir, exist_ok=True)

        # Create vector store for this chunk with a unique collection name
        chunk_store = Chroma.from_documents(
            documents=splits,
            persist_directory=temp_dir,
            embedding=agent.embeddings,
            collection_name=f"chunk_{multiprocessing.current_process().name}",
            collection_metadata=agent.collection_metadata
        )
        
        return splits, chunk_store, temp_dir
    except Exception as e:
        console.print(f"[red]Error processing document chunk: {str(e)}")
        return None, None, None

async def process_documents_parallel(documents, repo_dir, agent, index_path):
    # Determine number of chunks based on CPU cores
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(documents) // num_cores)
    document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
    
    all_splits = []
    temp_dirs = []
    
    try:
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Create tasks for parallel processing
            tasks = []
            for chunk in document_chunks:
                task = asyncio.create_task(process_document_chunk(chunk, repo_dir, agent, index_path, agent.progress_callback))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            for splits, _, temp_dir in results:
                if splits and temp_dir:
                    all_splits.extend(splits)
                    temp_dirs.append(temp_dir)
        
        # Clean up any existing index
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        
        # Create the final vector store
        final_store = Chroma.from_documents(
            documents=all_splits,
            persist_directory=index_path,
            embedding=agent.embeddings,
            collection_name="main",
            collection_metadata=agent.collection_metadata
        )
        
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not clean up temporary directory {temp_dir}: {str(e)}")
        
        return final_store, all_splits
    except Exception as e:
        # Clean up temporary directories in case of error
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        raise e 