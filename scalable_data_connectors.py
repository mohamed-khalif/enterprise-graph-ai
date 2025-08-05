import pandas as pd
import numpy as np
from typing import Dict, List, Iterator, Optional, Tuple
import os
import tempfile
from pathlib import Path
import asyncio
import aiofiles
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import boto3
from botocore.exceptions import ClientError
import pyarrow.parquet as pq
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class DataSource:
    """Configuration for different data sources"""
    source_type: str  # 'database', 's3', 'file', 'url'
    connection_string: str
    table_name: Optional[str] = None
    query: Optional[str] = None
    chunk_size: int = 10000
    max_rows: Optional[int] = None

class ScalableDataIngestor:
    """Handle large-scale data ingestion from multiple sources"""
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def ingest_from_database(self, 
                                 connection_string: str, 
                                 query: str,
                                 chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Stream data from database in chunks"""
        
        def read_chunk(offset: int, limit: int) -> pd.DataFrame:
            engine = create_engine(connection_string)
            
            # Add LIMIT and OFFSET to query
            paginated_query = f"""
            SELECT * FROM ({query}) as subquery 
            LIMIT {limit} OFFSET {offset}
            """
            
            return pd.read_sql(paginated_query, engine)
        
        offset = 0
        while True:
            try:
                # Use thread executor for database operations
                loop = asyncio.get_event_loop()
                chunk = await loop.run_in_executor(
                    self.executor, read_chunk, offset, chunk_size
                )
                
                if chunk.empty:
                    break
                    
                yield chunk
                offset += chunk_size
                
            except Exception as e:
                logging.error(f"Database ingestion error: {e}")
                break
    
    async def ingest_from_s3(self, 
                           bucket: str, 
                           key_prefix: str,
                           chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Stream data from S3 files"""
        
        s3_client = boto3.client('s3')
        
        try:
            # List all files with the prefix
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                
                # Skip directories
                if key.endswith('/'):
                    continue
                
                # Download file to temp location
                temp_file = os.path.join(self.temp_dir, f"s3_temp_{os.path.basename(key)}")
                s3_client.download_file(bucket, key, temp_file)
                
                # Stream the file in chunks
                async for chunk in self.ingest_from_file(temp_file, chunk_size):
                    yield chunk
                
                # Clean up temp file
                os.unlink(temp_file)
                
        except ClientError as e:
            logging.error(f"S3 ingestion error: {e}")
    
    async def ingest_from_file(self, 
                             file_path: str, 
                             chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Stream large files in chunks"""
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # Use pandas chunking for CSV
                chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
                for chunk in chunk_iter:
                    yield chunk
                    
            elif file_ext == '.parquet':
                # Use PyArrow for efficient Parquet reading
                parquet_file = pq.ParquetFile(file_path)
                
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    chunk = batch.to_pandas()
                    yield chunk
                    
            elif file_ext in ['.xlsx', '.xls']:
                # Excel files - load in memory but can be chunked if needed
                df = pd.read_excel(file_path)
                
                # Yield in chunks
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i+chunk_size]
                    
        except Exception as e:
            logging.error(f"File ingestion error: {e}")
    
    async def ingest_from_url(self, 
                            url: str, 
                            chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Stream data from URL (CSV, API endpoints)"""
        
        try:
            if url.endswith('.csv'):
                # Stream CSV from URL
                chunk_iter = pd.read_csv(url, chunksize=chunk_size)
                for chunk in chunk_iter:
                    yield chunk
            else:
                # For APIs, would need specific implementation
                # This is a placeholder for API streaming
                response_data = pd.read_json(url)
                
                # Yield in chunks
                for i in range(0, len(response_data), chunk_size):
                    yield response_data.iloc[i:i+chunk_size]
                    
        except Exception as e:
            logging.error(f"URL ingestion error: {e}")

class StreamingGraphProcessor:
    """Process large datasets in streaming fashion for graph creation"""
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.node_cache = {}
        self.edge_cache = []
        self.batch_size = 1000
        
    async def process_streaming_data(self, 
                                   data_source: DataSource,
                                   schema_ingestor) -> Dict:
        """Process large datasets in streaming fashion"""
        
        ingestor = ScalableDataIngestor(self.temp_dir)
        total_rows = 0
        sample_data = None
        
        # Determine ingestion method based on source type
        if data_source.source_type == 'database':
            data_stream = ingestor.ingest_from_database(
                data_source.connection_string, 
                data_source.query,
                data_source.chunk_size
            )
        elif data_source.source_type == 's3':
            # Parse S3 path
            s3_parts = data_source.connection_string.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            prefix = s3_parts[1] if len(s3_parts) > 1 else ''
            
            data_stream = ingestor.ingest_from_s3(bucket, prefix, data_source.chunk_size)
        elif data_source.source_type == 'file':
            data_stream = ingestor.ingest_from_file(
                data_source.connection_string,
                data_source.chunk_size
            )
        elif data_source.source_type == 'url':
            data_stream = ingestor.ingest_from_url(
                data_source.connection_string,
                data_source.chunk_size
            )
        else:
            raise ValueError(f"Unsupported source type: {data_source.source_type}")
        
        # Process chunks
        async for chunk in data_stream:
            if sample_data is None:
                # Use first chunk for schema analysis
                sample_data = chunk.copy()
                
            # Process chunk for graph construction
            await self.process_chunk(chunk, schema_ingestor)
            total_rows += len(chunk)
            
            # Memory management - flush caches periodically
            if len(self.edge_cache) > 10000:
                await self.flush_caches()
                
            # Optional: limit total rows processed
            if data_source.max_rows and total_rows >= data_source.max_rows:
                break
        
        # Final flush
        await self.flush_caches()
        
        # Return summary statistics
        return {
            'total_rows_processed': total_rows,
            'sample_analysis': schema_ingestor.analyze_csv(sample_data) if sample_data is not None else {},
            'nodes_created': len(self.node_cache),
            'edges_created': len(self.edge_cache)
        }
    
    async def process_chunk(self, chunk: pd.DataFrame, schema_ingestor):
        """Process a single chunk of data"""
        
        # Extract nodes from this chunk
        for idx, row in chunk.iterrows():
            node_id = f"row_{idx}"
            
            # Store node data (simplified - would use actual graph converter logic)
            self.node_cache[node_id] = {
                'data': row.to_dict(),
                'timestamp': pd.Timestamp.now()
            }
        
        # Extract edges (relationships) - simplified example
        # In practice, this would use the RelationalToGraphConverter logic
        
        # Simulate edge creation
        if len(chunk) > 1:
            for i in range(len(chunk) - 1):
                edge = {
                    'source': f"row_{i}",
                    'target': f"row_{i+1}",
                    'type': 'sequence',
                    'weight': 1.0
                }
                self.edge_cache.append(edge)
    
    async def flush_caches(self):
        """Write cached data to temporary storage"""
        
        # In practice, would write to efficient storage (Parquet, database)
        cache_file = os.path.join(self.temp_dir, f"graph_cache_{pd.Timestamp.now().timestamp()}.tmp")
        
        # Simplified - just clear caches for memory management
        self.node_cache.clear()
        self.edge_cache.clear()

class ProgressiveGNNTrainer:
    """Train GNN models on streaming/large datasets"""
    
    def __init__(self):
        self.model = None
        self.feature_buffer = []
        self.label_buffer = []
        self.batch_size = 1000
        
    async def train_on_stream(self, data_stream: Iterator[pd.DataFrame]):
        """Train GNN model incrementally on streaming data"""
        
        batch_count = 0
        
        async for chunk in data_stream:
            # Extract features and labels from chunk
            features, labels = self.extract_features_labels(chunk)
            
            self.feature_buffer.extend(features)
            self.label_buffer.extend(labels)
            
            # Train when buffer is full
            if len(self.feature_buffer) >= self.batch_size:
                await self.train_batch()
                batch_count += 1
                
                # Clear buffers
                self.feature_buffer.clear()
                self.label_buffer.clear()
        
        # Train final batch
        if self.feature_buffer:
            await self.train_batch()
        
        return {
            'batches_trained': batch_count,
            'model_ready': True
        }
    
    def extract_features_labels(self, chunk: pd.DataFrame) -> Tuple[List, List]:
        """Extract features and synthetic labels from chunk"""
        
        features = []
        labels = []
        
        for _, row in chunk.iterrows():
            # Simplified feature extraction
            feature_vector = [
                row.get('amount', 0),
                row.get('quantity', 0),
                1 if row.get('status') == 'cancelled' else 0
            ]
            features.append(feature_vector)
            
            # Synthetic label (in practice, use real labels)
            label = 1 if row.get('amount', 0) > 1000 else 0
            labels.append(label)
        
        return features, labels
    
    async def train_batch(self):
        """Train model on current batch"""
        
        # Simplified training step
        # In practice, would use actual GNN training logic
        
        print(f"Training batch with {len(self.feature_buffer)} samples")
        
        # Simulate training time
        await asyncio.sleep(0.1)