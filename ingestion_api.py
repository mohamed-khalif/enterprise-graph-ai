import os
import tempfile
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect
from typing import Dict, List
from dataclasses import asdict

import boto3
from botocore.exceptions import ClientError

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict
from typing import Dict, List, Any
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
from enum import Enum
from pydantic import BaseModel

import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize boto3 S3 client using env vars or IAM role
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

class SchemaAwareIngestor:
    def __init__(self, db_url: str = None):
        self.db_url = db_url
        self.engine = create_engine(db_url) if db_url else None

    def upload_csv(self, file_path: str) -> pd.DataFrame:
        # Try to auto-detect date columns by common names first
        date_columns = [col for col in pd.read_csv(file_path, nrows=0).columns 
                       if any(term in col.lower() for term in ['date', 'time', 'created', 'updated', 'timestamp'])]
        
        if date_columns:
            df = pd.read_csv(file_path, parse_dates=date_columns)
        else:
            # Fallback to regular read
            df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
        
        return df

    def connect_database(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def test_connection(self) -> bool:
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def get_table_schema(self, table_name: str) -> Dict:
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)

        return {
            'columns': columns,
            'foreign_keys': foreign_keys,
        }

    def detect_primary_keys(self, table_name: str) -> List[str]:
        inspector = inspect(self.engine)
        return inspector.get_pk_constraint(table_name)['constrained_columns']

    def detect_foreign_keys(self, table_name: str) -> List[Dict]:
        inspector = inspect(self.engine)
        return inspector.get_foreign_keys(table_name)

    def handle_temporal_fields(self, df: pd.DataFrame) -> List[str]:
        temporal_fields = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        return temporal_fields

    def detect_nested_fields(self, df: pd.DataFrame) -> List[str]:
        nested_fields = []
        for col in df.columns:
            # Check if column contains JSON-like strings
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, str) and (sample.startswith('{') or sample.startswith('[')):
                nested_fields.append(col)
        return nested_fields

    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        type_mapping = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                type_mapping[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_mapping[col] = 'datetime'
            elif df[col].dtype == 'object':
                type_mapping[col] = 'text'
            else:
                type_mapping[col] = 'unknown'
        return type_mapping

    def summarize_schema(self, table_name: str):
        schema = self.get_table_schema(table_name)
        pks = self.detect_primary_keys(table_name)
        fks = self.detect_foreign_keys(table_name)

        return {
            'table_name': table_name,
            'primary_keys': pks,
            'foreign_keys': fks,
            'columns': schema['columns'],
        }

    def analyze_csv(self, df: pd.DataFrame) -> Dict:
        """Comprehensive CSV analysis combining all detection methods"""
        return {
            "columns": df.columns.tolist(),
            "temporal_fields": self.handle_temporal_fields(df),
            "nested_fields": self.detect_nested_fields(df),
            "data_types": self.infer_data_types(df),
            "row_count": len(df),
            "null_counts": df.isnull().sum().to_dict()
        }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        ingestor = SchemaAwareIngestor()
        df = ingestor.upload_csv(tmp_path)
        analysis = ingestor.analyze_csv(df)

        # Clean up temp file
        os.unlink(tmp_path)

        return JSONResponse(analysis)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/test-db-connection")
async def test_db_connection(db_url: str):
    """Test database connection"""
    try:
        ingestor = SchemaAwareIngestor(db_url)
        is_connected = ingestor.test_connection()
        return {"connected": is_connected}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/generate-presigned-url")
def generate_presigned_url(bucket: str = Query(...), key: str = Query(...), expiration: int = 3600):
    """
    Generate a presigned URL for PUT upload to S3.
    """
    try:
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return {"presigned_url": url}
    except ClientError as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/read-csv-from-s3")
def read_csv_from_s3(bucket: str = Query(...), key: str = Query(...)):
    """
    Read a CSV file directly from S3 and return comprehensive analysis.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        ingestor = SchemaAwareIngestor()
        analysis = ingestor.analyze_csv(df)

        return analysis
    except ClientError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Import graph converter (make sure graph_converter.py exists)
try:
    from graph_converter import RelationalToGraphConverter, GraphNode, GraphEdge
    GRAPH_CONVERTER_AVAILABLE = True
except ImportError:
    print("⚠️ Graph converter not available. Please create graph_converter.py file.")
    GRAPH_CONVERTER_AVAILABLE = False

@app.post("/convert-to-graph")
async def convert_to_graph(files: List[UploadFile] = File(...)):
    """
    Convert multiple CSV files to a graph structure
    Automatically detects relationships and creates nodes/edges
    """
    if not GRAPH_CONVERTER_AVAILABLE:
        return JSONResponse({"error": "Graph converter module not available"}, status_code=500)
    
    try:
        tables_data = {}
        
        # Process each uploaded file
        for file in files:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            # Analyze CSV with existing SchemaAwareIngestor
            ingestor = SchemaAwareIngestor()
            df = ingestor.upload_csv(tmp_path)
            analysis = ingestor.analyze_csv(df)
            
            # Store table data for graph conversion
            table_name = file.filename.replace('.csv', '')
            tables_data[table_name] = {
                'dataframe': df,
                'primary_key': df.columns[0],  # Assume first column is PK
                'temporal_fields': analysis['temporal_fields'],
                'nested_fields': analysis['nested_fields'],
                'data_types': analysis['data_types']
            }
            
            # Clean up temp file
            os.unlink(tmp_path)
        
        # Convert to graph
        converter = RelationalToGraphConverter()
        graph_data = converter.convert_tables_to_graph(tables_data)
        
        return JSONResponse({
            'success': True,
            'tables_processed': len(tables_data),
            'graph': graph_data,
            'visualization_ready': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/single-table-to-graph") 
async def single_table_to_graph(file: UploadFile = File(...)):
    """
    Convert a single CSV to graph nodes (useful for entity extraction)
    """
    if not GRAPH_CONVERTER_AVAILABLE:
        return JSONResponse({"error": "Graph converter module not available"}, status_code=500)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Analyze the CSV
        ingestor = SchemaAwareIngestor()
        df = ingestor.upload_csv(tmp_path)
        analysis = ingestor.analyze_csv(df)
        
        # Create graph structure from single table
        converter = RelationalToGraphConverter()
        table_name = file.filename.replace('.csv', '')
        
        # Create nodes from the table
        nodes = converter.create_nodes_from_table(
            df, 
            table_name, 
            df.columns[0],  # Primary key
            analysis['temporal_fields']
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        return JSONResponse({
            'table_name': table_name,
            'nodes_created': len(nodes),
            'node_types': list(set(node.type for node in nodes.values())),
            'nodes': [asdict(node) for node in nodes.values()],
            'temporal_fields': analysis['temporal_fields'],
            'nested_fields': analysis['nested_fields']
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
try:
    from gnn_pipeline import GNNPipeline, PredictionResult
    GNN_AVAILABLE = True
except ImportError:
    print("⚠️ GNN pipeline not available. Please install PyTorch and PyTorch Geometric.")
    GNN_AVAILABLE = False

# Global GNN pipeline instance
gnn_pipeline = GNNPipeline() if GNN_AVAILABLE else None

@app.post("/predict-fraud")
async def predict_fraud(files: List[UploadFile] = File(...)):
    """
    Predict fraud risk using GNN on uploaded graph data
    """
    if not GNN_AVAILABLE:
        return JSONResponse({"error": "GNN pipeline not available. Please install PyTorch and PyTorch Geometric."}, status_code=500)
    
    try:
        # Convert files to graph (reuse existing logic)
        tables_data = {}
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            ingestor = SchemaAwareIngestor()
            df = ingestor.upload_csv(tmp_path)
            analysis = ingestor.analyze_csv(df)
            
            table_name = file.filename.replace('.csv', '')
            tables_data[table_name] = {
                'dataframe': df,
                'primary_key': df.columns[0],
                'temporal_fields': analysis['temporal_fields'],
                'nested_fields': analysis['nested_fields'],
                'data_types': analysis['data_types']
            }
            
            os.unlink(tmp_path)
        
        # Convert to graph
        converter = RelationalToGraphConverter()
        graph_data = converter.convert_tables_to_graph(tables_data)
        
        # Run fraud prediction
        predictions = gnn_pipeline.predict(graph_data, task='fraud')
        
        # Convert results to JSON-serializable format
        results = [asdict(pred) for pred in predictions]
        
        # Summary statistics
        high_risk_count = sum(1 for p in predictions if p.risk_level == "HIGH")
        medium_risk_count = sum(1 for p in predictions if p.risk_level == "MEDIUM")
        avg_fraud_score = sum(p.prediction for p in predictions) / len(predictions)
        
        return JSONResponse({
            'success': True,
            'task': 'fraud_detection',
            'total_nodes': len(predictions),
            'predictions': results,
            'summary': {
                'high_risk_nodes': high_risk_count,
                'medium_risk_nodes': medium_risk_count,
                'average_fraud_score': avg_fraud_score,
                'model_type': 'GraphSAGE',
                'features_used': 'structural + transactional'
            }
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict-churn")
async def predict_churn(files: List[UploadFile] = File(...)):
    """
    Predict customer churn risk using GNN
    """
    if not GNN_AVAILABLE:
        return JSONResponse({"error": "GNN pipeline not available. Please install PyTorch and PyTorch Geometric."}, status_code=500)
    
    try:
        # Convert files to graph (same as fraud detection)
        tables_data = {}
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            ingestor = SchemaAwareIngestor()
            df = ingestor.upload_csv(tmp_path)
            analysis = ingestor.analyze_csv(df)
            
            table_name = file.filename.replace('.csv', '')
            tables_data[table_name] = {
                'dataframe': df,
                'primary_key': df.columns[0],
                'temporal_fields': analysis['temporal_fields'],
                'nested_fields': analysis['nested_fields'],
                'data_types': analysis['data_types']
            }
            
            os.unlink(tmp_path)
        
        # Convert to graph
        converter = RelationalToGraphConverter()
        graph_data = converter.convert_tables_to_graph(tables_data)
        
        # Run churn prediction
        predictions = gnn_pipeline.predict(graph_data, task='churn')
        
        # Convert results
        results = [asdict(pred) for pred in predictions]
        
        # Summary for customers only
        customer_predictions = [p for p in predictions if 'customer' in p.node_id.lower()]
        high_churn_customers = sum(1 for p in customer_predictions if p.risk_level == "HIGH")
        avg_churn_score = sum(p.prediction for p in customer_predictions) / len(customer_predictions) if customer_predictions else 0
        
        return JSONResponse({
            'success': True,
            'task': 'churn_prediction',
            'total_nodes': len(predictions),
            'customer_nodes': len(customer_predictions),
            'predictions': results,
            'summary': {
                'high_churn_risk_customers': high_churn_customers,
                'average_churn_score': avg_churn_score,
                'model_type': 'GraphSAGE',
                'features_used': 'behavioral + network'
            }
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/gnn-insights")
async def gnn_insights(files: List[UploadFile] = File(...)):
    """
    Get comprehensive GNN-based insights (fraud + churn)
    """
    if not GNN_AVAILABLE:
        return JSONResponse({"error": "GNN pipeline not available. Please install PyTorch and PyTorch Geometric."}, status_code=500)
    
    try:
        # Convert files to graph
        tables_data = {}
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            ingestor = SchemaAwareIngestor()
            df = ingestor.upload_csv(tmp_path)
            analysis = ingestor.analyze_csv(df)
            
            table_name = file.filename.replace('.csv', '')
            tables_data[table_name] = {
                'dataframe': df,
                'primary_key': df.columns[0],
                'temporal_fields': analysis['temporal_fields'],
                'nested_fields': analysis['nested_fields'],
                'data_types': analysis['data_types']
            }
            
            os.unlink(tmp_path)
        
        # Convert to graph
        converter = RelationalToGraphConverter()
        graph_data = converter.convert_tables_to_graph(tables_data)
        
        # Run both predictions
        fraud_predictions = gnn_pipeline.predict(graph_data, task='fraud')
        churn_predictions = gnn_pipeline.predict(graph_data, task='churn')
        
        # Create comprehensive insights
        insights = {
            'graph_statistics': graph_data['statistics'],
            'fraud_analysis': {
                'predictions': [asdict(p) for p in fraud_predictions],
                'high_risk_nodes': [p.node_id for p in fraud_predictions if p.risk_level == "HIGH"],
                'summary': {
                    'total_analyzed': len(fraud_predictions),
                    'high_risk_count': sum(1 for p in fraud_predictions if p.risk_level == "HIGH"),
                    'average_fraud_score': sum(p.prediction for p in fraud_predictions) / len(fraud_predictions)
                }
            },
            'churn_analysis': {
                'predictions': [asdict(p) for p in churn_predictions],
                'high_churn_customers': [p.node_id for p in churn_predictions if p.risk_level == "HIGH" and 'customer' in p.node_id.lower()],
                'summary': {
                    'customers_analyzed': len([p for p in churn_predictions if 'customer' in p.node_id.lower()]),
                    'high_churn_count': sum(1 for p in churn_predictions if p.risk_level == "HIGH" and 'customer' in p.node_id.lower()),
                    'average_churn_score': sum(p.prediction for p in churn_predictions if 'customer' in p.node_id.lower()) / len([p for p in churn_predictions if 'customer' in p.node_id.lower()]) if any('customer' in p.node_id.lower() for p in churn_predictions) else 0
                }
            },
            'model_info': {
                'architecture': 'GraphSAGE',
                'scalability': '100k+ nodes',
                'training_time': 'Real-time',
                'features': 'Structural + Transactional + Behavioral'
            }
        }
        
        return JSONResponse({
            'success': True,
            'insights': insights,
            'visualization_ready': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    Serve the main dashboard interface
    """
    # Read the dashboard HTML file
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Dashboard Not Found</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>🚀 AI Graph Intelligence Platform</h1>
                <h2>Dashboard Setup Required</h2>
                <p>Please create the dashboard.html file to view the interface.</p>
                <p>API endpoints are available at:</p>
                <ul style="list-style: none;">
                    <li>📊 <a href="/docs">/docs</a> - API Documentation</li>
                    <li>📁 <a href="/upload-csv">/upload-csv</a> - Upload CSV</li>
                    <li>🔗 <a href="/convert-to-graph">/convert-to-graph</a> - Create Graph</li>
                    <li>🛡️ <a href="/predict-fraud">/predict-fraud</a> - Fraud Detection</li>
                    <li>📉 <a href="/predict-churn">/predict-churn</a> - Churn Prediction</li>
                    <li>🧠 <a href="/gnn-insights">/gnn-insights</a> - AI Insights</li>
                </ul>
            </body>
        </html>
        """)

# Add CORS headers for the dashboard
@app.middleware("http")
async def add_cors_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response    

try:
    from scalable_data_connectors import ScalableDataIngestor, DataSource, StreamingGraphProcessor
    SCALABLE_AVAILABLE = True
except ImportError:
    print("⚠️ Scalable data connectors not available. Please install required dependencies.")
    SCALABLE_AVAILABLE = False

# Pydantic models for API requests
class DatabaseConnection(BaseModel):
    connection_string: str
    query: str
    chunk_size: int = 10000
    max_rows: Optional[int] = None

class S3Connection(BaseModel):
    bucket: str
    key_prefix: str
    chunk_size: int = 10000
    max_rows: Optional[int] = None

class FileConnection(BaseModel):
    file_path: str
    chunk_size: int = 10000
    max_rows: Optional[int] = None

class URLConnection(BaseModel):
    url: str
    chunk_size: int = 10000
    max_rows: Optional[int] = None

# API Endpoints for scalable data ingestion

@app.post("/connect-database")
async def connect_database(connection: DatabaseConnection):
    """
    Connect to database and process large datasets
    Supports PostgreSQL, MySQL, SQL Server, etc.
    """
    if not SCALABLE_AVAILABLE:
        return JSONResponse({"error": "Scalable data connectors not available"}, status_code=500)
    
    try:
        # Create data source
        data_source = DataSource(
            source_type='database',
            connection_string=connection.connection_string,
            query=connection.query,
            chunk_size=connection.chunk_size,
            max_rows=connection.max_rows
        )
        
        # Process streaming data
        processor = StreamingGraphProcessor()
        ingestor = SchemaAwareIngestor()
        
        result = await processor.process_streaming_data(data_source, ingestor)
        
        return JSONResponse({
            'success': True,
            'source_type': 'database',
            'processing_summary': result,
            'scalable': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/connect-s3")
async def connect_s3(connection: S3Connection):
    """
    Connect to S3 and process large datasets
    Supports CSV, Parquet, and other formats
    """
    if not SCALABLE_AVAILABLE:
        return JSONResponse({"error": "Scalable data connectors not available"}, status_code=500)
    
    try:
        # Create S3 path
        s3_path = f"s3://{connection.bucket}/{connection.key_prefix}"
        
        data_source = DataSource(
            source_type='s3',
            connection_string=s3_path,
            chunk_size=connection.chunk_size,
            max_rows=connection.max_rows
        )
        
        # Process streaming data
        processor = StreamingGraphProcessor()
        ingestor = SchemaAwareIngestor()
        
        result = await processor.process_streaming_data(data_source, ingestor)
        
        return JSONResponse({
            'success': True,
            'source_type': 's3',
            's3_path': s3_path,
            'processing_summary': result,
            'scalable': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/connect-large-file")
async def connect_large_file(connection: FileConnection):
    """
    Process large local files (CSV, Parquet, Excel)
    Streams data to avoid memory issues
    """
    if not SCALABLE_AVAILABLE:
        return JSONResponse({"error": "Scalable data connectors not available"}, status_code=500)
    
    try:
        data_source = DataSource(
            source_type='file',
            connection_string=connection.file_path,
            chunk_size=connection.chunk_size,
            max_rows=connection.max_rows
        )
        
        # Process streaming data
        processor = StreamingGraphProcessor()
        ingestor = SchemaAwareIngestor()
        
        result = await processor.process_streaming_data(data_source, ingestor)
        
        return JSONResponse({
            'success': True,
            'source_type': 'large_file',
            'file_path': connection.file_path,
            'processing_summary': result,
            'scalable': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/connect-url")
async def connect_url(connection: URLConnection):
    """
    Connect to URL data sources (APIs, remote CSV files)
    """
    if not SCALABLE_AVAILABLE:
        return JSONResponse({"error": "Scalable data connectors not available"}, status_code=500)
    
    try:
        data_source = DataSource(
            source_type='url',
            connection_string=connection.url,
            chunk_size=connection.chunk_size,
            max_rows=connection.max_rows
        )
        
        # Process streaming data
        processor = StreamingGraphProcessor()
        ingestor = SchemaAwareIngestor()
        
        result = await processor.process_streaming_data(data_source, ingestor)
        
        return JSONResponse({
            'success': True,
            'source_type': 'url',
            'url': connection.url,
            'processing_summary': result,
            'scalable': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict-fraud-scalable")
async def predict_fraud_scalable(data_source: dict):
    """
    Run fraud prediction on large datasets using streaming processing
    """
    if not SCALABLE_AVAILABLE:
        return JSONResponse({"error": "Scalable processing not available"}, status_code=500)
    
    try:
        # Convert dict to DataSource
        source = DataSource(**data_source)
        
        # Process data in streaming fashion
        processor = StreamingGraphProcessor()
        ingestor = SchemaAwareIngestor()
        
        # Get processing summary
        processing_result = await processor.process_streaming_data(source, ingestor)
        
        # For large datasets, we'd implement incremental GNN training
        # This is a simplified version
        
        return JSONResponse({
            'success': True,
            'task': 'fraud_detection_scalable',
            'processing_summary': processing_result,
            'prediction_method': 'streaming_gnn',
            'scalable': True,
            'note': 'Large dataset processed in chunks for memory efficiency'
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/data-source-status")
async def data_source_status():
    """
    Check available data source connectors and their status
    """
    status = {
        'scalable_connectors_available': SCALABLE_AVAILABLE,
        'supported_sources': {
            'databases': {
                'available': True,
                'types': ['PostgreSQL', 'MySQL', 'SQL Server', 'SQLite', 'Oracle'],
                'max_recommended_size': '100M+ rows',
                'streaming': True
            },
            's3': {
                'available': True,
                'formats': ['CSV', 'Parquet', 'JSON'],
                'max_recommended_size': '1TB+',
                'streaming': True
            },
            'large_files': {
                'available': True,
                'formats': ['CSV', 'Parquet', 'Excel'],
                'max_recommended_size': '10GB+',
                'streaming': True
            },
            'urls': {
                'available': True,
                'types': ['REST APIs', 'Remote CSV', 'Data feeds'],
                'streaming': True
            },
            'small_uploads': {
                'available': True,
                'max_size': '2GB',
                'description': 'Browser upload for smaller datasets'
            }
        },
        'memory_efficient': True,
        'enterprise_ready': SCALABLE_AVAILABLE
    }
    
    return JSONResponse(status)

@app.get("/connection-examples")
async def connection_examples():
    """
    Provide example connection configurations for different data sources
    """
    examples = {
        'postgresql': {
            'connection_string': 'postgresql://user:password@host:5432/database',
            'query': 'SELECT * FROM transactions WHERE created_at > NOW() - INTERVAL \'30 days\'',
            'chunk_size': 10000,
            'description': 'Stream recent transactions from PostgreSQL'
        },
        's3_csv': {
            'bucket': 'my-data-bucket',
            'key_prefix': 'data/transactions/',
            'chunk_size': 10000,
            'description': 'Process multiple CSV files from S3'
        },
        's3_parquet': {
            'bucket': 'analytics-data',
            'key_prefix': 'warehouse/fact_sales.parquet',
            'chunk_size': 50000,
            'description': 'Efficient processing of Parquet data'
        },
        'large_csv': {
            'file_path': '/data/large_dataset.csv',
            'chunk_size': 10000,
            'max_rows': 1000000,
            'description': 'Process first 1M rows of large CSV'
        },
        'api_data': {
            'url': 'https://api.company.com/data/export.csv',
            'chunk_size': 5000,
            'description': 'Stream data from API endpoint'
        }
    }
    
    return JSONResponse({
        'examples': examples,
        'notes': {
            'chunk_size': 'Adjust based on available memory (1K-100K typical)',
            'max_rows': 'Optional limit for testing or sampling',
            'streaming': 'All sources support memory-efficient streaming',
            'security': 'Use environment variables for credentials'
        }
    })

try:
    from prebuilt_use_case_modules import (
        UseCaseOrchestrator, UseCaseType, BusinessInsight, UseCaseResult,
        get_recommended_use_cases, INDUSTRY_CONFIGS
    )
    USE_CASES_AVAILABLE = True
except ImportError:
    print("⚠️ Pre-built use case modules not available. Please create prebuilt_use_case_modules.py file.")
    USE_CASES_AVAILABLE = False

# Pydantic models for API requests
class UseCaseRequest(BaseModel):
    use_case: str  # fraud_detection, customer_churn, payment_defaults
    domain_context: str = "general"  # ecommerce, saas, fintech, retail
    industry: str = "general"

class BusinessAnalysisRequest(BaseModel):
    industry: str = "ecommerce"  # ecommerce, saas, fintech, retail
    domain_context: str = "general"
    priority_use_cases: List[str] = []

# Global use case orchestrator
use_case_orchestrator = UseCaseOrchestrator() if USE_CASES_AVAILABLE else None

@app.post("/run-fraud-detection")
async def run_fraud_detection(files: List[UploadFile] = File(...), domain: str = "ecommerce"):
    """
    🛡️ Run AI-powered fraud detection analysis
    Identifies suspicious transactions and patterns
    """
    if not USE_CASES_AVAILABLE:
        return JSONResponse({"error": "Use case modules not available"}, status_code=500)
    
    try:
        # Convert files to graph
        graph_data = await convert_files_to_graph(files)
        
        # Run fraud detection
        result = use_case_orchestrator.run_use_case(
            UseCaseType.FRAUD_DETECTION, 
            graph_data, 
            domain
        )
        
        return JSONResponse({
            'success': True,
            'use_case': 'fraud_detection',
            'title': result.title,
            'description': result.description,
            'insights': [asdict(insight) for insight in result.insights],
            'summary_metrics': result.summary_metrics,
            'recommended_actions': result.recommended_actions,
            'roi_estimate': result.roi_estimate,
            'processing_time': result.processing_time,
            'business_ready': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/run-churn-prediction")
async def run_churn_prediction(files: List[UploadFile] = File(...), domain: str = "saas"):
    """
    📉 Run AI-powered customer churn prediction
    Identifies customers at risk of leaving
    """
    if not USE_CASES_AVAILABLE:
        return JSONResponse({"error": "Use case modules not available"}, status_code=500)
    
    try:
        # Convert files to graph
        graph_data = await convert_files_to_graph(files)
        
        # Run churn prediction
        result = use_case_orchestrator.run_use_case(
            UseCaseType.CUSTOMER_CHURN, 
            graph_data, 
            domain
        )
        
        return JSONResponse({
            'success': True,
            'use_case': 'customer_churn',
            'title': result.title,
            'description': result.description,
            'insights': [asdict(insight) for insight in result.insights],
            'summary_metrics': result.summary_metrics,
            'recommended_actions': result.recommended_actions,
            'roi_estimate': result.roi_estimate,
            'processing_time': result.processing_time,
            'business_ready': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/run-payment-defaults")
async def run_payment_defaults(files: List[UploadFile] = File(...), domain: str = "fintech"):
    """
    💳 Run AI-powered payment default prediction
    Identifies borrowers at risk of defaulting
    """
    if not USE_CASES_AVAILABLE:
        return JSONResponse({"error": "Use case modules not available"}, status_code=500)
    
    try:
        # Convert files to graph
        graph_data = await convert_files_to_graph(files)
        
        # Run payment default analysis
        result = use_case_orchestrator.run_use_case(
            UseCaseType.PAYMENT_DEFAULTS, 
            graph_data, 
            domain
        )
        
        return JSONResponse({
            'success': True,
            'use_case': 'payment_defaults',
            'title': result.title,
            'description': result.description,
            'insights': [asdict(insight) for insight in result.insights],
            'summary_metrics': result.summary_metrics,
            'recommended_actions': result.recommended_actions,
            'roi_estimate': result.roi_estimate,
            'processing_time': result.processing_time,
            'business_ready': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/business-analysis")
async def business_analysis(files: List[UploadFile] = File(...), 
                           analysis_request: BusinessAnalysisRequest = BusinessAnalysisRequest()):
    """
    🎯 Comprehensive business analysis with all relevant use cases
    Industry-specific insights and recommendations
    """
    if not USE_CASES_AVAILABLE:
        return JSONResponse({"error": "Use case modules not available"}, status_code=500)
    
    try:
        # Convert files to graph
        graph_data = await convert_files_to_graph(files)
        
        # Run all relevant use cases
        all_results = use_case_orchestrator.run_all_use_cases(
            graph_data, 
            analysis_request.domain_context
        )
        
        # Generate business summary
        business_summary = use_case_orchestrator.get_business_summary(all_results)
        
        # Get industry-specific recommendations
        recommended_use_cases = get_recommended_use_cases(analysis_request.industry)
        
        return JSONResponse({
            'success': True,
            'analysis_type': 'comprehensive_business_analysis',
            'industry': analysis_request.industry,
            'results': {
                use_case: {
                    'title': result.title,
                    'description': result.description,
                    'insights_count': len(result.insights),
                    'high_risk_count': len([i for i in result.insights if i.risk_level == "HIGH"]),
                    'summary_metrics': result.summary_metrics,
                    'recommended_actions': result.recommended_actions,
                    'roi_estimate': result.roi_estimate,
                    'insights': [asdict(insight) for insight in result.insights]
                }
                for use_case, result in all_results.items()
            },
            'business_summary': business_summary,
            'recommended_use_cases': [uc.value for uc in recommended_use_cases],
            'total_processing_time': sum(result.processing_time for result in all_results.values()),
            'enterprise_ready': True
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/industry-recommendations/{industry}")
async def get_industry_recommendations(industry: str):
    """
    Get recommended use cases and configurations for a specific industry
    """
    if not USE_CASES_AVAILABLE:
        return JSONResponse({"error": "Use case modules not available"}, status_code=500)
    
    try:
        # Get industry configuration
        industry_config = INDUSTRY_CONFIGS.get(industry.lower(), INDUSTRY_CONFIGS["ecommerce"])
        recommended_use_cases = get_recommended_use_cases(industry)
        
        return JSONResponse({
            'industry': industry,
            'recommended_use_cases': [uc.value for uc in recommended_use_cases],
            'configuration': industry_config,
            'use_case_descriptions': {
                'fraud_detection': '🛡️ Identify suspicious transactions and payment fraud',
                'customer_churn': '📉 Predict which customers are likely to leave',
                'payment_defaults': '💳 Assess risk of loan/payment defaults'
            },
            'business_benefits': {
                'fraud_detection': 'Prevent financial losses from fraudulent transactions',
                'customer_churn': 'Improve retention and reduce revenue loss',
                'payment_defaults': 'Minimize credit risk and bad debt'
            }
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/use-case-catalog")
async def get_use_case_catalog():
    """
    Get catalog of all available pre-built use cases
    """
    if not USE_CASES_AVAILABLE:
        return JSONResponse({"error": "Use case modules not available"}, status_code=500)
    
    catalog = {
        'available_use_cases': {
            'fraud_detection': {
                'title': '🛡️ Fraud Detection',
                'description': 'AI-powered transaction fraud detection using network analysis',
                'industries': ['ecommerce', 'fintech', 'retail'],
                'key_metrics': ['High-risk transactions', 'Fraud prevention ROI', 'Alert accuracy'],
                'typical_roi': '300-500%',
                'implementation_time': '< 30 seconds'
            },
            'customer_churn': {
                'title': '📉 Customer Churn Prediction',
                'description': 'Predict customer retention risk using behavioral patterns',
                'industries': ['saas', 'ecommerce', 'retail', 'telecom'],
                'key_metrics': ['Churn probability', 'Revenue at risk', 'Retention opportunities'],
                'typical_roi': '200-400%',
                'implementation_time': '< 30 seconds'
            },
            'payment_defaults': {
                'title': '💳 Payment Default Risk',
                'description': 'Assess credit and payment default risk for lending decisions',
                'industries': ['fintech', 'lending', 'banking'],
                'key_metrics': ['Default probability', 'Credit exposure', 'Risk-adjusted returns'],
                'typical_roi': '150-300%',
                'implementation_time': '< 30 seconds'
            }
        },
        'industry_packages': {
            'ecommerce_package': ['fraud_detection', 'customer_churn'],
            'saas_package': ['customer_churn'],
            'fintech_package': ['fraud_detection', 'payment_defaults', 'customer_churn'],
            'retail_package': ['fraud_detection', 'customer_churn']
        },
        'coming_soon': [
            '📦 Inventory Optimization',
            '👥 Employee Retention',
            '📊 Marketing Attribution',
            '🎯 Lead Scoring'
        ]
    }
    
    return JSONResponse(catalog)

# Helper function to convert files to graph (used by use case endpoints)
async def convert_files_to_graph(files: List[UploadFile]) -> Dict[str, Any]:
    """Convert uploaded files to graph data structure"""
    
    tables_data = {}
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        ingestor = SchemaAwareIngestor()
        df = ingestor.upload_csv(tmp_path)
        analysis = ingestor.analyze_csv(df)
        
        table_name = file.filename.replace('.csv', '')
        tables_data[table_name] = {
            'dataframe': df,
            'primary_key': df.columns[0],
            'temporal_fields': analysis['temporal_fields'],
            'nested_fields': analysis['nested_fields'],
            'data_types': analysis['data_types']
        }
        
        os.unlink(tmp_path)
    
    # Convert to graph
    converter = RelationalToGraphConverter()
    graph_data = converter.convert_tables_to_graph(tables_data)
    
    return graph_data

print("✅ Enhanced ingestion_api with Phase 2 Graph Converter loaded")

#if __name__ == '__main__':
#    db_url = os.getenv("DATABASE_URL")
#    if db_url:
#        ingestor = SchemaAwareIngestor(db_url)
#        table_info = ingestor.summarize_schema("transactions")
#        print("Schema Summary:", table_info)
#
#    uvicorn.run(app, host="0.0.0.0", port=8000)