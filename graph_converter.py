import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx

@dataclass
class GraphNode:
    """Represents a typed node in the graph"""
    id: str
    type: str
    properties: Dict[str, Any]
    source_table: str
    primary_key: str

@dataclass
class GraphEdge:
    """Represents a typed edge in the graph"""
    source_id: str
    target_id: str
    edge_type: str
    properties: Dict[str, Any]
    timestamp: Optional[str] = None
    source_table: str = ""
    relationship_type: str = ""

class RelationalToGraphConverter:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_types: Dict[str, str] = {}
        self.edge_types: Dict[str, str] = {}
        self.graph = nx.MultiDiGraph()
        
    def detect_node_types(self, df: pd.DataFrame, table_name: str, primary_key: str) -> Dict[str, str]:
        """Detect node types based on table patterns and content"""
        node_types = {}
        
        # Common entity patterns
        entity_patterns = {
            'customer': ['customer', 'user', 'client', 'buyer'],
            'product': ['product', 'item', 'goods', 'merchandise'],
            'order': ['order', 'transaction', 'purchase', 'sale'],
            'location': ['address', 'location', 'place', 'city', 'state'],
            'organization': ['company', 'organization', 'business', 'vendor'],
            'person': ['person', 'individual', 'employee', 'staff']
        }
        
        table_lower = table_name.lower()
        
        # Match table name to entity type
        detected_type = 'entity'  # default
        for entity_type, patterns in entity_patterns.items():
            if any(pattern in table_lower for pattern in patterns):
                detected_type = entity_type
                break
                
        # Create node type for each row
        for idx, row in df.iterrows():
            node_id = f"{table_name}_{row[primary_key]}"
            node_types[node_id] = detected_type
            
        return node_types

    def create_nodes_from_table(self, df: pd.DataFrame, table_name: str, 
                               primary_key: str, temporal_fields: List[str]) -> Dict[str, GraphNode]:
        """Convert table rows to typed graph nodes"""
        nodes = {}
        
        # Detect node types
        node_types = self.detect_node_types(df, table_name, primary_key)
        
        for idx, row in df.iterrows():
            # Create unique node ID
            node_id = f"{table_name}_{row[primary_key]}"
            
            # Prepare node properties (exclude primary key from properties)
            properties = {}
            for col, value in row.items():
                if col != primary_key and pd.notna(value):
                    # Handle temporal fields
                    if col in temporal_fields and isinstance(value, (pd.Timestamp, datetime)):
                        properties[col] = value.isoformat()
                    # Handle potential JSON fields
                    elif isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            properties[col] = json.loads(value)
                        except:
                            properties[col] = value
                    else:
                        properties[col] = value
            
            # Create node
            node = GraphNode(
                id=node_id,
                type=node_types.get(node_id, 'entity'),
                properties=properties,
                source_table=table_name,
                primary_key=str(row[primary_key])
            )
            
            nodes[node_id] = node
            
        return nodes

    def detect_foreign_key_relationships(self, df: pd.DataFrame, table_name: str, 
                                       all_tables: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str]]:
        """Detect foreign key relationships by analyzing column patterns and data"""
        relationships = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Look for foreign key patterns
            if col_lower.endswith('_id'):
                # Extract potential referenced table name
                referenced_table_base = col_lower.replace('_id', '')
                
                # Try multiple table name variations
                possible_table_names = [
                    referenced_table_base,  # exact: customer_id -> customer
                    referenced_table_base + 's',  # plural: customer_id -> customers  
                    referenced_table_base + 'es' if referenced_table_base.endswith('s') else referenced_table_base + 's',
                ]
                
                # Check all tables for matches
                for other_table_name, other_df in all_tables.items():
                    other_table_lower = other_table_name.lower()
                    
                    # Check if any possible name matches the table name
                    name_match = any(possible_name == other_table_lower for possible_name in possible_table_names)
                    
                    # Also check partial matches
                    partial_match = (referenced_table_base in other_table_lower or 
                                   other_table_lower in referenced_table_base)
                    
                    if (name_match or partial_match) and len(other_df) > 0:
                        # Assume first column is primary key
                        other_pk = other_df.columns[0]
                        
                        # Verify relationship by checking if FK values exist in PK values
                        fk_values = set(df[col].dropna().astype(str))
                        pk_values = set(other_df[other_pk].astype(str))
                        
                        if fk_values:
                            intersection = fk_values.intersection(pk_values)
                            match_ratio = len(intersection) / len(fk_values)
                            
                            # If at least 30% of FK values exist in PK values, it's a relationship
                            if match_ratio > 0.3:
                                relationships.append((col, other_table_name, other_pk))
                                print(f"✅ Detected relationship: {table_name}.{col} -> {other_table_name}.{other_pk} (match: {match_ratio:.1%})")
                                break
        
        print(f"🔍 Found {len(relationships)} relationships for table {table_name}")
        return relationships

    def create_edges_from_relationships(self, df: pd.DataFrame, table_name: str,
                                     primary_key: str, relationships: List[Tuple[str, str, str]],
                                     temporal_fields: List[str]) -> List[GraphEdge]:
        """Create typed edges from foreign key relationships"""
        edges = []
        
        for idx, row in df.iterrows():
            source_node_id = f"{table_name}_{row[primary_key]}"
            
            # Find temporal data for this edge (use most recent timestamp)
            edge_timestamp = None
            for temp_field in temporal_fields:
                if temp_field in row and pd.notna(row[temp_field]):
                    if isinstance(row[temp_field], (pd.Timestamp, datetime)):
                        edge_timestamp = row[temp_field].isoformat() if hasattr(row[temp_field], 'isoformat') else str(row[temp_field])
                        break
            
            # Create edges for each relationship
            for fk_col, ref_table, ref_pk in relationships:
                if pd.notna(row[fk_col]):
                    target_node_id = f"{ref_table}_{row[fk_col]}"
                    
                    # Determine edge type based on relationship
                    edge_type = self.infer_edge_type(table_name, ref_table, fk_col)
                    
                    # Edge properties (include relevant row data)
                    edge_properties = {
                        'foreign_key': fk_col,
                        'source_primary_key': str(row[primary_key]),
                        'target_primary_key': str(row[fk_col])
                    }
                    
                    # Add numerical/categorical properties that might be relevant to the relationship
                    for col, value in row.items():
                        if col not in [primary_key, fk_col] and pd.notna(value):
                            if isinstance(value, (int, float)) or col.lower() in ['quantity', 'amount', 'price', 'rating']:
                                edge_properties[col] = value
                    
                    edge = GraphEdge(
                        source_id=source_node_id,
                        target_id=target_node_id,
                        edge_type=edge_type,
                        properties=edge_properties,
                        timestamp=edge_timestamp,
                        source_table=table_name,
                        relationship_type=f"{table_name}_to_{ref_table}"
                    )
                    
                    edges.append(edge)
        
        return edges

    def infer_edge_type(self, source_table: str, target_table: str, fk_column: str) -> str:
        """Infer semantic edge type from table and column names"""
        source_lower = source_table.lower()
        target_lower = target_table.lower()
        fk_lower = fk_column.lower()
        
        # Common relationship patterns
        if 'order' in source_lower and 'customer' in target_lower:
            return 'ORDERED_BY'
        elif 'order' in source_lower and 'product' in target_lower:
            return 'CONTAINS'
        elif 'customer' in source_lower and 'address' in target_lower:
            return 'LIVES_AT'
        elif 'product' in source_lower and 'category' in target_lower:
            return 'BELONGS_TO'
        elif 'employee' in source_lower and 'department' in target_lower:
            return 'WORKS_IN'
        elif 'review' in source_lower and 'product' in target_lower:
            return 'REVIEWS'
        elif 'payment' in source_lower and 'order' in target_lower:
            return 'PAYS_FOR'
        else:
            # Generic relationship based on FK column name
            if 'customer' in fk_lower:
                return 'BELONGS_TO_CUSTOMER'
            elif 'product' in fk_lower:
                return 'RELATES_TO_PRODUCT'
            else:
                return f"REFERENCES_{target_lower.upper()}"

    def convert_tables_to_graph(self, tables_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Main method to convert multiple tables to a graph structure"""
        all_nodes = {}
        all_edges = []
        
        # Extract DataFrames and metadata
        dataframes = {}
        for table_name, data in tables_data.items():
            dataframes[table_name] = data['dataframe']
        
        # Process each table
        for table_name, data in tables_data.items():
            df = data['dataframe']
            primary_key = data.get('primary_key', df.columns[0])  # Default to first column
            temporal_fields = data.get('temporal_fields', [])
            
            # Create nodes
            table_nodes = self.create_nodes_from_table(df, table_name, primary_key, temporal_fields)
            all_nodes.update(table_nodes)
            
            # Detect and create edges
            relationships = self.detect_foreign_key_relationships(df, table_name, dataframes)
            table_edges = self.create_edges_from_relationships(df, table_name, primary_key, 
                                                             relationships, temporal_fields)
            all_edges.extend(table_edges)
        
        # Build NetworkX graph for analysis
        self.build_networkx_graph(all_nodes, all_edges)
        
        # Return graph data
        return {
            'nodes': [asdict(node) for node in all_nodes.values()],
            'edges': [asdict(edge) for edge in all_edges],
            'statistics': self.get_graph_statistics(),
            'node_types': list(set(node.type for node in all_nodes.values())),
            'edge_types': list(set(edge.edge_type for edge in all_edges))
        }

    def build_networkx_graph(self, nodes: Dict[str, GraphNode], edges: List[GraphEdge]):
        """Build NetworkX graph for analysis"""
        self.graph = nx.MultiDiGraph()
        
        # Add nodes
        for node in nodes.values():
            self.graph.add_node(node.id, type=node.type, **node.properties)
        
        # Add edges
        for edge in edges:
            self.graph.add_edge(
                edge.source_id, 
                edge.target_id, 
                type=edge.edge_type,
                timestamp=edge.timestamp,
                **edge.properties
            )

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Calculate graph statistics"""
        if not self.graph.nodes():
            return {}
            
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types_count': len(set(nx.get_node_attributes(self.graph, 'type').values())),
            'edge_types_count': len(set(nx.get_edge_attributes(self.graph, 'type').values())),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'is_connected': nx.is_weakly_connected(self.graph),
            'number_of_components': nx.number_weakly_connected_components(self.graph)
        }