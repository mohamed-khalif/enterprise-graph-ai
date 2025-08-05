import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, RGCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json

@dataclass
class PredictionResult:
    """Results from GNN prediction"""
    node_id: str
    prediction: float
    confidence: float
    risk_level: str
    explanation: Dict[str, Any]

class GraphFeatureExtractor:
    """Extract features from graph structure and node properties"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def extract_structural_features(self, G: nx.Graph, node_id: str) -> Dict[str, float]:
        """Extract graph structural features for a node"""
        features = {}
        
        # Basic centrality measures
        try:
            features['degree_centrality'] = nx.degree_centrality(G)[node_id]
            features['betweenness_centrality'] = nx.betweenness_centrality(G)[node_id]
            features['closeness_centrality'] = nx.closeness_centrality(G)[node_id]
            features['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)[node_id]
        except:
            # Fallback for disconnected graphs
            features['degree_centrality'] = G.degree(node_id) / max(1, len(G.nodes()) - 1)
            features['betweenness_centrality'] = 0.0
            features['closeness_centrality'] = 0.0
            features['eigenvector_centrality'] = 0.0
            
        # Local structure features
        features['degree'] = G.degree(node_id)
        features['clustering_coefficient'] = nx.clustering(G, node_id)
        
        # Neighbor statistics
        neighbors = list(G.neighbors(node_id))
        features['num_neighbors'] = len(neighbors)
        if neighbors:
            neighbor_degrees = [G.degree(n) for n in neighbors]
            features['avg_neighbor_degree'] = np.mean(neighbor_degrees)
            features['max_neighbor_degree'] = np.max(neighbor_degrees)
        else:
            features['avg_neighbor_degree'] = 0.0
            features['max_neighbor_degree'] = 0.0
            
        return features
    
    def extract_node_features(self, node_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from node properties"""
        features = {}
        
        # Numerical features
        numerical_fields = ['quantity', 'unit_price', 'total_amount', 'discount_amount', 
                        'rating', 'shipping_cost', 'tax_amount', 'zip_code']
        
        for field in numerical_fields:
            if field in node_data:
                try:
                    features[field] = float(node_data[field])
                except (ValueError, TypeError):
                    features[field] = 0.0
            else:
                features[field] = 0.0
        
        # Categorical features (encoded as counts/flags) - with safe checking
        try:
            if 'payment_method' in node_data and node_data['payment_method']:
                payment_method = str(node_data['payment_method'])
                features['is_credit_card'] = 1.0 if 'Credit Card' in payment_method else 0.0
                features['is_paypal'] = 1.0 if 'PayPal' in payment_method else 0.0
                features['is_apple_pay'] = 1.0 if 'Apple Pay' in payment_method else 0.0
            else:
                features['is_credit_card'] = 0.0
                features['is_paypal'] = 0.0
                features['is_apple_pay'] = 0.0
        except (KeyError, TypeError, AttributeError):
            features['is_credit_card'] = 0.0
            features['is_paypal'] = 0.0
            features['is_apple_pay'] = 0.0
        
        try:
            if 'order_status' in node_data and node_data['order_status']:
                order_status = str(node_data['order_status'])
                features['is_shipped'] = 1.0 if order_status == 'Shipped' else 0.0
                features['is_delivered'] = 1.0 if order_status == 'Delivered' else 0.0
                features['is_cancelled'] = 1.0 if order_status == 'Cancelled' else 0.0
                features['is_processing'] = 1.0 if order_status == 'Processing' else 0.0
            else:
                features['is_shipped'] = 0.0
                features['is_delivered'] = 0.0
                features['is_cancelled'] = 0.0
                features['is_processing'] = 0.0
        except (KeyError, TypeError, AttributeError):
            features['is_shipped'] = 0.0
            features['is_delivered'] = 0.0
            features['is_cancelled'] = 0.0
            features['is_processing'] = 0.0
        
        # VIP status features (for customer nodes) - with safe checking
        try:
            if 'vip_status' in node_data and node_data['vip_status']:
                vip_status = str(node_data['vip_status'])
                features['is_vip_gold'] = 1.0 if vip_status == 'gold' else 0.0
                features['is_vip_platinum'] = 1.0 if vip_status == 'platinum' else 0.0
                features['is_vip_silver'] = 1.0 if vip_status == 'silver' else 0.0
            else:
                features['is_vip_gold'] = 0.0
                features['is_vip_platinum'] = 0.0
                features['is_vip_silver'] = 0.0
        except (KeyError, TypeError, AttributeError):
            features['is_vip_gold'] = 0.0
            features['is_vip_platinum'] = 0.0
            features['is_vip_silver'] = 0.0
        
        # Category features - with safe checking
        try:
            if 'category' in node_data and node_data['category']:
                category = str(node_data['category'])
                features['is_electronics'] = 1.0 if category == 'Electronics' else 0.0
                features['is_clothing'] = 1.0 if category == 'Clothing' else 0.0
                features['is_fitness'] = 1.0 if category == 'Fitness' else 0.0
            else:
                features['is_electronics'] = 0.0
                features['is_clothing'] = 0.0
                features['is_fitness'] = 0.0
        except (KeyError, TypeError, AttributeError):
            features['is_electronics'] = 0.0
            features['is_clothing'] = 0.0
            features['is_fitness'] = 0.0
        
        return features
    
    def create_feature_matrix(self, graph_data: Dict[str, Any]) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """Create feature matrix for all nodes"""
        
        # Build NetworkX graph for structural analysis
        G = nx.Graph()
        
        # Add nodes
        node_map = {}
        for i, node in enumerate(graph_data['nodes']):
            G.add_node(node['id'])
            node_map[node['id']] = i
        
        # Add edges
        for edge in graph_data['edges']:
            if edge['source_id'] in node_map and edge['target_id'] in node_map:
                G.add_edge(edge['source_id'], edge['target_id'])
        
        # Extract features for each node
        all_features = []
        node_ids = []
        
        for node in graph_data['nodes']:
            node_id = node['id']
            node_ids.append(node_id)
            
            # Combine structural and property features
            structural_features = self.extract_structural_features(G, node_id)
            property_features = self.extract_node_features(node['properties'])
            
            # Merge all features
            combined_features = {**structural_features, **property_features}
            all_features.append(combined_features)
        
        # Convert to matrix
        feature_names = list(all_features[0].keys())
        feature_matrix = np.array([[features[name] for name in feature_names] for features in all_features])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Normalize features
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        return torch.FloatTensor(feature_matrix), feature_names, node_map

class FraudDetectionGNN(nn.Module):
    """GNN model for fraud detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(FraudDetectionGNN, self).__init__()
        self.num_layers = num_layers
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(input_dim, hidden_dim, num_layers=2, out_channels=hidden_dim))
        
        # Additional layers for better representation
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=2, out_channels=hidden_dim))
        
        # Fraud prediction head
        self.fraud_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Fraud prediction
        fraud_score = self.fraud_classifier(x)
        
        return fraud_score.squeeze()

class ChurnPredictionGNN(nn.Module):
    """GNN model for customer churn prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(ChurnPredictionGNN, self).__init__()
        self.num_layers = num_layers
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(input_dim, hidden_dim, num_layers=2, out_channels=hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=2, out_channels=hidden_dim))
        
        # Churn prediction head
        self.churn_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Churn prediction
        churn_score = self.churn_classifier(x)
        
        return churn_score.squeeze()

class GNNPipeline:
    """Main GNN pipeline for predictions"""
    
    def __init__(self):
        self.feature_extractor = GraphFeatureExtractor()
        self.fraud_model = None
        self.churn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_graph_data(self, graph_data: Dict[str, Any]) -> Data:
        """Convert graph data to PyTorch Geometric format"""
        
        # Extract features
        x, feature_names, node_map = self.feature_extractor.create_feature_matrix(graph_data)
        
        # Create edge index
        edge_list = []
        edge_weights = []
        
        for edge in graph_data['edges']:
            if edge['source_id'] in node_map and edge['target_id'] in node_map:
                source_idx = node_map[edge['source_id']]
                target_idx = node_map[edge['target_id']]
                
                edge_list.append([source_idx, target_idx])
                edge_list.append([target_idx, source_idx])  # Make undirected
                
                # Use edge properties as weights (e.g., transaction amount)
                weight = edge['properties'].get('total_amount', 1.0)
                edge_weights.extend([weight, weight])
        
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_attr = torch.FloatTensor(edge_weights)
        else:
            edge_index = torch.LongTensor([[], []])
            edge_attr = torch.FloatTensor([])
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data, feature_names, node_map
    
    def generate_synthetic_labels(self, graph_data: Dict[str, Any], task: str = 'fraud') -> Dict[str, float]:
        """Generate synthetic labels for demonstration (replace with real labels in production)"""
        labels = {}
        
        for node in graph_data['nodes']:
            node_id = node['id']
            properties = node['properties']
            
            if task == 'fraud':
                # Synthetic fraud detection based on suspicious patterns
                fraud_score = 0.0
                
                # High transaction amounts
                if 'total_amount' in properties and properties['total_amount'] > 1000:
                    fraud_score += 0.3
                
                # Cancelled orders are suspicious
                if properties.get('order_status') == 'Cancelled':
                    fraud_score += 0.4
                
                # Multiple high-value orders from same customer quickly
                if 'customer_id' in properties:
                    fraud_score += 0.2
                
                # Random noise
                fraud_score += np.random.normal(0, 0.1)
                labels[node_id] = min(max(fraud_score, 0.0), 1.0)
                
            elif task == 'churn':
                # Synthetic churn prediction based on behavior patterns
                churn_score = 0.0
                
                # Low ratings indicate dissatisfaction
                if 'rating' in properties and properties['rating'] < 4.0:
                    churn_score += 0.3
                
                # Cancelled orders indicate problems
                if properties.get('order_status') == 'Cancelled':
                    churn_score += 0.4
                
                # Bronze VIP customers more likely to churn
                if properties.get('vip_status') == 'bronze':
                    churn_score += 0.2
                
                # Random noise
                churn_score += np.random.normal(0, 0.1)
                labels[node_id] = min(max(churn_score, 0.0), 1.0)
        
        return labels
    
    def train_model(self, graph_data: Dict[str, Any], task: str = 'fraud', epochs: int = 50):
        """Train GNN model on graph data"""
        
        # Prepare data
        data, feature_names, node_map = self.prepare_graph_data(graph_data)
        
        # Generate synthetic labels (replace with real labels)
        labels_dict = self.generate_synthetic_labels(graph_data, task)
        labels = torch.FloatTensor([labels_dict.get(node['id'], 0.0) for node in graph_data['nodes']])
        
        # Initialize model
        input_dim = data.x.shape[1]
        
        if task == 'fraud':
            model = FraudDetectionGNN(input_dim)
        else:
            model = ChurnPredictionGNN(input_dim)
        
        model = model.to(self.device)
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.BCELoss()
        
        # Simple training loop (in production, use proper train/val split)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = criterion(out, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Store trained model
        if task == 'fraud':
            self.fraud_model = model
        else:
            self.churn_model = model
        
        return model, feature_names, node_map
    
    def predict(self, graph_data: Dict[str, Any], task: str = 'fraud') -> List[PredictionResult]:
        """Make predictions on graph data"""
        
        # Always prepare data first
        data, feature_names, node_map = self.prepare_graph_data(graph_data)
        data = data.to(self.device)
        
        # Select model
        model = self.fraud_model if task == 'fraud' else self.churn_model
        
        if model is None:
            # Train model if not already trained
            print(f"Training {task} model...")
            model, feature_names, node_map = self.train_model(graph_data, task)
            # Re-prepare data after training
            data, feature_names, node_map = self.prepare_graph_data(graph_data)
            data = data.to(self.device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(data.x, data.edge_index)
            predictions = predictions.cpu().numpy()
        
        # Convert to results
        results = []
        for i, node in enumerate(graph_data['nodes']):
            score = float(predictions[i])
            
            # Determine risk level
            if score > 0.7:
                risk_level = "HIGH"
            elif score > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Create explanation
            explanation = {
                "score": score,
                "factors": self._explain_prediction(node, score, task),
                "node_type": node['type'],
                "source_table": node['source_table']
            }
            
            result = PredictionResult(
                node_id=node['id'],
                prediction=score,
                confidence=min(abs(score - 0.5) * 2, 1.0),  # Distance from neutral
                risk_level=risk_level,
                explanation=explanation
            )
            
            results.append(result)
        
        return results
    
    def _explain_prediction(self, node: Dict, score: float, task: str) -> List[str]:
        """Generate explanation for prediction"""
        factors = []
        properties = node['properties']
        
        if task == 'fraud':
            if properties.get('total_amount', 0) > 1000:
                factors.append("High transaction amount")
            if properties.get('order_status') == 'Cancelled':
                factors.append("Order was cancelled")
            if score > 0.5:
                factors.append("Suspicious network patterns detected")
        
        elif task == 'churn':
            if properties.get('rating', 5) < 4.0:
                factors.append("Low customer rating")
            if properties.get('order_status') == 'Cancelled':
                factors.append("Recent order cancellation")
            if properties.get('vip_status') == 'bronze':
                factors.append("Low-tier customer status")
        
        return factors if factors else ["Based on graph structure analysis"]