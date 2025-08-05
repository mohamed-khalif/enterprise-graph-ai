import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from enum import Enum

class UseCaseType(Enum):
    FRAUD_DETECTION = "fraud_detection"
    CUSTOMER_CHURN = "customer_churn" 
    PAYMENT_DEFAULTS = "payment_defaults"
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    EMPLOYEE_RETENTION = "employee_retention"
    MARKETING_ATTRIBUTION = "marketing_attribution"

@dataclass
class BusinessInsight:
    """Structured business insight with actionable recommendations"""
    insight_type: str
    entity_id: str
    entity_name: str
    risk_score: float
    risk_level: str  # HIGH, MEDIUM, LOW
    business_impact: str
    recommended_actions: List[str]
    confidence: float
    explanation: str
    financial_impact: Optional[float] = None
    urgency: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    department: str = "General"

@dataclass
class UseCaseResult:
    """Results from a specific use case analysis"""
    use_case: str
    title: str
    description: str
    insights: List[BusinessInsight]
    summary_metrics: Dict[str, Any]
    recommended_actions: List[str]
    roi_estimate: Optional[float] = None
    processing_time: float = 0.0

class FraudDetectionModule:
    """Pre-built fraud detection for e-commerce and financial services"""
    
    def __init__(self):
        self.name = "Fraud Detection Engine"
        self.description = "AI-powered transaction fraud detection using network analysis"
        
    def analyze(self, graph_data: Dict[str, Any], domain_context: str = "ecommerce") -> UseCaseResult:
        """Run fraud detection analysis"""
        
        insights = []
        
        # Extract transaction-like entities
        transaction_nodes = [node for node in graph_data.get('nodes', []) 
                           if self._is_transaction_node(node)]
        
        for node in transaction_nodes:
            risk_factors = self._analyze_fraud_risk_factors(node, graph_data)
            risk_score = self._calculate_fraud_risk_score(risk_factors)
            
            if risk_score > 0.3:  # Only flag potential fraud cases
                insight = BusinessInsight(
                    insight_type="fraud_alert",
                    entity_id=node['id'],
                    entity_name=self._get_entity_display_name(node),
                    risk_score=risk_score,
                    risk_level=self._get_risk_level(risk_score),
                    business_impact=self._estimate_fraud_impact(node, risk_score),
                    recommended_actions=self._get_fraud_actions(risk_factors, domain_context),
                    confidence=min(risk_score * 1.2, 1.0),
                    explanation=self._explain_fraud_risk(risk_factors),
                    financial_impact=self._estimate_financial_fraud_impact(node),
                    urgency=self._get_fraud_urgency(risk_score),
                    department="Risk Management"
                )
                insights.append(insight)
        
        # Calculate summary metrics
        summary_metrics = {
            "total_transactions_analyzed": len(transaction_nodes),
            "high_risk_transactions": len([i for i in insights if i.risk_level == "HIGH"]),
            "medium_risk_transactions": len([i for i in insights if i.risk_level == "MEDIUM"]),
            "estimated_fraud_prevented": sum([i.financial_impact or 0 for i in insights]),
            "average_risk_score": np.mean([i.risk_score for i in insights]) if insights else 0,
            "alert_rate": len(insights) / len(transaction_nodes) if transaction_nodes else 0
        }
        
        return UseCaseResult(
            use_case="fraud_detection",
            title="🛡️ Fraud Detection Results",
            description="AI-powered analysis identifying suspicious transactions and patterns",
            insights=insights,
            summary_metrics=summary_metrics,
            recommended_actions=self._get_overall_fraud_recommendations(insights),
            roi_estimate=summary_metrics["estimated_fraud_prevented"] * 0.8  # 80% prevention rate
        )
    
    def _is_transaction_node(self, node: Dict) -> bool:
        """Determine if node represents a transaction"""
        node_id = node['id'].lower()
        properties = node.get('properties', {})
        
        return (
            'order' in node_id or 'transaction' in node_id or 'payment' in node_id or
            'total_amount' in properties or 'amount' in properties or
            'order_status' in properties or 'payment_method' in properties
        )
    
    def _analyze_fraud_risk_factors(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze various fraud risk factors"""
        properties = node.get('properties', {})
        risk_factors = {}
        
        # High amount risk
        amount = properties.get('total_amount', properties.get('amount', 0))
        if amount > 1000:
            risk_factors['high_amount'] = min(amount / 5000, 1.0)
        
        # Cancelled orders
        if properties.get('order_status') == 'Cancelled':
            risk_factors['cancelled_order'] = 0.8
        
        # Payment method risk
        payment_method = properties.get('payment_method', '')
        if 'Apple Pay' in payment_method or 'PayPal' in payment_method:
            risk_factors['secure_payment'] = -0.2  # Lower risk
        
        # Unusual timing
        created_at = properties.get('created_at', properties.get('order_date'))
        if created_at:
            risk_factors.update(self._analyze_timing_patterns(created_at))
        
        # Network analysis
        risk_factors.update(self._analyze_network_patterns(node, graph_data))
        
        return risk_factors
    
    def _analyze_timing_patterns(self, timestamp_str: str) -> Dict[str, float]:
        """Analyze timing-based fraud indicators"""
        try:
            if isinstance(timestamp_str, str):
                timestamp = pd.to_datetime(timestamp_str)
            else:
                timestamp = timestamp_str
                
            patterns = {}
            
            # Weekend transactions (slightly higher risk)
            if timestamp.weekday() >= 5:
                patterns['weekend_transaction'] = 0.1
            
            # Late night transactions (higher risk)
            if timestamp.hour < 6 or timestamp.hour > 22:
                patterns['unusual_hour'] = 0.3
                
            return patterns
        except:
            return {}
    
    def _analyze_network_patterns(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze network-based fraud indicators"""
        patterns = {}
        
        # Find related edges
        related_edges = [edge for edge in graph_data.get('edges', []) 
                        if edge['source_id'] == node['id'] or edge['target_id'] == node['id']]
        
        # Multiple rapid transactions from same customer
        customer_edges = [edge for edge in related_edges if 'customer' in edge.get('target_id', '').lower()]
        if len(customer_edges) > 3:
            patterns['rapid_customer_activity'] = 0.4
        
        return patterns
    
    def _calculate_fraud_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall fraud risk score"""
        if not risk_factors:
            return 0.0
        
        # Weighted sum of risk factors
        weights = {
            'high_amount': 0.3,
            'cancelled_order': 0.4,
            'secure_payment': 0.2,
            'unusual_hour': 0.3,
            'weekend_transaction': 0.1,
            'rapid_customer_activity': 0.4
        }
        
        score = sum(risk_factors.get(factor, 0) * weight 
                   for factor, weight in weights.items())
        
        return min(max(score, 0), 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _estimate_fraud_impact(self, node: Dict, risk_score: float) -> str:
        properties = node.get('properties', {})
        amount = properties.get('total_amount', properties.get('amount', 0))
        
        if risk_score > 0.7:
            return f"HIGH RISK: Potential loss of £{amount:,.2f}"
        elif risk_score > 0.4:
            return f"MEDIUM RISK: Potential loss of £{amount:,.2f}"
        else:
            return f"LOW RISK: Minimal financial exposure"
    
    def _get_fraud_actions(self, risk_factors: Dict, domain: str) -> List[str]:
        actions = []
        
        if 'high_amount' in risk_factors:
            actions.append("⚠️ Verify customer identity and payment method")
        
        if 'cancelled_order' in risk_factors:
            actions.append("🔍 Investigate cancellation reason and customer history")
        
        if 'rapid_customer_activity' in risk_factors:
            actions.append("📊 Review customer transaction patterns")
        
        if 'unusual_hour' in risk_factors:
            actions.append("🕐 Flag for manual review due to unusual timing")
        
        if not actions:
            actions.append("✅ Monitor transaction for unusual patterns")
        
        return actions
    
    def _explain_fraud_risk(self, risk_factors: Dict) -> str:
        explanations = []
        
        if 'high_amount' in risk_factors:
            explanations.append("unusually high transaction amount")
        if 'cancelled_order' in risk_factors:
            explanations.append("order was cancelled")
        if 'unusual_hour' in risk_factors:
            explanations.append("transaction occurred at unusual hour")
        if 'rapid_customer_activity' in risk_factors:
            explanations.append("multiple rapid transactions from customer")
        
        if explanations:
            return f"Flagged due to: {', '.join(explanations)}"
        else:
            return "Low-risk transaction with normal patterns"
    
    def _estimate_financial_fraud_impact(self, node: Dict) -> float:
        properties = node.get('properties', {})
        return float(properties.get('total_amount', properties.get('amount', 0)))
    
    def _get_fraud_urgency(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_entity_display_name(self, node: Dict) -> str:
        node_id = node['id']
        if '_' in node_id:
            return node_id.split('_')[-1]
        return node_id
    
    def _get_overall_fraud_recommendations(self, insights: List[BusinessInsight]) -> List[str]:
        recommendations = []
        
        high_risk_count = len([i for i in insights if i.risk_level == "HIGH"])
        if high_risk_count > 0:
            recommendations.append(f"🚨 Immediately review {high_risk_count} high-risk transactions")
        
        total_impact = sum([i.financial_impact or 0 for i in insights])
        if total_impact > 10000:
            recommendations.append(f"💰 Potential fraud exposure: £{total_impact:,.2f}")
        
        recommendations.extend([
            "🔧 Implement real-time fraud monitoring",
            "📊 Review fraud detection thresholds monthly",
            "👥 Train customer service team on fraud indicators"
        ])
        
        return recommendations

class CustomerChurnModule:
    """Pre-built customer churn prediction for SaaS and subscription businesses"""
    
    def __init__(self):
        self.name = "Customer Churn Predictor"
        self.description = "AI-powered churn prediction using behavioral and network signals"
    
    def analyze(self, graph_data: Dict[str, Any], domain_context: str = "saas") -> UseCaseResult:
        """Run churn prediction analysis"""
        
        insights = []
        
        # Extract customer nodes
        customer_nodes = [node for node in graph_data.get('nodes', []) 
                         if self._is_customer_node(node)]
        
        for node in customer_nodes:
            churn_factors = self._analyze_churn_risk_factors(node, graph_data)
            churn_score = self._calculate_churn_risk_score(churn_factors)
            
            if churn_score > 0.3:  # Only flag at-risk customers
                insight = BusinessInsight(
                    insight_type="churn_risk",
                    entity_id=node['id'],
                    entity_name=self._get_customer_display_name(node),
                    risk_score=churn_score,
                    risk_level=self._get_risk_level(churn_score),
                    business_impact=self._estimate_churn_impact(node, churn_score, domain_context),
                    recommended_actions=self._get_churn_actions(churn_factors, domain_context),
                    confidence=min(churn_score * 1.1, 1.0),
                    explanation=self._explain_churn_risk(churn_factors),
                    financial_impact=self._estimate_financial_churn_impact(node, domain_context),
                    urgency=self._get_churn_urgency(churn_score),
                    department="Customer Success"
                )
                insights.append(insight)
        
        # Calculate summary metrics
        summary_metrics = {
            "total_customers_analyzed": len(customer_nodes),
            "high_churn_risk": len([i for i in insights if i.risk_level == "HIGH"]),
            "medium_churn_risk": len([i for i in insights if i.risk_level == "MEDIUM"]),
            "estimated_revenue_at_risk": sum([i.financial_impact or 0 for i in insights]),
            "churn_rate_predicted": len(insights) / len(customer_nodes) if customer_nodes else 0,
            "average_customer_value": np.mean([self._estimate_financial_churn_impact(node, domain_context) 
                                             for node in customer_nodes])
        }
        
        return UseCaseResult(
            use_case="customer_churn",
            title="📉 Customer Churn Prediction",
            description="AI-powered analysis identifying customers at risk of churning",
            insights=insights,
            summary_metrics=summary_metrics,
            recommended_actions=self._get_overall_churn_recommendations(insights),
            roi_estimate=summary_metrics["estimated_revenue_at_risk"] * 0.3  # 30% retention improvement
        )
    
    def _is_customer_node(self, node: Dict) -> bool:
        """Determine if node represents a customer"""
        return (
            'customer' in node['id'].lower() or 
            node.get('type') == 'customer' or
            'vip_status' in node.get('properties', {})
        )
    
    def _analyze_churn_risk_factors(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze various churn risk factors"""
        properties = node.get('properties', {})
        risk_factors = {}
        
        # VIP status (lower tier = higher churn risk)
        vip_status = properties.get('vip_status', 'bronze').lower()
        if vip_status == 'bronze':
            risk_factors['low_tier_customer'] = 0.4
        elif vip_status == 'silver':
            risk_factors['medium_tier_customer'] = 0.2
        else:  # gold, platinum
            risk_factors['high_tier_customer'] = -0.3
        
        # Customer lifetime value
        ltv = properties.get('lifetime_value', 0)
        if ltv < 500:
            risk_factors['low_value_customer'] = 0.3
        
        # Signup recency (new customers at higher risk)
        signup_date = properties.get('signup_date')
        if signup_date:
            risk_factors.update(self._analyze_customer_tenure(signup_date))
        
        # Activity analysis
        risk_factors.update(self._analyze_customer_activity(node, graph_data))
        
        return risk_factors
    
    def _analyze_customer_tenure(self, signup_date_str: str) -> Dict[str, float]:
        """Analyze customer tenure patterns"""
        try:
            signup_date = pd.to_datetime(signup_date_str)
            tenure_days = (datetime.now() - signup_date).days
            
            patterns = {}
            
            # New customers (higher churn risk)
            if tenure_days < 30:
                patterns['new_customer'] = 0.5
            elif tenure_days < 90:
                patterns['recent_customer'] = 0.3
            elif tenure_days > 365:
                patterns['loyal_customer'] = -0.2
                
            return patterns
        except:
            return {}
    
    def _analyze_customer_activity(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze customer activity patterns"""
        patterns = {}
        
        # Find customer's orders/transactions
        customer_orders = [edge for edge in graph_data.get('edges', []) 
                          if edge['target_id'] == node['id']]
        
        # Low activity = higher churn risk
        if len(customer_orders) < 2:
            patterns['low_activity'] = 0.4
        elif len(customer_orders) > 5:
            patterns['high_activity'] = -0.3
        
        # Recent cancellations
        recent_cancellations = 0
        for edge in customer_orders:
            edge_props = edge.get('properties', {})
            if 'Cancelled' in str(edge_props.get('order_status', '')):
                recent_cancellations += 1
        
        if recent_cancellations > 0:
            patterns['recent_cancellations'] = 0.6
        
        return patterns
    
    def _calculate_churn_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall churn risk score"""
        if not risk_factors:
            return 0.2  # Default low risk
        
        weights = {
            'low_tier_customer': 0.3,
            'low_value_customer': 0.2,
            'new_customer': 0.4,
            'recent_customer': 0.2,
            'loyal_customer': 0.3,
            'low_activity': 0.4,
            'high_activity': 0.3,
            'recent_cancellations': 0.5
        }
        
        score = 0.2 + sum(risk_factors.get(factor, 0) * weight 
                         for factor, weight in weights.items())
        
        return min(max(score, 0), 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _estimate_churn_impact(self, node: Dict, churn_score: float, domain: str) -> str:
        ltv = node.get('properties', {}).get('lifetime_value', 1000)
        
        if domain == "saas":
            monthly_value = ltv / 12
            return f"Potential monthly revenue loss: £{monthly_value:,.2f}"
        else:
            return f"Potential customer value loss: £{ltv:,.2f}"
    
    def _get_churn_actions(self, risk_factors: Dict, domain: str) -> List[str]:
        actions = []
        
        if 'new_customer' in risk_factors:
            actions.append("🎯 Implement onboarding campaign")
        
        if 'low_activity' in risk_factors:
            actions.append("📧 Send re-engagement email sequence")
        
        if 'recent_cancellations' in risk_factors:
            actions.append("☎️ Schedule customer success call")
        
        if 'low_tier_customer' in risk_factors:
            actions.append("⬆️ Offer upgrade incentives")
        
        if domain == "saas":
            actions.append("📊 Review product usage analytics")
        
        return actions or ["✅ Monitor customer health score"]
    
    def _explain_churn_risk(self, risk_factors: Dict) -> str:
        explanations = []
        
        if 'new_customer' in risk_factors:
            explanations.append("new customer (higher churn risk)")
        if 'low_tier_customer' in risk_factors:
            explanations.append("bronze tier customer")
        if 'low_activity' in risk_factors:
            explanations.append("low purchase activity")
        if 'recent_cancellations' in risk_factors:
            explanations.append("recent order cancellations")
        
        return f"Risk factors: {', '.join(explanations)}" if explanations else "Standard customer profile"
    
    def _estimate_financial_churn_impact(self, node: Dict, domain: str) -> float:
        ltv = node.get('properties', {}).get('lifetime_value', 1000)
        
        if domain == "saas":
            return ltv / 12  # Monthly recurring revenue
        else:
            return ltv * 0.5  # Estimated annual value
    
    def _get_churn_urgency(self, churn_score: float) -> str:
        if churn_score >= 0.8:
            return "HIGH"
        elif churn_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_customer_display_name(self, node: Dict) -> str:
        props = node.get('properties', {})
        return props.get('customer_name', node['id'].split('_')[-1])
    
    def _get_overall_churn_recommendations(self, insights: List[BusinessInsight]) -> List[str]:
        recommendations = []
        
        high_risk_count = len([i for i in insights if i.risk_level == "HIGH"])
        if high_risk_count > 0:
            recommendations.append(f"🚨 Immediate intervention needed for {high_risk_count} high-risk customers")
        
        total_revenue_risk = sum([i.financial_impact or 0 for i in insights])
        recommendations.append(f"💰 Revenue at risk: £{total_revenue_risk:,.2f}/month")
        
        recommendations.extend([
            "📞 Implement proactive customer success outreach",
            "🎁 Design retention offers for at-risk segments",
            "📊 Monitor customer health scores weekly"
        ])
        
        return recommendations

class PaymentDefaultsModule:
    """Pre-built payment default prediction for lending and financial services"""
    
    def __init__(self):
        self.name = "Payment Default Predictor"
        self.description = "AI-powered default risk assessment for loans and credit"
    
    def analyze(self, graph_data: Dict[str, Any], domain_context: str = "lending") -> UseCaseResult:
        """Run payment default risk analysis"""
        
        insights = []
        
        # Look for loan/payment related entities
        borrower_nodes = [node for node in graph_data.get('nodes', []) 
                         if self._is_borrower_node(node)]
        
        for node in borrower_nodes:
            default_factors = self._analyze_default_risk_factors(node, graph_data)
            default_score = self._calculate_default_risk_score(default_factors)
            
            if default_score > 0.3:  # Flag risky borrowers
                insight = BusinessInsight(
                    insight_type="default_risk",
                    entity_id=node['id'],
                    entity_name=self._get_borrower_display_name(node),
                    risk_score=default_score,
                    risk_level=self._get_risk_level(default_score),
                    business_impact=self._estimate_default_impact(node, default_score),
                    recommended_actions=self._get_default_actions(default_factors, domain_context),
                    confidence=min(default_score * 1.15, 1.0),
                    explanation=self._explain_default_risk(default_factors),
                    financial_impact=self._estimate_financial_default_impact(node),
                    urgency=self._get_default_urgency(default_score),
                    department="Risk Management"
                )
                insights.append(insight)
        
        summary_metrics = {
            "total_borrowers_analyzed": len(borrower_nodes),
            "high_default_risk": len([i for i in insights if i.risk_level == "HIGH"]),
            "medium_default_risk": len([i for i in insights if i.risk_level == "MEDIUM"]),
            "total_exposure_at_risk": sum([i.financial_impact or 0 for i in insights]),
            "default_rate_predicted": len(insights) / len(borrower_nodes) if borrower_nodes else 0
        }
        
        return UseCaseResult(
            use_case="payment_defaults",
            title="💳 Payment Default Risk Assessment",
            description="AI-powered analysis identifying borrowers at risk of default",
            insights=insights,
            summary_metrics=summary_metrics,
            recommended_actions=self._get_overall_default_recommendations(insights),
            roi_estimate=summary_metrics["total_exposure_at_risk"] * 0.4  # 40% loss prevention
        )
    
    def _is_borrower_node(self, node: Dict) -> bool:
        """Determine if node represents a borrower/customer with payment obligations"""
        properties = node.get('properties', {})
        return (
            'customer' in node['id'].lower() or
            'payment_method' in properties or
            'total_amount' in properties or
            any(key in properties for key in ['credit_score', 'loan_amount', 'monthly_payment'])
        )
    
    def _analyze_default_risk_factors(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze payment default risk factors"""
        properties = node.get('properties', {})
        risk_factors = {}
        
        # Payment history analysis
        risk_factors.update(self._analyze_payment_history(node, graph_data))
        
        # Amount-based risk
        total_amount = properties.get('total_amount', properties.get('loan_amount', 0))
        if total_amount > 5000:
            risk_factors['high_exposure'] = min(total_amount / 20000, 0.4)
        
        # Customer tier (proxy for creditworthiness)
        vip_status = properties.get('vip_status', 'bronze').lower()
        if vip_status == 'bronze':
            risk_factors['low_creditworthiness'] = 0.3
        elif vip_status in ['gold', 'platinum']:
            risk_factors['high_creditworthiness'] = -0.2
        
        return risk_factors
    
    def _analyze_payment_history(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze payment behavior patterns"""
        patterns = {}
        
    def _analyze_payment_history(self, node: Dict, graph_data: Dict) -> Dict[str, float]:
        """Analyze payment behavior patterns"""
        patterns = {}
        
        # Find related payment transactions
        related_transactions = [edge for edge in graph_data.get('edges', []) 
                              if edge['target_id'] == node['id'] or edge['source_id'] == node['id']]
        
        # Count failed/cancelled payments
        failed_payments = 0
        total_payments = len(related_transactions)
        
        for transaction in related_transactions:
            props = transaction.get('properties', {})
            status = props.get('order_status', props.get('payment_status', ''))
            
            if 'Cancel' in status or 'Failed' in status or 'Declined' in status:
                failed_payments += 1
        
        if total_payments > 0:
            failure_rate = failed_payments / total_payments
            if failure_rate > 0.2:
                patterns['high_payment_failure_rate'] = failure_rate
            elif failure_rate == 0:
                patterns['perfect_payment_history'] = -0.3
        
        return patterns
    
    def _calculate_default_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall default risk score"""
        if not risk_factors:
            return 0.2  # Default baseline risk
        
        weights = {
            'high_exposure': 0.3,
            'low_creditworthiness': 0.4,
            'high_creditworthiness': 0.3,
            'high_payment_failure_rate': 0.5,
            'perfect_payment_history': 0.4
        }
        
        score = 0.2 + sum(risk_factors.get(factor, 0) * weight 
                         for factor, weight in weights.items())
        
        return min(max(score, 0), 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _estimate_default_impact(self, node: Dict, default_score: float) -> str:
        properties = node.get('properties', {})
        exposure = properties.get('total_amount', properties.get('loan_amount', 1000))
        
        expected_loss = exposure * default_score
        return f"Expected loss: £{expected_loss:,.2f} (exposure: £{exposure:,.2f})"
    
    def _get_default_actions(self, risk_factors: Dict, domain: str) -> List[str]:
        actions = []
        
        if 'high_payment_failure_rate' in risk_factors:
            actions.append("📞 Contact customer to update payment method")
        
        if 'high_exposure' in risk_factors:
            actions.append("📋 Require additional documentation/collateral")
        
        if 'low_creditworthiness' in risk_factors:
            actions.append("⚠️ Consider reducing credit limit or requiring guarantor")
        
        actions.extend([
            "📊 Monitor payment patterns closely",
            "🔍 Review credit bureau reports"
        ])
        
        return actions
    
    def _explain_default_risk(self, risk_factors: Dict) -> str:
        explanations = []
        
        if 'high_exposure' in risk_factors:
            explanations.append("high loan/credit amount")
        if 'low_creditworthiness' in risk_factors:
            explanations.append("lower credit tier customer")
        if 'high_payment_failure_rate' in risk_factors:
            explanations.append("history of payment failures")
        if 'perfect_payment_history' in risk_factors:
            explanations.append("excellent payment history")
        
        return f"Risk assessment based on: {', '.join(explanations)}" if explanations else "Standard risk profile"
    
    def _estimate_financial_default_impact(self, node: Dict) -> float:
        properties = node.get('properties', {})
        return float(properties.get('total_amount', properties.get('loan_amount', 1000)))
    
    def _get_default_urgency(self, default_score: float) -> str:
        if default_score >= 0.8:
            return "HIGH"
        elif default_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_borrower_display_name(self, node: Dict) -> str:
        props = node.get('properties', {})
        return props.get('customer_name', props.get('borrower_name', node['id'].split('_')[-1]))
    
    def _get_overall_default_recommendations(self, insights: List[BusinessInsight]) -> List[str]:
        recommendations = []
        
        high_risk_count = len([i for i in insights if i.risk_level == "HIGH"])
        if high_risk_count > 0:
            recommendations.append(f"🚨 Immediate risk assessment required for {high_risk_count} high-risk borrowers")
        
        total_exposure = sum([i.financial_impact or 0 for i in insights])
        recommendations.append(f"💰 Total exposure at risk: £{total_exposure:,.2f}")
        
        recommendations.extend([
            "📈 Implement dynamic credit scoring",
            "🔄 Review payment terms for high-risk accounts",
            "📱 Offer payment plan restructuring options"
        ])
        
        return recommendations

class UseCaseOrchestrator:
    """Orchestrates multiple use case modules and provides unified results"""
    
    def __init__(self):
        self.modules = {
            UseCaseType.FRAUD_DETECTION: FraudDetectionModule(),
            UseCaseType.CUSTOMER_CHURN: CustomerChurnModule(),
            UseCaseType.PAYMENT_DEFAULTS: PaymentDefaultsModule()
        }
    
    def run_use_case(self, use_case_type: UseCaseType, graph_data: Dict[str, Any], 
                     domain_context: str = "general") -> UseCaseResult:
        """Run a specific use case analysis"""
        
        if use_case_type not in self.modules:
            raise ValueError(f"Use case {use_case_type} not implemented")
        
        module = self.modules[use_case_type]
        start_time = datetime.now()
        
        try:
            result = module.analyze(graph_data, domain_context)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result
        except Exception as e:
            logging.error(f"Error running use case {use_case_type}: {e}")
            raise
    
    def run_all_use_cases(self, graph_data: Dict[str, Any], 
                         domain_context: str = "general") -> Dict[str, UseCaseResult]:
        """Run all available use cases and return comprehensive results"""
        
        results = {}
        
        for use_case_type, module in self.modules.items():
            try:
                result = self.run_use_case(use_case_type, graph_data, domain_context)
                results[use_case_type.value] = result
            except Exception as e:
                logging.error(f"Failed to run {use_case_type}: {e}")
                # Create error result
                results[use_case_type.value] = UseCaseResult(
                    use_case=use_case_type.value,
                    title=f"❌ {module.name} - Error",
                    description=f"Analysis failed: {str(e)}",
                    insights=[],
                    summary_metrics={},
                    recommended_actions=["🔧 Check data quality and try again"]
                )
        
        return results
    
    def get_business_summary(self, all_results: Dict[str, UseCaseResult]) -> Dict[str, Any]:
        """Generate executive summary across all use cases"""
        
        total_insights = sum(len(result.insights) for result in all_results.values())
        total_financial_impact = sum(
            sum(insight.financial_impact or 0 for insight in result.insights)
            for result in all_results.values()
        )
        
        high_priority_insights = []
        for result in all_results.values():
            high_priority_insights.extend([
                insight for insight in result.insights 
                if insight.risk_level == "HIGH" and insight.urgency == "HIGH"
            ])
        
        # Calculate potential ROI
        total_roi_estimate = sum(result.roi_estimate or 0 for result in all_results.values())
        
        return {
            "executive_summary": {
                "total_insights_generated": total_insights,
                "high_priority_alerts": len(high_priority_insights),
                "total_financial_exposure": total_financial_impact,
                "estimated_roi": total_roi_estimate,
                "top_recommendations": self._get_top_recommendations(all_results),
                "departments_affected": list(set(
                    insight.department for result in all_results.values() 
                    for insight in result.insights
                ))
            },
            "priority_actions": [
                {
                    "action": insight.recommended_actions[0] if insight.recommended_actions else "Review case",
                    "entity": insight.entity_name,
                    "impact": insight.financial_impact,
                    "department": insight.department
                }
                for insight in high_priority_insights[:5]  # Top 5 priority actions
            ]
        }
    
    def _get_top_recommendations(self, all_results: Dict[str, UseCaseResult]) -> List[str]:
        """Extract top recommendations across all use cases"""
        all_recommendations = []
        
        for result in all_results.values():
            all_recommendations.extend(result.recommended_actions[:2])  # Top 2 from each
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # Top 5 overall

# Industry-specific configurations
INDUSTRY_CONFIGS = {
    "ecommerce": {
        "fraud_detection": {"enabled": True, "priority": "HIGH"},
        "customer_churn": {"enabled": True, "priority": "MEDIUM"},
        "payment_defaults": {"enabled": True, "priority": "MEDIUM"}
    },
    "saas": {
        "customer_churn": {"enabled": True, "priority": "HIGH"},
        "fraud_detection": {"enabled": False, "priority": "LOW"},
        "payment_defaults": {"enabled": True, "priority": "MEDIUM"}
    },
    "fintech": {
        "fraud_detection": {"enabled": True, "priority": "HIGH"},
        "payment_defaults": {"enabled": True, "priority": "HIGH"},
        "customer_churn": {"enabled": True, "priority": "MEDIUM"}
    },
    "retail": {
        "fraud_detection": {"enabled": True, "priority": "HIGH"},
        "customer_churn": {"enabled": True, "priority": "HIGH"},
        "payment_defaults": {"enabled": False, "priority": "LOW"}
    }
}

def get_recommended_use_cases(industry: str) -> List[UseCaseType]:
    """Get recommended use cases for a specific industry"""
    config = INDUSTRY_CONFIGS.get(industry.lower(), INDUSTRY_CONFIGS["ecommerce"])
    
    recommended = []
    for use_case_name, settings in config.items():
        if settings["enabled"] and settings["priority"] in ["HIGH", "MEDIUM"]:
            try:
                use_case_type = UseCaseType(use_case_name)
                recommended.append(use_case_type)
            except ValueError:
                continue
    
    return recommended