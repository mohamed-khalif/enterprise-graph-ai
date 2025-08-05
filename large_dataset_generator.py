import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_business_dataset(n_records=100000):
    """
    Generate a realistic large business dataset for testing your AI platform
    Creates interconnected sales opportunities, accounts, and reps
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate accounts first
    n_accounts = n_records // 10  # 10 opportunities per account on average
    
    industries = ['Technology', 'Finance', 'Healthcare', 'Manufacturing', 'Retail', 'Energy']
    company_suffixes = ['Corp', 'Inc', 'LLC', 'Ltd', 'Group', 'Solutions', 'Systems', 'Technologies']
    
    accounts_data = []
    for i in range(n_accounts):
        accounts_data.append({
            'account_id': f'ACC-{i+1:06d}',
            'company_name': f'Company-{i+1} {random.choice(company_suffixes)}',
            'industry': random.choice(industries),
            'annual_revenue': np.random.lognormal(15, 1.5),  # Realistic revenue distribution
            'health_score': np.random.normal(7.5, 1.5),
            'employee_count': int(np.random.lognormal(6, 1.2)),
            'created_at': datetime.now() - timedelta(days=random.randint(1, 1000))
        })
    
    accounts_df = pd.DataFrame(accounts_data)
    
    # Generate sales reps
    n_reps = max(50, n_records // 2000)  # Realistic rep-to-opportunity ratio
    
    rep_names = [f'Rep-{i+1}' for i in range(n_reps)]
    territories = ['North', 'South', 'East', 'West', 'Central', 'International']
    
    reps_data = []
    for i in range(n_reps):
        reps_data.append({
            'rep_id': f'REP-{i+1:03d}',
            'rep_name': rep_names[i],
            'territory': random.choice(territories),
            'experience_years': random.randint(1, 15),
            'quota': np.random.normal(1000000, 300000),
            'hire_date': datetime.now() - timedelta(days=random.randint(30, 2000))
        })
    
    reps_df = pd.DataFrame(reps_data)
    
    # Generate opportunities with realistic relationships
    stages = ['Discovery', 'Qualified', 'Proposal', 'Negotiation', 'Closed-Won', 'Closed-Lost']
    sources = ['Inbound Lead', 'Cold Outreach', 'Referral', 'Trade Show', 'Partner', 'Website']
    
    opportunities_data = []
    for i in range(n_records):
        # Create realistic relationships
        account = accounts_df.iloc[i % len(accounts_df)]
        rep = reps_df.iloc[i % len(reps_df)]
        
        # Realistic deal size based on company size and industry
        base_deal_size = account['annual_revenue'] * 0.001  # 0.1% of annual revenue
        deal_size = max(1000, int(np.random.lognormal(np.log(base_deal_size), 0.8)))
        
        # Stage progression affects other fields
        stage = random.choice(stages)
        days_in_stage = random.randint(1, 180) if stage != 'Closed-Won' else 0
        
        # Engagement score correlates with stage
        if stage in ['Closed-Won', 'Negotiation']:
            engagement_score = np.random.normal(8.5, 1.0)
        elif stage in ['Proposal', 'Qualified']:
            engagement_score = np.random.normal(7.0, 1.5)
        else:
            engagement_score = np.random.normal(5.5, 2.0)
        
        engagement_score = max(1, min(10, engagement_score))
        
        opportunities_data.append({
            'opportunity_id': f'OPP-{i+1:06d}',
            'account_id': account['account_id'],
            'rep_id': rep['rep_id'],
            'deal_size': deal_size,
            'stage': stage,
            'source': random.choice(sources),
            'industry': account['industry'],
            'contact_engagement_score': round(engagement_score, 1),
            'days_in_stage': days_in_stage,
            'demo_completed': random.choice(['TRUE', 'FALSE']),
            'proposal_sent': random.choice(['TRUE', 'FALSE']),
            'close_date': datetime.now() + timedelta(days=random.randint(-30, 180)),
            'created_at': datetime.now() - timedelta(days=random.randint(1, 365))
        })
    
    opportunities_df = pd.DataFrame(opportunities_data)
    
    return opportunities_df, accounts_df, reps_df

def save_datasets(opportunities_df, accounts_df, reps_df, size_label):
    """Save the generated datasets to CSV files"""
    
    opportunities_df.to_csv(f'opportunities_{size_label}.csv', index=False)
    accounts_df.to_csv(f'accounts_{size_label}.csv', index=False)
    reps_df.to_csv(f'reps_{size_label}.csv', index=False)
    
    print(f"✅ Generated {size_label} datasets:")
    print(f"   📊 Opportunities: {len(opportunities_df):,} records")
    print(f"   🏢 Accounts: {len(accounts_df):,} records") 
    print(f"   👥 Reps: {len(reps_df):,} records")
    print(f"   📁 Files: opportunities_{size_label}.csv, accounts_{size_label}.csv, reps_{size_label}.csv")

if __name__ == "__main__":
    # Generate different sized datasets for testing
    
    print("🚀 Generating test datasets for cloud AI platform...\n")
    
    # Medium dataset (10K opportunities)
    print("1️⃣ Generating medium dataset (10K records)...")
    opp_medium, acc_medium, rep_medium = generate_large_business_dataset(10000)
    save_datasets(opp_medium, acc_medium, rep_medium, "medium_10k")
    
    print("\n" + "="*60 + "\n")
    
    # Large dataset (100K opportunities)  
    print("2️⃣ Generating large dataset (100K records)...")
    opp_large, acc_large, rep_large = generate_large_business_dataset(100000)
    save_datasets(opp_large, acc_large, rep_large, "large_100k")
    
    print("\n" + "="*60 + "\n")
    
    # Enterprise dataset (500K opportunities)
    print("3️⃣ Generating enterprise dataset (500K records)...")
    opp_enterprise, acc_enterprise, rep_enterprise = generate_large_business_dataset(500000)
    save_datasets(opp_enterprise, acc_enterprise, rep_enterprise, "enterprise_500k")
    
    print("\n🎉 All datasets generated successfully!")
    print("\nRecommended testing order:")
    print("1. Test with medium_10k (~$0.05 cost)")
    print("2. Test with large_100k (~$0.50 cost)")  
    print("3. Test with enterprise_500k (~$2.50 cost)")
    print("\nReady to test your cloud AI platform! 🚀")