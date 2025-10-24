"""
Sample data generator for customer churn prediction.

Generates synthetic customer data that follows the schema defined in configs/schema.yaml.
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any
import os


class SampleDataGenerator:
    """Generates synthetic customer churn data."""
    
    def __init__(self, schema_path: str = "configs/schema.yaml"):
        """Initialize with schema configuration."""
        self.schema_path = schema_path
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema configuration."""
        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_customers(self, n_customers: int = 500, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic customer data."""
        np.random.seed(seed)
        
        # Generate customer IDs
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        
        # Generate demographics
        ages = np.random.normal(35, 12, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)  # Reasonable age range
        
        genders = np.random.choice(['M', 'F', 'Other'], n_customers, p=[0.45, 0.45, 0.1])
        countries = np.random.choice(['US', 'CA', 'UK', 'DE', 'FR', 'AU'], n_customers, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
        
        # Generate subscription data
        subscription_types = np.random.choice(['Basic', 'Premium', 'Enterprise'], n_customers, p=[0.6, 0.3, 0.1])
        subscription_durations = np.random.exponential(180, n_customers).astype(int)  # Days since subscription
        subscription_durations = np.clip(subscription_durations, 1, 1095)  # 1 day to 3 years
        
        # Monthly revenue based on subscription type
        monthly_revenues = np.where(
            subscription_types == 'Basic', 
            np.random.normal(29, 5, n_customers),
            np.where(
                subscription_types == 'Premium',
                np.random.normal(79, 10, n_customers),
                np.random.normal(199, 25, n_customers)
            )
        )
        monthly_revenues = np.clip(monthly_revenues, 0, None)
        
        # Generate usage analytics (last 30 days)
        # Higher usage for longer subscribers
        usage_multiplier = np.clip(subscription_durations / 180, 0.5, 2.0)
        
        login_frequencies = np.random.poisson(15 * usage_multiplier, n_customers)
        feature_usage_counts = np.random.poisson(8 * usage_multiplier, n_customers)
        session_durations = np.random.exponential(25 * usage_multiplier, n_customers)
        page_views = np.random.poisson(50 * usage_multiplier, n_customers)
        
        # Generate support data
        support_tickets = np.random.poisson(0.5, n_customers)  # Most customers have 0-1 tickets
        support_resolution_days = np.where(
            support_tickets > 0,
            np.random.exponential(3, n_customers),
            0
        )
        
        # Generate payment data
        payment_failures = np.random.poisson(0.2, n_customers)  # Most customers have 0 failures
        days_since_payment = np.random.exponential(15, n_customers).astype(int)
        days_since_payment = np.clip(days_since_payment, 1, 90)
        
        # Generate marketing engagement
        email_opens = np.random.poisson(3, n_customers)
        email_clicks = np.random.poisson(1, n_customers)
        
        # Generate churn labels (target variable)
        # Churn probability based on multiple factors
        churn_base_prob = 0.15  # Base churn rate
        
        # Risk factors
        high_support_risk = (support_tickets > 2) * 0.3
        payment_risk = (payment_failures > 1) * 0.4
        low_usage_risk = (login_frequencies < 5) * 0.25
        new_customer_risk = (subscription_durations < 30) * 0.2
        
        churn_probabilities = np.clip(
            churn_base_prob + high_support_risk + payment_risk + low_usage_risk + new_customer_risk,
            0.0, 1.0
        )
        
        churn_labels = np.random.binomial(1, churn_probabilities, n_customers)
        
        # Generate leaky features (only for churned customers)
        churn_dates = np.where(
            churn_labels == 1,
            pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 30, n_customers), unit='D'),
            pd.NaT
        )
        
        cancellation_reasons = np.where(
            churn_labels == 1,
            np.random.choice(['Price', 'Features', 'Support', 'Competitor', 'Other'], n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            None
        )
        
        final_billing_amounts = np.where(
            churn_labels == 1,
            monthly_revenues * np.random.uniform(0.8, 1.2, n_customers),
            np.nan
        )
        
        retention_offers_accepted = np.where(
            churn_labels == 1,
            np.random.choice([True, False], n_customers, p=[0.3, 0.7]),
            None
        )
        
        post_churn_support_contacts = np.where(
            churn_labels == 1,
            np.random.poisson(0.5, n_customers),
            0
        )
        
        # Create DataFrame
        data = {
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'location_country': countries,
            'subscription_type': subscription_types,
            'subscription_duration_days': subscription_durations,
            'monthly_revenue': monthly_revenues,
            'login_frequency': login_frequencies,
            'feature_usage_count': feature_usage_counts,
            'session_duration_avg': session_durations,
            'page_views_total': page_views,
            'support_tickets_count': support_tickets,
            'support_tickets_avg_resolution_days': support_resolution_days,
            'payment_failures_count': payment_failures,
            'days_since_last_payment': days_since_payment,
            'email_opens_count': email_opens,
            'email_clicks_count': email_clicks,
            'churn_probability': churn_labels.astype(float),
            # Leaky features
            'churn_date': churn_dates,
            'cancellation_reason': cancellation_reasons,
            'final_billing_amount': final_billing_amounts,
            'retention_offer_accepted': retention_offers_accepted,
            'post_churn_support_contacts': post_churn_support_contacts
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values to make it realistic
        missing_rates = {
            'age': 0.02,
            'gender': 0.01,
            'location_country': 0.01,
            'session_duration_avg': 0.05,
            'support_tickets_avg_resolution_days': 0.1
        }
        
        for col, rate in missing_rates.items():
            if col in df.columns:
                missing_mask = np.random.random(len(df)) < rate
                df.loc[missing_mask, col] = np.nan
        
        return df
    
    def save_sample_data(self, output_path: str = "data/sample.csv", n_customers: int = 500):
        """Generate and save sample data."""
        df = self.generate_customers(n_customers)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} customer records and saved to {output_path}")
        print(f"Churn rate: {df['churn_probability'].mean():.2%}")
        return df


def main():
    """Generate sample data."""
    generator = SampleDataGenerator()
    df = generator.save_sample_data()
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Total customers: {len(df)}")
    print(f"Churn rate: {df['churn_probability'].mean():.2%}")
    print(f"Missing values per column:")
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 0:
            print(f"  {col}: {missing_pct:.1f}%")


if __name__ == "__main__":
    main()
