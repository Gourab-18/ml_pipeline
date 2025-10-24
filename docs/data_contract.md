# Data Contract: Customer Churn Prediction

## Overview
This document defines the data contract for the customer churn prediction system, including train/test split rules, label latency, and data quality requirements.

## Data Sources and Access

### Primary Data Sources
1. **Customer Database** (`customer_db`)
   - **Access**: Direct database connection
   - **Update Frequency**: Real-time
   - **Retention**: 5 years historical data
   - **Key Tables**: customers, subscriptions, demographics

2. **Usage Analytics** (`analytics_db`)
   - **Access**: Data warehouse via API
   - **Update Frequency**: Daily batch (T+1)
   - **Retention**: 2 years historical data
   - **Key Tables**: user_sessions, feature_usage, page_views

3. **Support System** (`support_db`)
   - **Access**: REST API
   - **Update Frequency**: Real-time
   - **Retention**: 3 years historical data
   - **Key Tables**: tickets, interactions, resolutions

4. **Billing System** (`billing_db`)
   - **Access**: Secure database connection
   - **Update Frequency**: Real-time
   - **Retention**: 7 years historical data
   - **Key Tables**: payments, invoices, billing_events

5. **Marketing Platform** (`marketing_db`)
   - **Access**: API with rate limiting
   - **Update Frequency**: Daily batch
   - **Retention**: 1 year historical data
   - **Key Tables**: campaigns, email_events, responses

## Train/Test Split Rules

### Temporal Split Strategy
- **Training Period**: 2023-01-01 to 2023-10-31 (10 months)
- **Validation Period**: 2023-11-01 to 2023-11-30 (1 month)
- **Test Period**: 2023-12-01 to 2023-12-31 (1 month)
- **Prediction Period**: 2024-01-01 onwards

### Split Rationale
1. **Temporal Ordering**: Maintains chronological order to prevent data leakage
2. **Seasonality**: Captures seasonal patterns in customer behavior
3. **Business Cycles**: Aligns with quarterly business cycles
4. **Sufficient Data**: Ensures adequate sample sizes for all splits

### Data Volume Targets
- **Training Set**: ~400K customers
- **Validation Set**: ~50K customers  
- **Test Set**: ~50K customers
- **Minimum Churn Rate**: 10% in each split

## Label Latency and Availability

### Label Definition
- **Churn Event**: Customer cancels subscription
- **Label Window**: 30 days from prediction date
- **Label Latency**: 30 days (labels available T+30)

### Label Generation Process
1. **Daily Process**: Run at 2 AM UTC
2. **Data Cutoff**: Features from T-1, labels from T-30
3. **Label Validation**: Automated quality checks
4. **Label Distribution**: Available via API within 1 hour

### Label Quality Assurance
- **Automated Checks**: 
  - Churn rate within expected range (8-20%)
  - No duplicate labels
  - Temporal consistency checks
- **Manual Review**: Weekly sample validation
- **Error Handling**: Failed labels logged and retried

## Data Quality Requirements

### Completeness Requirements
- **Minimum Data Coverage**: 95% of features non-null
- **Maximum Missing Rate**: 30% for any single feature
- **Customer Coverage**: 98% of active customers included

### Accuracy Requirements
- **Data Validation**: Automated schema validation
- **Outlier Detection**: Statistical outlier identification
- **Consistency Checks**: Cross-source data validation
- **Freshness**: Data no older than 24 hours

### Privacy and Security
- **PII Handling**: Customer IDs hashed, no direct PII
- **Data Encryption**: All data encrypted in transit and at rest
- **Access Controls**: Role-based access with audit logging
- **Compliance**: GDPR and CCPA compliant

## Feature Engineering Rules

### Temporal Features
- **Lookback Windows**: 7, 30, 90 days for different features
- **Aggregation Methods**: Mean, max, min, count, trend
- **Time Zones**: All timestamps converted to UTC
- **Holiday Handling**: Business day calculations

### Categorical Features
- **Encoding**: One-hot encoding for low cardinality, target encoding for high cardinality
- **Unknown Handling**: "Unknown" category for missing values
- **Cardinality Limits**: Max 100 unique values per categorical feature

### Numerical Features
- **Scaling**: StandardScaler for neural networks
- **Outlier Treatment**: Winsorization at 1st and 99th percentiles
- **Missing Imputation**: Median for numerical, mode for categorical

## Data Pipeline Requirements

### Ingestion Pipeline
- **Frequency**: Daily batch processing
- **Processing Time**: Complete within 4 hours
- **Error Handling**: Failed records logged and retried
- **Monitoring**: Pipeline health checks and alerts

### Feature Store
- **Storage**: Time-series feature store
- **Retention**: 2 years of feature history
- **Access**: Low-latency API for model serving
- **Versioning**: Feature versioning and lineage tracking

### Data Lineage
- **Tracking**: Full data lineage from source to model
- **Documentation**: Automated documentation generation
- **Change Management**: Schema change notifications
- **Rollback**: Ability to rollback to previous versions

## Monitoring and Alerting

### Data Quality Monitoring
- **Missing Data**: Alert if >5% missing for any feature
- **Distribution Drift**: Statistical tests for feature drift
- **Volume Anomalies**: Alert if data volume changes >20%
- **Latency Monitoring**: Alert if data is >24 hours old

### Model Performance Monitoring
- **Prediction Drift**: Monitor prediction distribution changes
- **Feature Importance**: Track feature importance stability
- **Business Metrics**: Monitor actual vs predicted churn rates
- **Model Decay**: Automated retraining triggers

## Compliance and Governance

### Data Governance
- **Data Catalog**: Comprehensive data dictionary
- **Lineage Tracking**: Full data lineage documentation
- **Access Logging**: All data access logged and audited
- **Retention Policies**: Automated data retention enforcement

### Regulatory Compliance
- **GDPR**: Right to be forgotten implementation
- **CCPA**: Data subject access request handling
- **SOX**: Financial data audit requirements
- **Industry Standards**: SOC 2 Type II compliance

## Change Management

### Schema Changes
- **Versioning**: Semantic versioning for schema changes
- **Backward Compatibility**: 6-month backward compatibility
- **Migration**: Automated migration scripts
- **Testing**: Comprehensive testing before deployment

### Data Source Changes
- **Notification**: 30-day advance notice for changes
- **Impact Assessment**: Business impact analysis
- **Mitigation**: Backup plans for source failures
- **Documentation**: Updated documentation and training

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Next Review**: 2024-04-15  
**Approved By**: Data Engineering Lead, ML Engineering Lead, Compliance Officer
