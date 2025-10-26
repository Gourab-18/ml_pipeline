Theoretical Tasks
1. Schema-Driven Development Analysis
Goal: Understand the data contract pattern
Task: Read 
/configs/schema.yaml
 and document:
What are the 5 leaky features and why are they problematic?
What's the difference between impute_median, impute_mode, and impute_zero policies?
How does the schema prevent data leakage in production?
2. CI/CD Pipeline Understanding
Goal: Learn the automated quality gates
Task: Analyze 
.github/workflows/ci.yml
 and answer:
What are the 4 code quality checks that run before tests?
Why does the pipeline test on Python 3.9, 3.10, and 3.11?
What happens if test coverage drops?
3. Data Quality Constraints
Goal: Understand data validation rules
Task: From 
schema.yaml
, explain:
Why is max_missing_percentage set to 30%?
What would happen if two features have 0.96 correlation?
Which features have missing_policy: "forbidden" and why?
4. Feature Engineering Strategy
Goal: Learn about derived features
Task: Review the feature_engineering section in schema and:
Explain what "interaction features" are and give 2 examples
Why would login_frequency * session_duration_avg be useful?
What's the difference between temporal and aggregation features?
Practical Tasks
5. Run the Test Suite
Goal: Understand existing test coverage
Task:
bash
make test
Document which tests pass/fail
Check the coverage report in htmlcov/index.html
Identify which modules have <80% coverage
6. Validate Sample Data
Goal: Learn the data loader validation
Task:
bash
python src/data/loader.py
Observe what validation checks run
Identify which features have missing values
Check if any leaky features are flagged
7. Code Quality Check
Goal: Understand code standards
Task:
bash
make lint
Note any linting errors or warnings
Run make format to see what changes
Understand why mypy type checking matters
8. Explore the EDA Notebook
Goal: Understand data characteristics
Task: Open 
notebooks/01_eda.ipynb
 and:
Run all cells to see data distributions
Identify which features have the most missing values
Note any correlations between features and churn
9. Trace Data Flow
Goal: Understand the pipeline architecture
Task: Map out the data flow by reading:
src/data/loader.py
 → How data enters the system
src/preprocessing/ → What transformations happen
src/models/ → How models consume the data
Document the flow in a diagram or text
10. Add a New Feature to Schema
Goal: Practice schema modification
Task: Add a new feature to 
configs/schema.yaml
:
yaml
last_login_days:
  type: "integer"
  description: "Days since last login"
  missing_policy: "impute_median"
  leaky: false
Run the data loader to see validation
Check if tests still pass
Understand the impact of schema changes
Bonus Challenge Tasks
11. Dependency Analysis
Review 
requirements.txt
 and identify:
Which libraries are for ML (model building)?
Which are for MLOps (monitoring, serving)?
Why are both PyTorch and TensorFlow included?
12. Makefile Workflow
Run make help and test each command
Understand the purpose of make clean
Identify what's missing in make train-sample


Advanced Theoretical Tasks
13. Fold-Safe Preprocessing Deep Dive
Goal: Understand data leakage prevention in cross-validation
Task: Study 
src/preprocessing/pipeline.py
 and answer:
Why does 
fit_on()
 only use training data?
What would happen if you fit transformers on the entire dataset?
How does the pipeline prevent target leakage during feature engineering?
Trace the flow: training data → fit → validation data → transform
14. Embedding Dimension Strategy
Goal: Learn categorical feature handling for neural networks
Task: Analyze 
EmbeddingMapper
 in 
transformers.py
:
Why is the formula embedding_dim = min(50, (cardinality + 1) // 2)?
When should you use onehot vs embed for categorical features?
Calculate embedding dimensions for: 10, 100, 1000, 10000 unique categories
Why is customer_id embedded instead of one-hot encoded?
15. Data Contract Analysis
Goal: Understand production ML data requirements
Task: Read 
docs/data_contract.md
 and document:
Why is temporal split used instead of random split?
What is "label latency" and why is it 30 days?
How does the pipeline handle the T+30 delay in getting labels?
What are the implications of daily batch predictions vs real-time?
16. Business Metrics vs ML Metrics
Goal: Connect technical metrics to business value
Task: From 
docs/problem_spec.md
, analyze:
Why is AUC-ROC chosen over accuracy?
What does "Precision@10%" mean for the business?
Calculate ROI: If model saves $2M annually but costs $50K, what's the return?
Why is inference latency <100ms important for batch predictions?
Advanced Practical Tasks
17. Build the Preprocessing Pipeline
Goal: Understand end-to-end data transformation
Task:
bash
python src/preprocessing/pipeline.py
Observe which features get scaled vs one-hot vs embedded
Check the output shape after transformation
Modify 
configs/feature_list.csv
 to change age from scale to onehot
Re-run and observe the impact on output dimensions
18. Test-Driven Development Exploration
Goal: Learn testing patterns for ML pipelines
Task: Read 
tests/test_loader.py
 (lines 1-100) and:
Identify the 3 types of tests (initialization, validation, error handling)
Run a single test: pytest tests/test_loader.py::TestDataLoader::test_load_valid_data -v
Add a new test for checking leaky features are properly flagged
Understand why 
setup_method()
 and 
teardown_method()
 are used
19. Feature Cardinality Analysis
Goal: Understand the relationship between cardinality and encoding strategy
Task: Analyze 
configs/feature_list.csv
:
Plot cardinality vs action (scale/onehot/embed/drop)
What's the threshold between onehot and embed? (Hint: look at cardinality values)
Why are features with cardinality <10 one-hot encoded?
Why is customer_id (cardinality 500) embedded instead of dropped?
20. Transformer State Management
Goal: Understand stateful transformers and serialization
Task: In 
src/preprocessing/transformers.py
:
Identify which transformers store state (hint: is_fitted flag)
What happens if you call 
transform()
 before 
fit()
?
How would you save/load a fitted transformer for production?
Write pseudocode for serializing the entire preprocessing pipeline
21. Data Quality Validation
Goal: Implement custom validation rules
Task: Extend 
DataLoader.validate_data_quality()
:
Add a check for duplicate customer_id values
Add a check for negative values in age or monthly_revenue
Add a check for future dates in temporal features
Run your enhanced validation on sample data
22. Missing Value Strategy Comparison
Goal: Understand different imputation strategies
Task: Create a notebook to compare:
Load sample data with missing values
Apply impute_median, impute_mode, impute_zero to same feature
Compare the distributions before/after imputation
Document when each strategy is appropriate
Why does subscription_type have missing_policy: forbidden?
Integration & Architecture Tasks
23. End-to-End Pipeline Trace
Goal: Map the complete data flow
Task: Create a flowchart showing:
Raw data → 
DataLoader
 → Schema validation
Validated data → 
PreprocessingPipeline
 → Feature selection
Selected features → Transformers → Encoded features
Encoded features → Model (not implemented yet)
Model output → Business action
Identify where data leakage could occur at each step
24. CI/CD Pipeline Enhancement
Goal: Improve automated testing
Task: Modify 
.github/workflows/ci.yml
 to:
Add a step to generate sample data before tests
Add a step to check if 
configs/schema.yaml
 is valid YAML
Add a step to verify all features in schema exist in 
feature_list.csv
Add a step to check preprocessing pipeline can run without errors
Test your changes locally with make test
Bonus Challenge: Code Quality & Design Patterns
25. Design Pattern Identification
Goal: Recognize ML engineering patterns
Task: Identify these patterns in the codebase:
Transformer Pattern: Where is fit/transform used? (sklearn-style)
Factory Pattern: Where are objects created? (e.g., 
create_preprocessing_pipeline()
)
Strategy Pattern: Where are different algorithms swapped? (e.g., imputation strategies)
Singleton Pattern: Is the schema loaded once or multiple times?
Document 3 design patterns you find with code examples