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


New Advanced Learning Tasks (Based on Your Recent Code)
I can see you've added lightweight transformers, sklearn baselines, cross-validation, and ANN demos. Here are 10 fresh tasks focusing on these new components:

New Theoretical Tasks
26. Lightweight vs Heavy Preprocessing Comparison
Goal: Understand the trade-offs between implementations
Task: Compare 
transformers.py
 vs 
lightweight_transformers.py
:
Why does lightweight version avoid TensorFlow/Keras?
What's the difference between StringLookup (TF) and OneHotEncoder (sklearn)?
When would you use lightweight vs full preprocessing?
Performance implications: Import time, memory usage, execution speed
Document 3 key differences with code examples
27. Cross-Validation Strategy Analysis
Goal: Understand fold-safe CV for ML pipelines
Task: Study 
demo_cv_sklearn_quick.py
 and answer:
Why is the pipeline re-fitted inside each fold (line 42-43)?
What is "Out-of-Fold" (OOF) prediction and why is it important?
Why use stratify=df[target_col] in train_test_split?
What would happen if you fit the pipeline once on full data before CV?
Calculate: With 3 folds, what % of data is used for training in each fold?
28. Baseline Model Philosophy
Goal: Understand why simple baselines matter
Task: Analyze 
demo_sklearn_baseline.py
:
Why start with LogisticRegression before complex ANNs?
What does max_iter=200 mean and why might it fail?
Why is proba > 0.5 threshold used for binary classification?
If ANN gets AUC=0.86 and LogReg gets AUC=0.84, is ANN worth the complexity?
List 3 advantages of sklearn baselines over neural networks
29. Embedding vs One-Hot Trade-offs
Goal: Deep dive into categorical encoding strategies
Task: From 
lightweight_transformers.py
 (lines 237-240):
Why does lightweight version treat embed same as onehot?
Calculate memory: One-hot for 500 categories vs embedding dim 50
When does embedding become necessary? (hint: check customer_id)
What's the curse of dimensionality with one-hot encoding?
Design a rule: "Use embedding when cardinality > X"
New Practical Tasks
30. Run and Compare Baselines
Goal: Execute and benchmark different approaches
Task:
bash
python demo_sklearn_baseline.py
python demo_cv_sklearn_quick.py
Record AUC scores from both runs
Check artifacts/cv_sklearn/quick_run/oof_predictions.csv
Compare single holdout vs 3-fold CV results
Plot OOF predictions distribution
Which approach gives more reliable performance estimates?
31. Lightweight Pipeline Deep Dive
Goal: Understand the fast preprocessing implementation
Task:
bash
python demo_lightweight.py
Time the execution (add import time and measure)
Compare output shapes with full pipeline
Modify to use 5 features instead of all features
Add print statements to see transformation of one categorical feature
Document the flow: raw data → imputation → encoding → concatenation
32. Feature Encoding Experiment
Goal: Observe impact of encoding strategies
Task: Modify 
configs/feature_list.csv
:
Change gender from onehot to scale (intentionally wrong)
Run 
demo_sklearn_baseline.py
 and observe errors/warnings
Change customer_id from embed to onehot
Observe output dimension explosion
Document: Why does encoding strategy matter for model performance?
33. Cross-Validation Artifacts Analysis
Goal: Understand CV output structure
Task: After running 
demo_cv_sklearn_quick.py
:
bash
ls -la artifacts/cv_sklearn/quick_run/
Examine fold_0/oof_fold.csv structure
Load oof_predictions.csv and verify all rows are predicted
Calculate per-fold accuracy manually from fold CSVs
Compare with mean accuracy printed by script
Why save OOF predictions instead of just metrics?
34. Model Architecture Exploration
Goal: Understand ANN model structure
Task: Study 
demo_ann_simple.py
:
Run the demo: python demo_ann_simple.py
Identify what hidden_layers=[64, 32] means
What is dropout_rate=0.3 and why use it?
What is l2_reg=0.001 for?
Modify to hidden_layers=[128, 64, 32] and observe model summary
35. Pipeline State Debugging
Goal: Understand transformer state management
Task: Create a test script:
python
from src.preprocessing.lightweight_transformers import LightweightNumericTransformer
import pandas as pd

# Test unfitted transformer
transformer = LightweightNumericTransformer()
try:
    transformer.transform(pd.Series([1, 2, 3]))
except ValueError as e:
    print(f"Expected error: {e}")

# Test fitted transformer
transformer.fit(pd.Series([1, 2, 3, 4, 5]))
result = transformer.transform(pd.Series([6, 7, 8]))
print(f"Transformed: {result}")
Understand why is_fitted flag is critical
What happens if you fit twice?
How would you serialize this for production?
Integration & System Design Tasks
36. End-to-End Baseline Pipeline
Goal: Build complete baseline workflow
Task: Create my_baseline_experiment.py:
Load data with 
DataLoader
Split into train/val/test (60/20/20)
Fit lightweight pipeline on train only
Train LogisticRegression
Evaluate on val and test sets separately
Save model, predictions, and metrics to artifacts/my_experiment/
Document: Why separate val and test sets?
37. Performance Profiling
Goal: Identify bottlenecks in the pipeline
Task: Add timing to 
demo_sklearn_baseline.py
:
python
import time

t0 = time.time()
# ... data loading ...
print(f"Data loading: {time.time()-t0:.2f}s")

t0 = time.time()
# ... preprocessing ...
print(f"Preprocessing: {time.time()-t0:.2f}s")
Measure: loading, preprocessing, training, inference
Which step takes longest?
How would you optimize the slowest step?
Compare lightweight vs full preprocessing speed
38. Hyperparameter Sensitivity Analysis
Goal: Understand model sensitivity to parameters
Task: Modify 
demo_sklearn_baseline.py
:
Test LogisticRegression with different C values: [0.01, 0.1, 1.0, 10.0]
Test different max_iter: [50, 100, 200, 500]
Record AUC for each combination
Plot: C vs AUC, max_iter vs AUC
Which parameter has more impact on performance?
Bonus Advanced Challenges
39. Build Your Own Transformer
Goal: Implement custom preprocessing logic
Task: Create LightweightLogTransformer in 
lightweight_transformers.py
:
python
class LightweightLogTransformer:
    """Log transform for skewed numerical features."""
    def fit(self, X: pd.Series):
        # Store min value for handling zeros
        pass
    
    def transform(self, X: pd.Series):
        # Apply log(X + 1) transformation
        pass
Apply to monthly_revenue feature
Compare distributions before/after
Integrate into 
LightweightPreprocessingPipeline
40. Ensemble Baseline
Goal: Combine multiple simple models
Task: Extend 
demo_sklearn_baseline.py
:
Train 3 models: LogisticRegression, RandomForest, GradientBoosting
Average their predictions
Compare ensemble AUC vs individual model AUCs
Does simple averaging beat the best single model?
Research: What is "stacking" and how is it different?
Task Progression Map
Task #	Type	Difficulty	New Component Focus
26	Theoretical	Medium	Lightweight Preprocessing
27	Theoretical	Hard	Cross-Validation
28	Theoretical	Easy	Baseline Models
29	Theoretical	Medium	Categorical Encoding
30	Practical	Easy	Running Baselines
31	Practical	Easy	Lightweight Pipeline
32	Practical	Medium	Encoding Experiments
33	Practical	Medium	CV Artifacts
34	Practical	Medium	ANN Architecture
35	Practical	Hard	State Management
36	Integration	Hard	Complete Workflow
37	Integration	Medium	Performance Profiling
38	Integration	Medium	Hyperparameter Tuning
39	Advanced	Hard	Custom Transformers
40	Advanced	Hard	Ensemble Methods
Recommended Learning Path
Week 1 - Foundations:

Tasks 26, 28, 30, 31 (Understand lightweight preprocessing & baselines)
Week 2 - Cross-Validation:

Tasks 27, 30, 33 (Master fold-safe CV and OOF predictions)
Week 3 - Experimentation:

Tasks 32, 34, 37, 38 (Hands-on experiments with encoding & hyperparameters)
Week 4 - Advanced:

Tasks 29, 35, 36, 39, 40 (Build custom components & complete pipelines)
Key Insights to Discover
Why lightweight matters: 10x faster iteration during development
Fold-safe preprocessing: The #1 cause of data leakage in ML competitions
Baseline first: Always establish simple baseline before complex models
OOF predictions: Essential for reliable model evaluation
Encoding strategy: Can make 10x difference in model size and performance
These tasks will help you understand the practical engineering aspects of building production ML systems, not just the theory! 🚀