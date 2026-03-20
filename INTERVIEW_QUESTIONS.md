# ML Pipeline Project - Comprehensive Interview Questions

## Project Overview
**Project**: Production-Ready ML Pipeline for Customer Churn Prediction  
**Tech Stack**: TensorFlow/Keras, scikit-learn, XGBoost, LightGBM, FastAPI, pytest  
**Complexity**: End-to-end ML lifecycle with preprocessing, training, evaluation, explainability, and serving

---

## 1. Problem Understanding & Business Context

### Basic Questions
1. **Can you walk me through the business problem this ML pipeline solves?**
   - Expected: Customer churn prediction for subscription service, 30-day prediction horizon
   - Follow-up: What's the business impact? (Cost of churn: $500 LTV, retention cost: $50)

2. **What are your success metrics and why did you choose them?**
   - Expected: Primary metric is ROC-AUC ≥ 0.85, secondary metrics include Precision@10%, Recall@10%
   - Follow-up: Why ROC-AUC over other metrics? (Balanced performance across thresholds)

3. **What's the prediction frequency and inference latency requirement?**
   - Expected: Daily batch predictions, <100ms per prediction
   - Follow-up: How does this affect your architecture choices?

### Advanced Questions
4. **How did you handle the temporal nature of this problem?**
   - Expected: Discuss data contract, label latency, temporal splits in CV
   - Follow-up: What features are leaky and how did you identify them?

5. **What are the key business constraints that influenced your technical decisions?**
   - Expected: GDPR compliance, model size <1GB, budget constraints
   - Follow-up: How did these constraints affect your model selection?

---

## 2. Data Engineering & Preprocessing

### Basic Questions
6. **How did you handle missing values in your dataset?**
   - Expected: Schema-driven approach with different strategies (impute_median, impute_zero, impute_mode)
   - Follow-up: Why different strategies for different features?

7. **What preprocessing steps did you implement?**
   - Expected: Numeric scaling, categorical encoding (one-hot/embeddings), imputation
   - Follow-up: Why did you create both lightweight and full pipelines?

8. **How do you ensure your preprocessing doesn't leak information?**
   - Expected: Fold-safe design - fit on train, transform on validation
   - Follow-up: Can you explain what would happen if you fit on the entire dataset?

### Advanced Questions
9. **Explain your "fold-safe preprocessing pipeline" design. Why is this critical?**
   - Expected: Prevents data leakage by ensuring normalizers/encoders only see training data
   - Follow-up: How did you implement this in cross-validation?

10. **You have both a lightweight and full preprocessing pipeline. When would you use each?**
    - Expected: Lightweight (sklearn-only) for fast iteration, full (TensorFlow) for production
    - Follow-up: What are the performance differences?

11. **How did you handle high-cardinality categorical features?**
    - Expected: Embedding layers with automatic dimension calculation (min(50, vocab_size//2))
    - Follow-up: Why embeddings instead of one-hot encoding?

12. **Walk me through your schema validation approach.**
    - Expected: YAML-based schema with type checking, missing value policies, leakage indicators
    - Follow-up: How does this help in production?

---

## 3. Model Architecture & Training

### Basic Questions
13. **What models did you implement and why?**
    - Expected: Tabular ANN with embeddings, XGBoost, LightGBM baselines
    - Follow-up: Why multiple models?

14. **Describe the architecture of your TabularANN model.**
    - Expected: Separate input layers for numeric/categorical, embedding layers, MLP trunk with dropout/L2
    - Follow-up: Why dual outputs (probabilities + logits)?

15. **What training callbacks did you implement?**
    - Expected: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    - Follow-up: What are the benefits of each?

### Advanced Questions
16. **Explain your cross-validation strategy. Why K-Fold instead of simple train/test split?**
    - Expected: More robust evaluation, OOF predictions, per-fold artifacts
    - Follow-up: How do you ensure reproducibility?

17. **You implemented nested CV with HPO. Can you explain this architecture?**
    - Expected: Outer loop for evaluation, inner loop for hyperparameter search
    - Follow-up: What's the computational cost? When would you skip nested CV?

18. **How did you handle class imbalance (if present)?**
    - Expected: Discuss metrics choice (ROC-AUC, PR-AUC), potential class weights
    - Follow-up: Why is PR-AUC important for imbalanced datasets?

19. **Why do you output both probabilities and logits from your model?**
    - Expected: Logits needed for Platt scaling calibration
    - Follow-up: What's the mathematical relationship between them?

20. **How do you prevent overfitting in your neural network?**
    - Expected: Dropout (0.3), L2 regularization (0.001), early stopping
    - Follow-up: How did you choose these hyperparameter values?

---

## 4. Model Evaluation & Calibration

### Basic Questions
21. **What evaluation metrics do you track?**
    - Expected: ROC-AUC, PR-AUC, Brier score, ECE, Accuracy, Precision, Recall, F1
    - Follow-up: What does each metric tell you?

22. **What is calibration and why is it important?**
    - Expected: Ensuring predicted probabilities match actual frequencies
    - Follow-up: Give an example where calibration matters.

23. **What calibration methods did you implement?**
    - Expected: Platt scaling (logistic regression on logits), Isotonic regression
    - Follow-up: When would you use one over the other?

### Advanced Questions
24. **Explain Expected Calibration Error (ECE). How do you compute it?**
    - Expected: Bin predictions, compare mean predicted prob to mean actual frequency
    - Follow-up: What's a good ECE value?

25. **How does Platt scaling work mathematically?**
    - Expected: Fits logistic regression on logits to calibrate probabilities
    - Follow-up: Why use logits instead of probabilities?

26. **You have automatic best calibrator selection. What metric do you optimize?**
    - Expected: Brier score by default, can use ECE
    - Follow-up: How do you compare calibration methods fairly?

27. **What's the difference between Brier score and log loss?**
    - Expected: Both measure probability accuracy, Brier is MSE, log loss is cross-entropy
    - Follow-up: When would one be preferred over the other?

---

## 5. Model Explainability

### Basic Questions
28. **What explainability methods did you implement?**
    - Expected: Permutation importance, SHAP values
    - Follow-up: Why both methods?

29. **How does permutation importance work?**
    - Expected: Shuffle feature values, measure performance drop
    - Follow-up: What are the limitations?

30. **What are SHAP values?**
    - Expected: Shapley values from game theory, shows feature contribution per prediction
    - Follow-up: When would you use SHAP over permutation importance?

### Advanced Questions
31. **You compute permutation importance on OOF predictions. Why?**
    - Expected: No model refitting needed, uses unbiased predictions
    - Follow-up: What's the computational advantage?

32. **How do you handle correlated features in permutation importance?**
    - Expected: Acknowledge the issue, discuss potential solutions (group permutation)
    - Follow-up: How does correlation affect interpretation?

33. **SHAP can be computationally expensive. How did you handle this?**
    - Expected: KernelExplainer for model-agnostic approach, sample subset for computation
    - Follow-up: What are faster alternatives for specific model types?

---

## 6. Model Serving & Production

### Basic Questions
34. **How do you serve your models in production?**
    - Expected: FastAPI REST API with lazy TensorFlow loading
    - Follow-up: Why FastAPI over Flask?

35. **What's the startup time of your server?**
    - Expected: ~0.5s without model, 5-10s with TensorFlow model first load
    - Follow-up: How did you achieve fast startup?

36. **What endpoints does your API expose?**
    - Expected: /predict, /load, /health, /model_info
    - Follow-up: Why is /health endpoint important?

### Advanced Questions
37. **Explain your "lazy TensorFlow loading" optimization.**
    - Expected: TensorFlow only imported when model is loaded, not at startup
    - Follow-up: What's the benefit? (Server starts in <1s)

38. **How do you handle batch predictions in your API?**
    - Expected: Accept list of feature dictionaries, process in single forward pass
    - Follow-up: What's the throughput improvement?

39. **You support both TensorFlow and sklearn models. How did you design this?**
    - Expected: Abstract model interface, auto-detection of model type
    - Follow-up: What are the tradeoffs between model types?

40. **How do you ensure calibrated probabilities are returned in production?**
    - Expected: Calibrator loaded with model, applied automatically in predict endpoint
    - Follow-up: What if calibrator is missing?

41. **What would you add for production monitoring?**
    - Expected: Logging, metrics (latency, throughput), model drift detection
    - Follow-up: How would you detect model drift?

---

## 7. Testing & Code Quality

### Basic Questions
42. **What testing strategy did you implement?**
    - Expected: Comprehensive pytest suite covering all components
    - Follow-up: What's your test coverage?

43. **What types of tests did you write?**
    - Expected: Unit tests, integration tests, API tests
    - Follow-up: Give an example of each.

### Advanced Questions
44. **How do you test your preprocessing pipeline?**
    - Expected: Test fit/transform separately, check output shapes, validate no leakage
    - Follow-up: How do you test for data leakage?

45. **How do you test your cross-validation implementation?**
    - Expected: Test fold splits, OOF predictions, artifact saving
    - Follow-up: How do you ensure reproducibility in tests?

46. **What would you add to improve test coverage?**
    - Expected: Property-based testing, performance tests, end-to-end tests
    - Follow-up: What are the challenges?

---

## 8. System Design & Architecture

### Basic Questions
47. **Walk me through your project structure.**
    - Expected: Modular design with src/, tests/, docs/, configs/, artifacts/
    - Follow-up: Why this structure?

48. **How do you manage configuration?**
    - Expected: YAML schema, feature_list.csv, environment variables
    - Follow-up: What are the benefits of external configuration?

### Advanced Questions
49. **How would you scale this system to handle 1M predictions per day?**
    - Expected: Batch processing, model caching, horizontal scaling, async processing
    - Follow-up: What would be the bottlenecks?

50. **How would you implement A/B testing for model versions?**
    - Expected: Multi-model serving, traffic splitting, metric tracking
    - Follow-up: How do you ensure fair comparison?

51. **How would you handle model retraining in production?**
    - Expected: Automated pipeline, validation checks, gradual rollout
    - Follow-up: What triggers retraining?

52. **Explain your artifact management strategy.**
    - Expected: Per-fold artifacts, versioned exports, metadata tracking
    - Follow-up: How do you handle artifact storage at scale?

---

## 9. Performance & Optimization

### Basic Questions
53. **What are the memory requirements of your system?**
    - Expected: Server 50-100MB, sklearn model 100-200MB, TF model 300-500MB
    - Follow-up: How did you measure this?

54. **How long does training take?**
    - Expected: ANN 30-60s/fold, XGBoost 5-15s/fold, LightGBM 3-10s/fold
    - Follow-up: What are the bottlenecks?

### Advanced Questions
55. **How would you optimize inference latency?**
    - Expected: Model quantization, batch processing, caching, model distillation
    - Follow-up: What are the accuracy tradeoffs?

56. **You use tensorflow-cpu. When would you use GPU?**
    - Expected: Large datasets, deep networks, training time critical
    - Follow-up: What's the cost-benefit analysis?

57. **How would you reduce model size?**
    - Expected: Pruning, quantization, knowledge distillation, feature selection
    - Follow-up: What's the impact on accuracy?

---

## 10. ML Best Practices & Trade-offs

### Basic Questions
58. **What ML best practices did you follow?**
    - Expected: Reproducibility, fold-safe preprocessing, comprehensive evaluation, testing
    - Follow-up: Which was most challenging to implement?

59. **What would you do differently if you started over?**
    - Expected: Honest reflection on design choices
    - Follow-up: What did you learn?

### Advanced Questions
60. **Explain the bias-variance tradeoff in your model selection.**
    - Expected: Neural networks high variance, tree models high bias, regularization
    - Follow-up: How do you find the sweet spot?

61. **How do you handle concept drift in production?**
    - Expected: Monitoring, retraining triggers, model versioning
    - Follow-up: What metrics indicate drift?

62. **What are the ethical considerations for this churn prediction system?**
    - Expected: Fairness across customer segments, privacy, transparency
    - Follow-up: How would you test for bias?

63. **How would you explain your model's predictions to non-technical stakeholders?**
    - Expected: SHAP visualizations, feature importance, example predictions
    - Follow-up: What if they disagree with the model?

---

## 11. Deep Dive: Technical Implementation

### Code-Level Questions
64. **Walk me through how you implemented fold-safe preprocessing in cross-validation.**
    - Expected: Fit transformers on train indices, transform both train and val
    - Follow-up: Show me the code structure.

65. **How do you handle variable-length categorical vocabularies across folds?**
    - Expected: Fit on train fold, handle unknown categories in validation
    - Follow-up: What happens with unseen categories?

66. **Explain your TabularANN._prepare_inputs() method.**
    - Expected: Splits flat array into dict matching model inputs using feature_info
    - Follow-up: Why this design?

67. **How do you save and load normalizer layers?**
    - Expected: Serialize config + weights, deserialize on load
    - Follow-up: What challenges did you face?

68. **Walk me through your OOF prediction generation.**
    - Expected: Collect predictions from each fold's validation set, concatenate
    - Follow-up: How do you ensure correct ordering?

### Design Pattern Questions
69. **What design patterns did you use in this project?**
    - Expected: Factory pattern (model creation), Strategy pattern (calibration), Pipeline pattern
    - Follow-up: Why these patterns?

70. **How would you refactor this code for better maintainability?**
    - Expected: Abstract base classes, dependency injection, configuration management
    - Follow-up: What are the tradeoffs?

---

## 12. Debugging & Troubleshooting

### Scenario-Based Questions
71. **Your model performs well in CV but poorly in production. What do you check?**
    - Expected: Data distribution shift, label leakage, preprocessing bugs, temporal issues
    - Follow-up: How do you debug each?

72. **Your API is returning 500 errors. How do you debug?**
    - Expected: Check logs, validate input format, test model loading, check dependencies
    - Follow-up: What monitoring would help?

73. **Your model's calibration is poor. What could be wrong?**
    - Expected: Insufficient calibration data, distribution shift, model overconfidence
    - Follow-up: How do you fix it?

74. **Training is taking too long. How do you optimize?**
    - Expected: Reduce data size, simplify model, optimize preprocessing, use GPU
    - Follow-up: What's the impact on accuracy?

---

## 13. Extensions & Future Work

### Open-Ended Questions
75. **How would you add feature engineering to this pipeline?**
    - Expected: Temporal features, interaction features, aggregations
    - Follow-up: How do you validate new features?

76. **How would you implement online learning?**
    - Expected: Incremental updates, sliding window, model versioning
    - Follow-up: What are the challenges?

77. **How would you add multi-class classification support?**
    - Expected: Change output layer, update metrics, modify calibration
    - Follow-up: What breaks in current implementation?

78. **How would you implement model ensembling?**
    - Expected: Weighted averaging, stacking, blending
    - Follow-up: How do you determine weights?

79. **How would you add AutoML capabilities?**
    - Expected: Automated feature engineering, hyperparameter optimization, model selection
    - Follow-up: What libraries would you use?

80. **How would you deploy this on Kubernetes?**
    - Expected: Containerization, horizontal pod autoscaling, service mesh
    - Follow-up: What are the challenges?

---

## 14. Comparison & Trade-offs

### Comparative Questions
81. **Why did you choose TensorFlow over PyTorch?**
    - Expected: SavedModel format, production serving, ecosystem
    - Follow-up: When would you choose PyTorch?

82. **Compare XGBoost vs LightGBM vs Neural Networks for this problem.**
    - Expected: Speed, accuracy, interpretability, deployment tradeoffs
    - Follow-up: Which would you choose for production?

83. **Why FastAPI over Flask or Django?**
    - Expected: Performance, async support, automatic API docs, type hints
    - Follow-up: What are the downsides?

84. **Platt scaling vs Isotonic regression - when to use each?**
    - Expected: Platt for parametric, Isotonic for non-parametric, data size considerations
    - Follow-up: Which is more robust?

---

## 15. Real-World Scenarios

### Behavioral/Situational Questions
85. **A stakeholder wants 99% recall. How do you respond?**
    - Expected: Discuss precision-recall tradeoff, business impact, threshold tuning
    - Follow-up: How do you find the optimal threshold?

86. **Your model is biased against a customer segment. How do you handle this?**
    - Expected: Fairness metrics, resampling, fairness constraints, stakeholder communication
    - Follow-up: What are the ethical implications?

87. **You need to deploy tomorrow but accuracy is only 80%. What do you do?**
    - Expected: Risk assessment, baseline comparison, monitoring plan, stakeholder communication
    - Follow-up: How do you prioritize?

88. **A competitor claims 95% accuracy. Your model is 85%. How do you respond?**
    - Expected: Question metric definition, dataset differences, reproducibility, business value
    - Follow-up: What questions would you ask?

---

## 16. Advanced ML Concepts

### Theoretical Questions
89. **Explain the mathematical foundation of SHAP values.**
    - Expected: Shapley values from cooperative game theory, fair attribution
    - Follow-up: What are the computational challenges?

90. **How does early stopping prevent overfitting?**
    - Expected: Monitors validation loss, stops when no improvement, prevents memorization
    - Follow-up: What's the optimal patience value?

91. **Explain the relationship between logits, probabilities, and log-odds.**
    - Expected: Logit = log(p/(1-p)), sigmoid(logit) = probability
    - Follow-up: Why are logits useful?

92. **What is the Expected Calibration Error measuring exactly?**
    - Expected: Weighted average of calibration error across probability bins
    - Follow-up: How do you choose number of bins?

93. **Explain gradient boosting vs neural networks for tabular data.**
    - Expected: Feature interactions, training dynamics, inductive biases
    - Follow-up: Why do tree models often win on tabular data?

---

## 17. Documentation & Communication

### Soft Skills Questions
94. **How did you document this project?**
    - Expected: README, API docs, problem spec, data contract, walkthroughs
    - Follow-up: Who is your target audience?

95. **How would you onboard a new team member to this codebase?**
    - Expected: Documentation, code walkthrough, pair programming, tests
    - Follow-up: What would you improve?

96. **How do you communicate model performance to business stakeholders?**
    - Expected: Business metrics, visualizations, examples, impact analysis
    - Follow-up: How do you handle pushback?

---

## 18. Project Management & Workflow

### Process Questions
97. **How did you prioritize features in this project?**
    - Expected: MVP first, iterative development, stakeholder feedback
    - Follow-up: What would you cut for a faster delivery?

98. **How do you handle technical debt in ML projects?**
    - Expected: Refactoring, testing, documentation, monitoring
    - Follow-up: Give an example from this project.

99. **What was the most challenging part of this project?**
    - Expected: Honest reflection on technical or organizational challenges
    - Follow-up: How did you overcome it?

100. **If you had 2 more weeks, what would you add?**
     - Expected: Feature engineering, AutoML, better monitoring, deployment automation
     - Follow-up: What's the highest priority?

---

## Interview Tips for Presenting This Project

### Preparation Checklist
- [ ] Be ready to live-code key components (preprocessing, model creation, CV)
- [ ] Prepare architecture diagrams (data flow, system design)
- [ ] Have metrics and results ready (CV scores, calibration curves)
- [ ] Understand every line of code you wrote
- [ ] Be honest about limitations and areas for improvement
- [ ] Prepare 2-3 interesting technical challenges you solved
- [ ] Have examples of predictions and explanations ready
- [ ] Know the business context and impact

### Key Talking Points
1. **End-to-end ML lifecycle** - From data loading to production serving
2. **Production-ready design** - Testing, monitoring, serving, versioning
3. **Best practices** - Fold-safe preprocessing, calibration, explainability
4. **Scalability** - Modular design, efficient serving, batch processing
5. **Trade-offs** - Model complexity vs interpretability, speed vs accuracy

### Red Flags to Avoid
- ❌ Not understanding your own code
- ❌ Claiming 100% accuracy or perfect model
- ❌ Ignoring data leakage concerns
- ❌ Not knowing when to use which model
- ❌ Unable to explain design decisions
- ❌ No testing or validation strategy
- ❌ Ignoring production considerations

### Impressive Highlights
- ✅ Fold-safe preprocessing preventing data leakage
- ✅ Comprehensive evaluation with calibration
- ✅ Multiple model architectures with fair comparison
- ✅ Production-ready API with lazy loading
- ✅ Explainability with SHAP and permutation importance
- ✅ Extensive testing and documentation
- ✅ Nested CV with HPO
- ✅ Schema-driven data validation

---

## Question Difficulty Distribution

- **Junior Level (1-2 years)**: Questions 1-30, 42-48, 53-54, 94-95
- **Mid Level (2-4 years)**: Questions 31-41, 49-52, 55-70, 81-84, 96-98
- **Senior Level (4+ years)**: Questions 71-80, 85-93, 99-100

---

**Good luck with your interviews! This project demonstrates strong ML engineering skills and production-ready thinking.**
