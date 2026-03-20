# Feature Improvements for ML Pipeline Project

## Overview
These 5 features are designed to deepen your understanding of different aspects of the ML pipeline while adding real value to the project. Each feature is scoped as a small-to-medium task (2-8 hours) and touches different components of the system.

---

## Feature 1: Model Performance Dashboard 📊

### Difficulty: Medium
### Time Estimate: 4-6 hours
### Learning Focus: Visualization, Metrics Analysis, Web Development

### Description
Create an interactive HTML dashboard that visualizes model performance metrics from cross-validation runs. This will help you understand model evaluation deeply and practice data visualization.

### What You'll Build
- **Interactive Dashboard** using Plotly or Matplotlib + HTML
- **Metrics Visualization**: ROC curves, PR curves, calibration plots, confusion matrices
- **Comparison View**: Side-by-side comparison of ANN vs XGBoost vs LightGBM
- **Feature Importance**: Bar charts showing top features from permutation importance
- **Training History**: Loss curves, learning rate schedules

### Implementation Steps
1. **Create `src/visualization/dashboard.py`**
   ```python
   def generate_dashboard(cv_run_dir: str, output_path: str = "dashboard.html"):
       """Generate interactive HTML dashboard from CV run artifacts."""
       # Load metrics from all folds
       # Create plotly figures for each metric
       # Combine into single HTML file
   ```

2. **Add metrics loading utility**
   ```python
   def load_cv_metrics(cv_run_dir: str) -> Dict[str, Any]:
       """Load all metrics from a CV run."""
       # Read fold_metrics.json from each fold
       # Aggregate across folds
       # Return structured metrics dict
   ```

3. **Create visualization functions**
   - `plot_roc_curves()` - ROC curve for each fold + average
   - `plot_calibration_curves()` - Calibration plots before/after
   - `plot_feature_importance()` - Top 20 features with error bars
   - `plot_confusion_matrix()` - Aggregated confusion matrix
   - `plot_training_history()` - Loss/accuracy over epochs

4. **Add CLI command**
   ```bash
   python -m src.visualization.dashboard --run-dir artifacts/cv/latest --output dashboard.html
   ```

### What You'll Learn
- How to read and aggregate metrics from multiple folds
- Understanding of different evaluation metrics visually
- Plotly/Matplotlib for interactive visualizations
- How calibration affects probability distributions
- Feature importance interpretation

### Bonus Extensions
- Add model comparison across multiple runs
- Include SHAP summary plots
- Add filtering by fold or metric threshold
- Export dashboard as PDF report

---

## Feature 2: Automated Feature Engineering Pipeline 🔧

### Difficulty: Medium
### Time Estimate: 5-8 hours
### Learning Focus: Feature Engineering, Data Transformations, Pipeline Design

### Description
Implement an automated feature engineering module that creates temporal, interaction, and aggregation features. This will deepen your understanding of feature engineering and how it integrates with preprocessing.

### What You'll Build
- **Temporal Features**: Days since events, time-based aggregations
- **Interaction Features**: Product/ratio of related features
- **Aggregation Features**: Rolling statistics, percentile ranks
- **Automatic Feature Selection**: Based on correlation/importance

### Implementation Steps
1. **Create `src/feature_engineering/engineer.py`**
   ```python
   class FeatureEngineer:
       def __init__(self, config_path: str = "configs/feature_engineering.yaml"):
           """Initialize with feature engineering rules."""
           
       def fit(self, df: pd.DataFrame):
           """Learn feature engineering parameters from training data."""
           
       def transform(self, df: pd.DataFrame) -> pd.DataFrame:
           """Apply feature engineering transformations."""
           
       def get_feature_names(self) -> List[str]:
           """Return names of engineered features."""
   ```

2. **Implement feature generators**
   ```python
   def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
       """Create time-based features."""
       # days_since_subscription
       # login_frequency_trend (last 7 vs 30 days)
       # payment_recency_score
       
   def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
       """Create feature interactions."""
       # login_frequency * session_duration (engagement score)
       # support_tickets / subscription_duration (support intensity)
       # email_clicks / email_opens (email engagement rate)
       
   def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
       """Create statistical aggregations."""
       # percentile_rank of monthly_revenue
       # z-score of login_frequency
       # binned age groups
   ```

3. **Create configuration file `configs/feature_engineering.yaml`**
   ```yaml
   temporal_features:
     - name: login_frequency_trend
       type: ratio
       numerator: login_frequency_last_7d
       denominator: login_frequency_last_30d
       
   interaction_features:
     - name: engagement_score
       type: product
       features: [login_frequency, session_duration_avg]
       
   aggregation_features:
     - name: revenue_percentile
       type: percentile_rank
       feature: monthly_revenue
   ```

4. **Integrate with preprocessing pipeline**
   ```python
   # In lightweight_transformers.py
   def create_lightweight_pipeline(enable_feature_engineering: bool = False):
       if enable_feature_engineering:
           engineer = FeatureEngineer()
           # Add to pipeline
   ```

5. **Add tests in `tests/test_feature_engineering.py`**

### What You'll Learn
- How feature engineering impacts model performance
- Temporal feature creation and its importance
- Interaction features and domain knowledge
- How to make feature engineering fold-safe
- Configuration-driven feature generation

### Validation
- Train model with/without engineered features
- Compare ROC-AUC improvement
- Check feature importance of new features
- Ensure no data leakage in temporal features

### Bonus Extensions
- Automatic feature selection based on importance
- Polynomial features for numeric columns
- Target encoding for high-cardinality categoricals
- Feature interaction discovery using decision trees

---

## Feature 3: Model Drift Detection & Monitoring 🔍

### Difficulty: Small-Medium
### Time Estimate: 3-5 hours
### Learning Focus: Production ML, Statistical Testing, Monitoring

### Description
Implement a drift detection system that monitors data distribution changes and model performance degradation. This is crucial for production ML systems.

### What You'll Build
- **Data Drift Detection**: Statistical tests for feature distribution changes
- **Prediction Drift**: Monitor prediction distribution over time
- **Performance Monitoring**: Track metrics on recent data
- **Alert System**: Flag when drift exceeds thresholds

### Implementation Steps
1. **Create `src/monitoring/drift_detector.py`**
   ```python
   class DriftDetector:
       def __init__(self, reference_data: pd.DataFrame, 
                    reference_predictions: np.ndarray,
                    threshold: float = 0.05):
           """Initialize with reference (training) data."""
           self.reference_stats = self._compute_statistics(reference_data)
           self.reference_pred_dist = self._compute_pred_distribution(reference_predictions)
           
       def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
           """Detect drift in feature distributions using KS test."""
           
       def detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict[str, Any]:
           """Detect drift in prediction distribution."""
           
       def generate_drift_report(self) -> str:
           """Generate human-readable drift report."""
   ```

2. **Implement statistical tests**
   ```python
   def kolmogorov_smirnov_test(reference: np.ndarray, 
                                current: np.ndarray,
                                threshold: float = 0.05) -> Dict[str, Any]:
       """KS test for continuous features."""
       from scipy.stats import ks_2samp
       statistic, p_value = ks_2samp(reference, current)
       return {
           'statistic': statistic,
           'p_value': p_value,
           'drift_detected': p_value < threshold
       }
       
   def chi_square_test(reference: np.ndarray,
                       current: np.ndarray,
                       threshold: float = 0.05) -> Dict[str, Any]:
       """Chi-square test for categorical features."""
       from scipy.stats import chisquare
       # Compute frequency distributions
       # Run chi-square test
   ```

3. **Add monitoring metrics**
   ```python
   def compute_psi(reference: np.ndarray, 
                   current: np.ndarray,
                   bins: int = 10) -> float:
       """Population Stability Index (PSI) for drift detection."""
       # Bin both distributions
       # Compute PSI = sum((current% - reference%) * ln(current%/reference%))
       # PSI < 0.1: no drift, 0.1-0.2: moderate, >0.2: significant
   ```

4. **Create monitoring dashboard endpoint**
   ```python
   # In src/serve/app.py
   @app.post("/monitor/drift")
   def check_drift(request: DriftCheckRequest):
       """Check for data/prediction drift."""
       detector = DriftDetector(reference_data, reference_preds)
       drift_report = detector.detect_data_drift(request.current_data)
       return drift_report
   ```

5. **Add tests in `tests/test_drift_detection.py`**

### What You'll Learn
- Statistical tests for distribution comparison (KS test, Chi-square)
- Population Stability Index (PSI) calculation
- How data drift affects model performance
- Production monitoring best practices
- When to retrain models

### Validation
- Create synthetic drift scenarios (shift mean, change variance)
- Verify drift detection triggers correctly
- Test with real data from different time periods
- Measure false positive/negative rates

### Bonus Extensions
- Add concept drift detection (performance degradation)
- Implement CUSUM for sequential drift detection
- Add email/Slack alerts when drift detected
- Create drift visualization dashboard
- Implement automatic retraining triggers

---

## Feature 4: Hyperparameter Optimization with Optuna 🎯

### Difficulty: Medium
### Time Estimate: 4-6 hours
### Learning Focus: HPO, Bayesian Optimization, Experiment Tracking

### Description
Replace the simple grid search with Optuna for more efficient hyperparameter optimization. This will teach you modern HPO techniques and how to integrate them into your pipeline.

### What You'll Build
- **Optuna Integration**: Bayesian optimization for hyperparameters
- **Multi-objective Optimization**: Optimize for both accuracy and calibration
- **Pruning**: Early stopping of unpromising trials
- **Visualization**: Optimization history, parameter importance

### Implementation Steps
1. **Create `src/training/optuna_hpo.py`**
   ```python
   import optuna
   from optuna.pruners import MedianPruner
   from optuna.samplers import TPESampler
   
   class OptunaHPO:
       def __init__(self, n_trials: int = 50, 
                    timeout: int = 3600,
                    n_jobs: int = 1):
           """Initialize Optuna HPO."""
           self.n_trials = n_trials
           self.timeout = timeout
           self.n_jobs = n_jobs
           
       def objective(self, trial: optuna.Trial) -> float:
           """Objective function for a single trial."""
           # Suggest hyperparameters
           learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
           dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
           l2_reg = trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True)
           hidden_layers = trial.suggest_categorical('hidden_layers', 
                                                      [[128,64], [128,64,32], [256,128,64]])
           
           # Train model with these hyperparameters
           # Return validation metric
           
       def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
           """Run hyperparameter optimization."""
           study = optuna.create_study(
               direction='maximize',
               sampler=TPESampler(),
               pruner=MedianPruner()
           )
           study.optimize(self.objective, n_trials=self.n_trials, 
                         timeout=self.timeout, n_jobs=self.n_jobs)
           
           return {
               'best_params': study.best_params,
               'best_value': study.best_value,
               'study': study
           }
   ```

2. **Add multi-objective optimization**
   ```python
   def multi_objective(self, trial: optuna.Trial) -> Tuple[float, float]:
       """Optimize for both ROC-AUC and calibration (Brier score)."""
       # Train model
       # Return (roc_auc, -brier_score)  # Negative because we maximize
   ```

3. **Add pruning callback for early stopping**
   ```python
   class OptunaPruningCallback(tf.keras.callbacks.Callback):
       def __init__(self, trial: optuna.Trial, monitor: str = 'val_loss'):
           self.trial = trial
           self.monitor = monitor
           
       def on_epoch_end(self, epoch, logs=None):
           current_score = logs.get(self.monitor)
           self.trial.report(current_score, epoch)
           if self.trial.should_prune():
               raise optuna.TrialPruned()
   ```

4. **Create visualization functions**
   ```python
   def visualize_optimization(study: optuna.Study, output_dir: str):
       """Create optimization visualizations."""
       # Optimization history
       fig1 = optuna.visualization.plot_optimization_history(study)
       fig1.write_html(f"{output_dir}/optimization_history.html")
       
       # Parameter importance
       fig2 = optuna.visualization.plot_param_importances(study)
       fig2.write_html(f"{output_dir}/param_importance.html")
       
       # Parallel coordinate plot
       fig3 = optuna.visualization.plot_parallel_coordinate(study)
       fig3.write_html(f"{output_dir}/parallel_coordinate.html")
   ```

5. **Integrate with CV pipeline**
   ```python
   # In src/training/cv.py
   def run_kfold_cv(..., use_optuna: bool = False, optuna_trials: int = 50):
       if use_optuna:
           hpo = OptunaHPO(n_trials=optuna_trials)
           best_params = hpo.optimize(X_train, y_train)
           base_params.update(best_params['best_params'])
   ```

6. **Add CLI command**
   ```bash
   python -m src.training.optuna_hpo --data data/sample.csv --n-trials 50 --output artifacts/hpo
   ```

### What You'll Learn
- Bayesian optimization vs grid/random search
- Tree-structured Parzen Estimator (TPE) algorithm
- Multi-objective optimization trade-offs
- Pruning strategies for efficient search
- Hyperparameter importance analysis
- How to visualize optimization process

### Validation
- Compare Optuna vs grid search on same budget
- Measure time to find good hyperparameters
- Analyze parameter importance
- Test multi-objective optimization

### Bonus Extensions
- Add XGBoost/LightGBM hyperparameter optimization
- Implement warm-start from previous studies
- Add distributed optimization (parallel trials)
- Create automated HPO reports
- Integrate with MLflow for experiment tracking

---

## Feature 5: Explainability API with LIME 🔬

### Difficulty: Small-Medium
### Time Estimate: 3-4 hours
### Learning Focus: Model Interpretability, API Design, Instance-level Explanations

### Description
Add LIME (Local Interpretable Model-agnostic Explanations) to complement your existing SHAP implementation. Create API endpoints for on-demand explanations. This will deepen your understanding of different explainability techniques.

### What You'll Build
- **LIME Integration**: Instance-level explanations
- **Explanation API**: REST endpoints for explanations
- **Comparison Tool**: LIME vs SHAP side-by-side
- **Visualization**: HTML explanation reports

### Implementation Steps
1. **Create `src/explainability/lime_explainer.py`**
   ```python
   from lime.lime_tabular import LimeTabularExplainer
   
   class LIMEExplainer:
       def __init__(self, X_train: np.ndarray, 
                    feature_names: List[str],
                    categorical_features: List[int] = None):
           """Initialize LIME explainer."""
           self.explainer = LimeTabularExplainer(
               X_train,
               feature_names=feature_names,
               categorical_features=categorical_features,
               mode='classification'
           )
           
       def explain_instance(self, 
                           instance: np.ndarray,
                           predict_fn: Callable,
                           num_features: int = 10) -> Dict[str, Any]:
           """Explain a single prediction."""
           explanation = self.explainer.explain_instance(
               instance,
               predict_fn,
               num_features=num_features
           )
           
           return {
               'feature_contributions': dict(explanation.as_list()),
               'prediction': predict_fn(instance.reshape(1, -1))[0],
               'intercept': explanation.intercept[1],
               'local_r2': explanation.score
           }
           
       def explain_batch(self, 
                        instances: np.ndarray,
                        predict_fn: Callable) -> List[Dict[str, Any]]:
           """Explain multiple instances."""
           return [self.explain_instance(inst, predict_fn) 
                   for inst in instances]
   ```

2. **Add comparison utility**
   ```python
   def compare_lime_shap(instance: np.ndarray,
                         model: Any,
                         X_train: np.ndarray,
                         feature_names: List[str]) -> Dict[str, Any]:
       """Compare LIME and SHAP explanations for same instance."""
       # Get LIME explanation
       lime_exp = lime_explainer.explain_instance(instance, model.predict_proba)
       
       # Get SHAP explanation
       shap_exp = shap_explainer.shap_values(instance)
       
       # Compare top features
       return {
           'lime_top_features': lime_exp['feature_contributions'],
           'shap_top_features': dict(zip(feature_names, shap_exp)),
           'agreement_score': compute_agreement(lime_exp, shap_exp)
       }
   ```

3. **Add API endpoints in `src/serve/app.py`**
   ```python
   class ExplainRequest(BaseModel):
       features: Dict[str, Any]
       method: str = 'lime'  # 'lime', 'shap', or 'both'
       num_features: int = 10
   
   @app.post("/explain")
   def explain_prediction(request: ExplainRequest):
       """Get explanation for a prediction."""
       if not _model_state:
           raise HTTPException(status_code=400, detail="No model loaded")
       
       # Prepare input
       instance = prepare_inputs([request.features], _model_state['feature_info'])
       
       # Get prediction
       prediction = _model_state['model'].predict(instance)
       
       # Get explanation
       if request.method == 'lime':
           explanation = lime_explainer.explain_instance(
               instance[0], 
               _model_state['model'].predict_proba,
               num_features=request.num_features
           )
       elif request.method == 'shap':
           # SHAP explanation
           pass
       elif request.method == 'both':
           # Both explanations
           pass
       
       return {
           'prediction': float(prediction[0]),
           'explanation': explanation
       }
   ```

4. **Create visualization**
   ```python
   def create_explanation_html(explanation: Dict[str, Any],
                               instance: Dict[str, Any],
                               output_path: str):
       """Create HTML visualization of explanation."""
       # Create bar chart of feature contributions
       # Add instance values
       # Add prediction probability
       # Save as HTML
   ```

5. **Add tests in `tests/test_lime_explainer.py`**

### What You'll Learn
- How LIME works (local linear approximations)
- Difference between LIME and SHAP
- When to use each explainability method
- API design for ML explanations
- Instance-level vs global explanations

### Validation
- Compare LIME vs SHAP on same instances
- Measure explanation stability (run multiple times)
- Validate explanations make intuitive sense
- Test API endpoint performance

### Bonus Extensions
- Add counterfactual explanations (what-if analysis)
- Implement anchor explanations
- Create interactive explanation dashboard
- Add explanation caching for common patterns
- Implement explanation consistency checks

---

## Implementation Priority & Learning Path

### Recommended Order
1. **Start with Feature 2 (Feature Engineering)** - Deepens data understanding
2. **Then Feature 4 (Optuna HPO)** - Improves model performance
3. **Then Feature 3 (Drift Detection)** - Adds production awareness
4. **Then Feature 1 (Dashboard)** - Ties everything together visually
5. **Finally Feature 5 (LIME)** - Adds another explainability dimension

### Time Commitment
- **Minimum**: Pick 2 features (6-12 hours total)
- **Recommended**: Complete 3-4 features (12-20 hours total)
- **Maximum**: All 5 features (20-30 hours total)

---

## Success Criteria

### For Each Feature
- [ ] Code is well-tested (add pytest tests)
- [ ] Documentation is updated (README, docstrings)
- [ ] Integration with existing pipeline works
- [ ] You can explain the feature in an interview
- [ ] You understand the trade-offs and limitations

### Overall Learning Goals
- [ ] Understand end-to-end ML pipeline deeply
- [ ] Can explain design decisions confidently
- [ ] Know when to use different techniques
- [ ] Understand production ML considerations
- [ ] Can debug issues independently

---

## Additional Resources

### For Feature Engineering
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle Feature Engineering Course](https://www.kaggle.com/learn/feature-engineering)

### For HPO
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hyperparameter Optimization Guide](https://arxiv.org/abs/1807.02811)

### For Drift Detection
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Monitoring ML Models in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

### For Explainability
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

---

**Choose the features that interest you most and align with your learning goals. Each one will make you a stronger ML engineer!** 🚀
