# 100 Days of ML From Scratch 🚀

Implementing core Machine Learning algorithms from scratch (NumPy only, no sklearn/tensorflow for the core logic) — one small step a day. Built to deepen ML fundamentals as an AI & Data Science student and to maintain a consistent GitHub streak with *real* progress.

## Rules
- No `sklearn.linear_model`, no `sklearn.tree`, etc. for the core algorithm — only NumPy/pandas for data handling, matplotlib for plots.
- Every algorithm folder has: `algorithm.py` (from-scratch implementation), `demo.ipynb` (usage + comparison with sklearn's version), and a short `notes.md` (math + intuition).
- Commit daily, even if it's just notes, a bug fix, or a visualization added to yesterday's algorithm.

## Repo Structure
```
100-days-of-ml/
├── README.md
├── utils/
│   ├── metrics.py        # accuracy, MSE, precision/recall, confusion matrix
│   ├── preprocessing.py  # scaling, train_test_split, encoding
│   └── plotting.py       # decision boundary plots, loss curves
├── day01_linear_regression/
├── day02_gradient_descent_variants/
├── ...
└── day100_capstone_project/
```

---

## Phase 1 (Days 1–15): Foundations & Regression
| Day | Topic |
|---|---|
| 1 | Linear Regression (Normal Equation) |
| 2 | Linear Regression (Batch Gradient Descent) |
| 3 | Stochastic & Mini-batch Gradient Descent |
| 4 | Polynomial Regression |
| 5 | Ridge Regression (L2) |
| 6 | Lasso Regression (L1) |
| 7 | Elastic Net |
| 8 | Logistic Regression (Binary) |
| 9 | Logistic Regression (Multiclass, Softmax) |
| 10 | Evaluation metrics module (accuracy, precision, recall, F1, ROC-AUC) |
| 11 | Train/test split + K-Fold CV from scratch |
| 12 | Feature scaling (standardization, normalization) from scratch |
| 13 | Regularization visualized (bias-variance tradeoff demo) |
| 14 | Perceptron |
| 15 | Mini-project: Predict house prices (regression pipeline using Days 1–14 code) |

## Phase 2 (Days 16–30): Distance & Tree-Based Methods
| Day | Topic |
|---|---|
| 16 | K-Nearest Neighbors (classification) |
| 17 | K-Nearest Neighbors (regression) |
| 18 | KD-Tree for faster KNN lookup |
| 19 | Decision Tree (Gini impurity, classification) |
| 20 | Decision Tree (entropy/information gain) |
| 21 | Decision Tree (regression, variance reduction) |
| 22 | Pruning a decision tree |
| 23 | Random Forest (bagging + feature randomness) |
| 24 | AdaBoost |
| 25 | Gradient Boosting (regression) |
| 26 | Gradient Boosting (classification) |
| 27 | XGBoost-style tree with regularization (simplified) |
| 28 | Feature importance from trees |
| 29 | Handling categorical variables (one-hot, target encoding) from scratch |
| 30 | Mini-project: Titanic survival prediction (compare all tree methods) |

## Phase 3 (Days 31–45): Probabilistic & Bayesian Methods
| Day | Topic |
|---|---|
| 31 | Naive Bayes (Gaussian) |
| 32 | Naive Bayes (Multinomial, for text) |
| 33 | Naive Bayes (Bernoulli) |
| 34 | Maximum Likelihood Estimation demo |
| 35 | Bayesian Linear Regression (intro) |
| 36 | Hidden Markov Model (basic) |
| 37 | Gaussian Mixture Models (intro to EM) |
| 38 | Expectation-Maximization algorithm |
| 39 | Text preprocessing pipeline (tokenize, TF-IDF from scratch) |
| 40 | Spam classifier using Naive Bayes + your TF-IDF |
| 41 | Bag-of-Words vs TF-IDF comparison |
| 42 | N-gram language model |
| 43 | Laplace smoothing implementation |
| 44 | Confusion matrix visualizer (upgrade utils/) |
| 45 | Mini-project: Email spam / sentiment classifier |

## Phase 4 (Days 46–60): Support Vector Machines & Optimization
| Day | Topic |
|---|---|
| 46 | SVM (hard margin, linear) |
| 47 | SVM (soft margin) |
| 48 | Kernel trick (polynomial kernel) |
| 49 | Kernel trick (RBF kernel) |
| 50 | SVM via SMO (simplified) |
| 51 | Gradient Descent variants: Momentum |
| 52 | RMSProp from scratch |
| 53 | Adam optimizer from scratch |
| 54 | Loss functions module (MSE, Cross-Entropy, Hinge, Huber) |
| 55 | Learning rate schedulers |
| 56 | Early stopping implementation |
| 57 | Hyperparameter tuning (grid search from scratch) |
| 58 | Hyperparameter tuning (random search) |
| 59 | Bias-variance tradeoff experiment suite |
| 60 | Mini-project: Digit classification with your own SVM |

## Phase 5 (Days 61–75): Unsupervised Learning
| Day | Topic |
|---|---|
| 61 | K-Means clustering |
| 62 | K-Means++ initialization |
| 63 | Hierarchical clustering (agglomerative) |
| 64 | DBSCAN |
| 65 | Silhouette score from scratch |
| 66 | Elbow method visualizer |
| 67 | PCA (from scratch, via eigen decomposition) |
| 68 | PCA (via SVD) |
| 69 | t-SNE (conceptual, simplified implementation) |
| 70 | Anomaly detection (Gaussian-based) |
| 71 | Isolation Forest (simplified) |
| 72 | Association rule mining (Apriori algorithm) |
| 73 | Recommender system: user-based collaborative filtering |
| 74 | Recommender system: item-based collaborative filtering |
| 75 | Mini-project: Customer segmentation (K-Means + PCA) |

## Phase 6 (Days 76–90): Neural Networks From Scratch
| Day | Topic |
|---|---|
| 76 | Single neuron + activation functions module |
| 77 | Forward propagation (2-layer NN) |
| 78 | Backpropagation (manual gradient derivation) |
| 79 | Full NN class (configurable layers) using only NumPy |
| 80 | Batch training + mini-batches for your NN |
| 81 | Add dropout regularization |
| 82 | Add batch normalization |
| 83 | Weight initialization strategies (Xavier, He) |
| 84 | Softmax + cross-entropy for multiclass NN |
| 85 | Convolution operation from scratch (single filter) |
| 86 | Build a mini CNN forward pass from scratch |
| 87 | Max pooling from scratch |
| 88 | Simple RNN cell from scratch |
| 89 | Backprop through time (BPTT) intuition demo |
| 90 | Mini-project: Handwritten digit classifier (your NN vs sklearn's MLP) |

## Phase 7 (Days 91–100): Capstone & Polish
| Day | Topic |
|---|---|
| 91 | Model comparison dashboard (all algorithms, one dataset) |
| 92 | Cross-validation framework across all your models |
| 93 | Write unit tests for utils/ and 3 core algorithms |
| 94 | Add type hints + docstrings across repo |
| 95 | Package repo as installable module (`pip install -e .`) |
| 96 | Write a blog-style post: "What I learned building 90 ML algorithms" |
| 97–99 | Capstone: end-to-end project using 5+ algorithms from the repo |
| 100 | Final polish: README overhaul, project write-up, LinkedIn/GitHub post |

---

## Tips for Commit Quality (not just quantity)
- Commit message format: `Day XX: <Algorithm> - <what you did>` e.g. `Day 23: Random Forest - added feature bagging`
- If a day's algorithm is too big for one sitting, split naturally: implementation one day, demo/visualization the next — both are legitimate commits.
- Keep a `notes.md` per folder with the math intuition in your own words — this is gold for interviews.
- Every 15 days = 1 mini-project tying the phase together — these are your strongest portfolio pieces.
