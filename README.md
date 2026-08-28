# Flipkart Order Intelligence & Support Assistant

An end-to-end AI/ML support system combining:

* **Part 1:** Return-risk prediction for customer orders
* **Part 2:** Product image classification using Fashion-MNIST
* **Part 3:** Policy retrieval, tool calling, grounded responses, and a LangGraph-based support agent

The project is designed as a modular support assistant that can answer policy questions, estimate the probability of an order being returned, and classify a product image.

---

## Project Overview

The system contains three connected components.

### Part 1 — Return-Risk Prediction

A machine-learning pipeline predicts the probability that an order will be returned.

The pipeline includes:

* Synthetic order dataset generation
* Data preprocessing
* Baseline evaluation
* Logistic Regression evaluation
* Random Forest training
* Cross-validation
* Threshold tuning
* Saved model verification

The final trained Random Forest model is stored in:

```text
part1/models/return_risk_model.pkl
```

The tuned Random Forest threshold is stored in:

```text
part1/models/return_risk_threshold.txt
```

---

### Part 2 — Product Image Classifier

A transfer-learning image classifier is trained using Fashion-MNIST.

The classifier recognizes the following product categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The trained model is stored in:

```text
part2/models/product_classifier.pt
```

Evaluation outputs are stored in:

```text
part2/results/
```

including:

```text
confusion_matrix.csv
per_class_metrics.csv
test_accuracy.txt
```

---

### Part 3 — AI Support Agent

Part 3 combines:

* Policy knowledge base
* Document chunking
* Vector indexing
* FAISS retrieval
* Retrieval evaluation
* Return-risk prediction tool
* Product-image classification tool
* LangGraph routing
* Prompt-injection protection
* Output groundedness protection
* Conversation state handling
* Mock LLM response generation
* Automated integration tests

The main agent is:

```text
part3/agent.py
```

The vector index is stored in:

```text
indexes/
```

The Part 3 test transcript is stored in:

```text
transcripts/part3_test_transcript.txt
```

---

# Architecture

```text
                         ┌──────────────────────────┐
                         │       User Query         │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    LangGraph Agent       │
                         │      part3/agent.py      │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                         ▼            ▼            ▼
                    ┌─────────┐ ┌──────────┐ ┌──────────────┐
                    │ Policy  │ │ Return   │ │ Product      │
                    │ Retrieval│ │ Risk Tool│ │ Image Tool   │
                    └────┬────┘ └────┬─────┘ └──────┬───────┘
                         │           │               │
                         ▼           ▼               ▼
                    FAISS Index   RF Model     Classifier
                         │           │               │
                         ▼           ▼               ▼
                    Policy KB   Probability     Category
                         │           │               │
                         └───────────┼───────────────┘
                                     ▼
                            ┌─────────────────┐
                            │ Grounded Answer │
                            └─────────────────┘
```

---

# Repository Structure

The repository intentionally keeps the existing project structure unchanged.

```text
flipkart-order-intelligence/
│
├── .gitignore
├── README.md
├── requirements.txt
├── run_tests.py
│
├── indexes/
│   ├── policy_chunks.json
│   └── policy_index.faiss
│
├── part1/
│   ├── generate_orders.py
│   ├── orders_dataset.csv
│   ├── results.txt
│   ├── train_return_risk.py
│   ├── verify_saved_model.py
│   │
│   └── models/
│       ├── return_risk_model.pkl
│       └── return_risk_threshold.txt
│
├── part2/
│   ├── train_classifier.py
│   │
│   ├── data/
│   │   ├── cache/
│   │   ├── FashionMNIST/
│   │   └── sample_images/
│   │
│   ├── models/
│   │   └── product_classifier.pt
│   │
│   └── results/
│       ├── confusion_matrix.csv
│       ├── per_class_metrics.csv
│       └── test_accuracy.txt
│
├── part3/
│   ├── agent.py
│   ├── build_index.py
│   ├── chunk_policies.py
│   ├── chunks.json
│   ├── evaluate_retrieval.py
│   ├── generate_answer.py
│   ├── policies.json
│   └── retrieve.py
│
└── transcripts/
    └── part3_test_transcript.txt
```

Generated Python cache directories such as `__pycache__/` are excluded from Git.

Large downloaded/cache datasets are also excluded where appropriate.

---

# Part 1 — Return-Risk Prediction

## Purpose

Predict the probability that an order will be returned.

The model uses order-level features including:

* Product category
* Price
* Discount percentage
* Payment method
* Customer tenure
* Previous orders
* Previous returns
* Delivery distance
* Delivery time
* Weekend order indicator
* Rating information

## Generate Dataset

From the repository root:

```bash
python part1/generate_orders.py
```

This generates:

```text
part1/orders_dataset.csv
```

## Train and Evaluate

```bash
python part1/train_return_risk.py
```

The trained model is saved as:

```text
part1/models/return_risk_model.pkl
```

The tuned threshold is saved as:

```text
part1/models/return_risk_threshold.txt
```

## Verify Saved Model

```bash
python part1/verify_saved_model.py
```

---

# Part 2 — Product Image Classifier

## Purpose

Classify product images into Fashion-MNIST categories.

The model is based on transfer learning and is saved as:

```text
part2/models/product_classifier.pt
```

## Train and Evaluate

From the repository root:

```bash
python part2/train_classifier.py
```

Evaluation outputs are written to:

```text
part2/results/
```

The results include:

```text
confusion_matrix.csv
per_class_metrics.csv
test_accuracy.txt
```

The downloaded Fashion-MNIST raw files and cached feature tensors are not required to be committed to Git and are excluded by `.gitignore`.

---

# Part 3 — Policy Retrieval and Support Agent

## Knowledge Base

Policy data is stored in:

```text
part3/policies.json
```

Policy chunks are generated using:

```bash
python part3/chunk_policies.py
```

The resulting chunks are stored in:

```text
part3/chunks.json
```

## Build Vector Index

Run:

```bash
python part3/build_index.py
```

This creates:

```text
indexes/policy_chunks.json
indexes/policy_index.faiss
```

## Evaluate Retrieval

Run:

```bash
python part3/evaluate_retrieval.py
```

Current retrieval evaluation:

```text
Average Precision@3 = 0.3889
Average Recall@3    = 1.0000
```

The evaluation used six policy queries.

### Retrieval Results

| Query                                                     | Precision@3 | Recall@3 |
| --------------------------------------------------------- | ----------: | -------: |
| How many days can I return apparel?                       |      0.3333 |   1.0000 |
| How long do I have to return electronics?                 |      0.3333 |   1.0000 |
| How many days do I have to return a home product?         |      0.3333 |   1.0000 |
| What happens to a cash-on-delivery refund after a return? |      0.5000 |   1.0000 |
| Can an eligible return be collected from my address?      |      0.5000 |   1.0000 |
| What should I do if my product arrives damaged?           |      0.3333 |   1.0000 |

Final:

```text
Average Precision@3 = 0.3889
Average Recall@3    = 1.0000
```

The perfect Recall@3 indicates that every relevant policy document was retrieved within the top three results for the evaluation queries.

---

# Part 3 Tools

The support agent exposes two main tools.

## Return-Risk Tool

The return-risk tool loads the saved Part 1 model and predicts:

```text
predicted_return_probability
risk_bucket
t_rf
bucket_cut_points
```

The saved Random Forest threshold is used to divide predictions into:

```text
Low
Medium
High
```

The currently saved threshold is:

```text
t*_rf = 0.4700
```

---

## Product Image Classification Tool

The product-image tool loads:

```text
part2/models/product_classifier.pt
```

and predicts the product category and confidence.

---

# LangGraph Agent

The main agent is implemented in:

```text
part3/agent.py
```

The graph contains the following required nodes:

```text
intent
retrieval
tool
response
```

The agent conditionally routes requests depending on their detected intent.

Supported behavior includes:

* Policy questions
* Return-risk questions
* Product-image classification
* Multi-turn order state
* Fresh conversation reset
* Prompt-injection protection
* Retrieval-grounded responses
* Mock LLM mode

---

# Running the Support Agent

From the repository root:

```bash
python part3/agent.py
```

The program starts an interactive support-agent session.

Example:

```text
Enter your query: what is the return policy for damaged items?
```

---

# Automated Part 3 Test Suite

Run the complete test suite from the repository root:

```bash
python run_tests.py
```

The suite covers:

1. Damaged-product policy retrieval
2. COD refund policy retrieval
3. Return-risk tool
4. Product-image classifier
5. Multi-turn conversation state
6. Fresh conversation reset
7. Prompt-injection guardrail
8. Output-groundedness guardrail
9. Saved-model spot check
10. LangGraph structure

The complete test transcript is saved to:

```text
transcripts/part3_test_transcript.txt
```

The final test run completed with:

```text
Passed: 10
Failed: 0
Total:  10

ALL PART 3 TESTS PASSED
```

---

# Guardrails

## Prompt Injection

The agent blocks instructions attempting to override its support-agent rules.

Example behavior:

```text
I can't follow instructions that attempt to override the support assistant's rules.
```

---

## Output Groundedness

Policy answers are checked against the configured similarity threshold.

Current threshold:

```text
0.4000
```

If retrieved policy evidence is insufficiently relevant, the system returns:

```text
I could not find sufficiently relevant information in the policy knowledge base to answer this question.
```

This prevents unsupported policy answers.

---

# Requirements

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

Recommended Python environment:

```text
Python 3.13
```

The project uses libraries including:

* pandas
* NumPy
* scikit-learn
* joblib
* PyTorch
* torchvision
* FAISS
* LangGraph
* sentence-transformers

Exact dependencies are listed in:

```text
requirements.txt
```

---

# Recommended Execution Order

For a fresh checkout, the recommended order is:

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Generate Part 1 dataset

```bash
python part1/generate_orders.py
```

## 3. Train Part 1 model

```bash
python part1/train_return_risk.py
```

## 4. Verify Part 1 saved model

```bash
python part1/verify_saved_model.py
```

## 5. Train Part 2 classifier

```bash
python part2/train_classifier.py
```

## 6. Build Part 3 policy chunks

```bash
python part3/chunk_policies.py
```

## 7. Build Part 3 FAISS index

```bash
python part3/build_index.py
```

## 8. Evaluate retrieval

```bash
python part3/evaluate_retrieval.py
```

## 9. Run complete Part 3 test suite

```bash
python run_tests.py
```

## 10. Run interactive support agent

```bash
python part3/agent.py
```

---

# Final Submission Artifacts

The repository contains the required artifacts for all three parts.

## Part 1

```text
part1/generate_orders.py
part1/orders_dataset.csv
part1/train_return_risk.py
part1/verify_saved_model.py
part1/models/return_risk_model.pkl
part1/models/return_risk_threshold.txt
part1/results.txt
```

## Part 2

```text
part2/train_classifier.py
part2/models/product_classifier.pt
part2/results/confusion_matrix.csv
part2/results/per_class_metrics.csv
part2/results/test_accuracy.txt
```

## Part 3

```text
part3/policies.json
part3/chunks.json
part3/chunk_policies.py
part3/build_index.py
part3/retrieve.py
part3/evaluate_retrieval.py
part3/generate_answer.py
part3/agent.py

indexes/policy_chunks.json
indexes/policy_index.faiss

transcripts/part3_test_transcript.txt
```

The repository also contains:

```text
run_tests.py
requirements.txt
README.md
.gitignore
```

---

# Submission

Only the public GitHub repository URL is submitted.

No separate files, screenshots, slides, videos, audio files, or PDF exports are submitted.

All required project artifacts are contained within this repository.

Main is the default branch with all up to date commits to check and grade upon.

feature/part1-return-risk is just a feature branch and should not be considered for grading.
