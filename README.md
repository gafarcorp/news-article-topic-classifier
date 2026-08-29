# News Article Topic Classifier

An NLP-based machine learning system that automatically classifies a news article's title and short description into one of four topics — **World, Sports, Business, or Sci/Tech** — using TF-IDF features and traditional classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVM).

Reference paper: Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.* [arXiv:1509.01626](https://arxiv.org/abs/1509.01626)

---

## 1. Project Overview

News is published continuously and at high volume, and manually tagging every article by topic doesn't scale. This project builds a complete, reproducible pipeline that reads a news article's title and description and predicts its topic automatically — with documented preprocessing, feature extraction, multiple compared models, honest evaluation, error analysis, and a working prediction function.

**Results at a glance** (on the held-out test set, 7,600 articles):

| Model | Accuracy | Macro F1 | Train Time |
|---|---|---|---|
| **Logistic Regression** (tuned) | **92.22%** | **0.9221** | ~4.4s |
| Linear SVM | 92.03% | 0.9201 | ~2.2s |
| Multinomial Naive Bayes | 90.39% | 0.9036 | ~0.03s |

---

## 2. Dataset

**AG News Classification Dataset** (Kaggle) — [kaggle.com/datasets/amananandrai/ag-news-classification-dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

- 120,000 training articles / 7,600 test articles, perfectly balanced across 4 classes (30,000 / 1,900 per class respectively).
- Files: `train.csv` and `test.csv`, each with a header row and three columns: `Class Index`, `Title`, `Description`.
- Class labels: `1 = World`, `2 = Sports`, `3 = Business`, `4 = Sci/Tech`.
- This is the same corpus, and the same standard train/test split, referenced in the Zhang, Zhao & LeCun paper — verified independently in the notebook against the paper's published statistics.

**You must download the dataset yourself** (Kaggle's terms don't allow redistribution). Get it via:
- the Kaggle website (Download button on the dataset page), or
- the Kaggle API: `kaggle datasets download -d amananandrai/ag-news-classification-dataset`

Unzip it so that `train.csv` and `test.csv` sit in the same folder as the notebook.

---

## 3. Project Files

| File | Description |
|---|---|
| `News_Article_Topic_Classifier.ipynb` | Main deliverable — the full, executed Jupyter notebook (theory, EDA, preprocessing, TF-IDF, model training/evaluation/comparison, error analysis, tuning, and a live prediction function). |
| `News_Article_Topic_Classifier_Report.docx` | Full 21-section written academic report (Title → References), matching the notebook's structure and results. |
| `News_Article_Topic_Classifier_Proposal_Report.docx` | Shorter 5-section project proposal report (Title/Objective, Problem Statement, Literature Context, Dataset Details, Methodology). |
| `News_Article_Topic_Classifier.zip` | Overleaf/LaTeX version of the full report — `main.tex` + figures, ready to compile to PDF. |
| `News_Article_Topic_Classifier_main.tex` / `_compiled.pdf` | Standalone copies of the LaTeX source and its compiled PDF. |
| `best_model_logreg.joblib` | The trained, tuned Logistic Regression model (the best-performing classifier). |
| `tfidf_vectorizer.joblib` | The fitted TF-IDF vectorizer used to featurize text — required alongside the model for inference. |

---

## 4. How to Run the Notebook

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud joblib jupyter
   ```

2. **Download the NLTK data** the notebook needs (stop words, WordNet lemmatizer):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

3. **Place the dataset** — download `train.csv` and `test.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) and put both files in the same directory as the notebook.

4. **Run all cells** top to bottom:
   ```bash
   jupyter notebook News_Article_Topic_Classifier.ipynb
   ```
   or, to execute non-interactively end-to-end:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace News_Article_Topic_Classifier.ipynb
   ```

Total runtime is a few minutes on a standard laptop (most of it spent on TF-IDF vectorization and model training over 120,000 articles).

---

## 5. Using the Saved Model for Predictions

The saved model and vectorizer let you classify new articles without retraining:

```python
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("best_model_logreg.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("\\", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def predict_topic(article_text):
    features = vectorizer.transform([clean_text(article_text)])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = proba[model.classes_.tolist().index(pred)]
    return {"predicted_topic": LABEL_MAP[pred], "confidence": round(float(confidence), 4)}

print(predict_topic("Apple announced a new generation of processors for its laptop lineup."))
# -> {'predicted_topic': 'Sci/Tech', 'confidence': 0.87...}
```

---

## 6. Methodology Summary

```
Dataset (Kaggle AG News)
   -> Dataset Exploration (EDA)
   -> Data Cleaning (HTML/URL removal, lowercasing, punctuation stripping)
   -> Text Preprocessing (tokenization, stop-word removal, lemmatization)
   -> Train/Test Split (pre-defined, 120,000 / 7,600 — no re-shuffling)
   -> Feature Extraction (TF-IDF, unigrams + bigrams, 30,000 features)
   -> Model Training (Logistic Regression, Naive Bayes, Linear SVM)
   -> Model Evaluation (accuracy, precision, recall, F1 — macro & weighted)
   -> Hyperparameter Tuning (GridSearchCV on Logistic Regression's C)
   -> Model Comparison & Error Analysis
   -> Prediction System (predict_topic() function)
```

Full design rationale, theory, and discussion of each step are in the notebook and the accompanying report.

---

## 7. Key Findings

- Logistic Regression and Linear SVM perform almost identically (~92% accuracy/macro-F1) and both clearly outperform Multinomial Naive Bayes (~90%), consistent with Naive Bayes' feature-independence assumption being a poor fit for correlated unigram+bigram TF-IDF features.
- The dominant classification error is **Business ↔ Sci/Tech confusion** — real news about technology companies routinely blends financial and technical framing, so this reflects genuine topical overlap rather than a model weakness.
- Hyperparameter tuning confirmed that scikit-learn's default regularization strength (`C=1`) was already near-optimal for this feature space.
- Traditional TF-IDF + linear models were deliberately prioritized over deep learning: the reference paper's own results show traditional methods remain competitive at AG News's scale (hundreds of thousands of samples), with character-level ConvNets only pulling ahead at multi-million-sample scale.

---

## 8. Limitations & Future Work

- Articles are short (title + one-sentence description only), which caps how much context any model can use.
- Only 4 broad topic classes — no fine-grained subtopics.
- No deep-learning/transformer model (e.g., DistilBERT) was trained for direct comparison.
- Future extensions: word embeddings (Word2Vec/GloVe), transformer fine-tuning, full article bodies, live API deployment.

See Sections 18–19 of the full report for details.

---

## 9. References

1. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. [arXiv:1509.01626](https://arxiv.org/abs/1509.01626)
2. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. [scikit-learn.org](https://scikit-learn.org/stable/documentation.html)
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly Media. [nltk.org](https://www.nltk.org/)
5. Gulli, A. AG's Corpus of News Articles. [di.unipi.it/~gulli](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
6. amananandrai. AG News Classification Dataset. Kaggle. [kaggle.com/datasets/amananandrai/ag-news-classification-dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
