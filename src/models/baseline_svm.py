"""TF-IDF vectorizer + Linear SVM benchmark classifier."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data.ingest import generate_review_corpus

def train_svm_baseline():
    df = generate_review_corpus(1200)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ("svm", LinearSVC(C=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("SVM Baseline Results:")
    print(classification_report(y_test, preds))
    return pipe

if __name__ == "__main__":
    train_svm_baseline()
