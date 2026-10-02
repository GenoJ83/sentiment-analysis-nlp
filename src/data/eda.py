"""Text length distribution and vocabulary analysis."""
import collections
from src.data.ingest import generate_review_corpus

def analyze_corpus_statistics():
    df = generate_review_corpus(1000)
    word_counts = [len(text.split()) for text in df["text"]]
    vocab = collections.Counter()
    for text in df["text"]:
        vocab.update(text.lower().split())

    stats = {
        "total_reviews": len(df),
        "mean_word_count": sum(word_counts) / len(word_counts),
        "vocab_size": len(vocab),
        "top_keywords": vocab.most_common(10)
    }
    print("Corpus Vocabulary Statistics:")
    print(stats)
    return stats

if __name__ == "__main__":
    analyze_corpus_statistics()
