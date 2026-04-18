# review_integrity.py

import re
import math
from collections import Counter
import nltk
nltk.download('vader_lexicon',              quiet=True)
nltk.download('stopwords',                  quiet=True)
nltk.download('punkt',                      quiet=True)  # sentence splitter
nltk.download('punkt_tab',                  quiet=True)
nltk.download('wordnet',                    quiet=True)  # lemmatizer dictionary
nltk.download('averaged_perceptron_tagger', quiet=True)  # POS tags for lemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

sia        = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


# ─── Sentiment helpers (unchanged) ────────────────────────────────────────────

def score_single_review(review_text: str) -> dict:
    return sia.polarity_scores(review_text)

def label_sentiment(compound_score: float) -> str:
    if compound_score >= 0.05:  return "Positive"
    if compound_score <= -0.05: return "Negative"
    return "Neutral"

def check_star_sentiment_agreement(star_rating: int, compound_score: float) -> bool:
    if star_rating >= 4: return compound_score >= 0.05
    if star_rating <= 2: return compound_score <= -0.05
    return True


# ─── Keyword config ────────────────────────────────────────────────────────────

PRODUCT_NOISE_WORDS = {
    "also", "like", "just", "really", "very", "good", "great", "nice", "love",
    "bought", "product", "item", "thing", "would", "could", "even", "much",
    "well", "still", "used", "using", "came", "come", "said", "says", "make",
    "made", "best", "ever", "back", "because", "dont", "didnt", "isnt", "wasnt",
    "this", "that", "with", "have", "been", "than", "them", "they", "from",
    "premier", "drink", "tried", "banana", "whilst", "available", "excellent",
    "getting", "going", "think", "know", "feel", "looks", "seems", "look",
    "give", "need", "want", "does", "work", "works", "worked", "will", "shall",
    "sure", "your", "their", "about", "there", "here", "when", "then",
    "these", "those", "some", "more", "less", "over", "same", "such",
}

PRODUCT_BOOST = {
    "quality", "durable", "material", "design", "finish", "texture", "weight",
    "size", "color", "colour", "thick", "thin", "soft", "hard", "sturdy",
    "cheap", "premium", "plastic", "metal", "screen", "battery", "charge",
    "charging", "cable", "case", "grip", "scratch", "clear", "protective",
    "fit", "install", "installation", "camera", "sound", "audio",
    "display", "bright", "accurate", "comfortable", "lightweight", "heavy",
}

# Meaningful two-word phrases — matched against raw lowercased text.
PRODUCT_BIGRAMS = {
    "battery life",  "build quality", "sound quality",  "image quality",
    "picture quality","print quality", "great quality",  "poor quality",
    "good quality",  "high quality",  "low quality",
    "fast shipping", "great value",   "good value",      "not worth",
    "easy install",  "easy setup",    "easy use",
    "highly recommend", "would recommend", "dont recommend",
    "well made",     "cheaply made",  "poorly made",
    "fits perfectly","stopped working","broke after",    "cracked after",
    "customer service","return policy",
    "not good",      "not great",     "not bad",
    "works great",   "works perfectly","works well",
    "falls apart",   "holds up",      "peeling off",
}

# Words that negate the following word — these form synthetic negative bigrams.
NEGATION_WORDS = {
    "not", "no", "never", "cant", "cannot", "wont", "dont",
    "doesnt", "didnt", "isnt", "wasnt", "barely", "hardly",
    "scarcely", "nothing", "neither",
}

MIN_DOC_FREQ = 2   # skip terms that only appear in a single review


# ─── Extraction helpers ────────────────────────────────────────────────────────

def _lemma(word: str) -> str:
    """Return the noun-lemma root (battery/batteries → battery, fits/fitting → fit)."""
    return lemmatizer.lemmatize(word.lower())


def _sentence_scores_for_term(term: str, reviews: list, field: str) -> list[float]:
    """
    Score only the sentences containing `term`, not the whole review.

    Why: "Love the build quality. Battery is terrible." scores ~neutral overall.
    Scoring the sentence "Battery is terrible." in isolation correctly returns
    a negative compound, so "battery" gets the right sentiment label.
    """
    scores = []
    pat = re.compile(re.escape(term), re.IGNORECASE)
    for review in reviews:
        text = review.get(field, "") or ""
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = text.split(".")
        for sent in sentences:
            if pat.search(sent):
                scores.append(sia.polarity_scores(sent)["compound"])
    return scores


def _negation_bigrams(text: str) -> list[str]:
    """
    Find "not working", "never fits", "doesn't charge" patterns.
    Returns 'not <word>' strings — injected as forced-negative keywords.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    pairs  = []
    for i, tok in enumerate(tokens[:-1]):
        if tok.replace("'", "") in NEGATION_WORDS:
            nxt = tokens[i + 1]
            if (len(nxt) >= 3
                    and nxt not in STOP_WORDS
                    and nxt not in PRODUCT_NOISE_WORDS):
                pairs.append(f"not {_lemma(nxt)}")
    return pairs


# ─── Main keyword extraction ───────────────────────────────────────────────────

def extract_common_keywords(reviews: list, top_n: int = 10) -> list:
    """
    Extracts the most meaningful product keywords from a set of reviews.

    Improvements over the original:
    1. Lemmatization      — "batteries"/"battery" count as one term; surface-
                            form splits no longer dilute a keyword's score.
    2. Negation detection — "not working", "never fits" are caught as explicit
                            negative signals rather than being discarded.
    3. Sentence-level     — each keyword's sentiment is scored on the specific
       sentiment            sentence(s) containing it, not the full review.
    4. IDF weighting      — terms that appear in almost every review (noise)
                            are down-ranked vs. distinctive praise/problem words.
    5. Min doc-frequency  — terms seen in only one review are filtered out;
                            they're likely product-specific one-offs, not patterns.
    6. Curated bigrams    — two-word phrases like "battery life" or "stopped
                            working" surface alongside single words.
    """
    total = len(reviews)
    if total == 0:
        return []

    word_counts:     Counter = Counter()
    word_doc_freq:   Counter = Counter()
    bigram_counts:   Counter = Counter()
    bigram_doc_freq: Counter = Counter()
    negation_counts: Counter = Counter()

    for review in reviews:
        body = review.get("body", "") or ""
        if not body:
            continue
        text_lower = body.lower()

        # ── lemmatized unigrams ────────────────────────────────────────────
        raw_words   = re.findall(r"[a-z]{4,}", text_lower)
        seen_lemmas: set[str] = set()
        for w in raw_words:
            lemma = _lemma(w)
            if lemma not in STOP_WORDS and lemma not in PRODUCT_NOISE_WORDS:
                word_counts[lemma] += 1
                if lemma not in seen_lemmas:
                    word_doc_freq[lemma] += 1
                    seen_lemmas.add(lemma)

        # ── curated bigrams ────────────────────────────────────────────────
        seen_bg: set[str] = set()
        for bg in PRODUCT_BIGRAMS:
            if bg in text_lower:
                bigram_counts[bg] += 1
                if bg not in seen_bg:
                    bigram_doc_freq[bg] += 1
                    seen_bg.add(bg)

        # ── negation bigrams ───────────────────────────────────────────────
        for neg in _negation_bigrams(body):
            negation_counts[neg] += 1

    # ── TF-IDF-style scoring ──────────────────────────────────────────────
    def idf(df: int) -> float:
        return math.log(total / (1 + df)) + 1.0

    scored: dict[str, float] = {}

    for lemma, count in word_counts.items():
        df = word_doc_freq[lemma]
        if df < MIN_DOC_FREQ:
            continue
        boost = 2.0 if lemma in PRODUCT_BOOST else 1.0
        scored[lemma] = count * idf(df) * boost

    for bg, count in bigram_counts.items():
        df = bigram_doc_freq[bg]
        if df < MIN_DOC_FREQ:
            continue
        scored[bg] = count * idf(df) * 3.0   # bigrams: 3× weight (more specific)

    for neg, count in negation_counts.items():
        if count >= MIN_DOC_FREQ:
            scored[neg] = count * 4.0         # negations: 4× weight (high signal)

    top_terms = sorted(scored, key=scored.__getitem__, reverse=True)[:top_n]

    # ── sentence-level sentiment for each top term ────────────────────────
    keywords = []
    for term in top_terms:
        is_negation = term.startswith("not ") and " " in term
        is_bigram   = " " in term

        if is_bigram:
            raw_count = negation_counts[term] if is_negation else bigram_counts[term]
        else:
            raw_count = word_counts[term]

        if is_negation:
            sentiment = "negative"   # negation phrases are always negative by construction
        else:
            sscores = _sentence_scores_for_term(term, reviews, field="body")
            avg     = sum(sscores) / len(sscores) if sscores else 0.0
            sentiment = "positive" if avg >= 0.05 else "negative" if avg <= -0.05 else "neutral"

        keywords.append({"word": term, "count": raw_count, "sentiment": sentiment})

    return keywords


# ─── Review integrity analysis (logic unchanged) ──────────────────────────────

def analyze_review_integrity(reviews: list) -> dict:
    if not reviews:
        return {"error": "No reviews found for this product."}

    review_details  = []
    compound_scores = []
    verified_count  = 0
    agreement_count = 0
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for review in reviews:
        if not isinstance(review, dict):
            continue
        body        = review.get("body", "")
        star_rating = review.get("rating", 3)
        is_verified = review.get("verifiedPurchase", False)
        if not body:
            continue

        vader_scores = score_single_review(body)
        compound     = vader_scores["compound"]
        label        = label_sentiment(compound)
        agrees       = check_star_sentiment_agreement(star_rating, compound)

        compound_scores.append(compound)
        sentiment_counts[label] += 1
        if is_verified:  verified_count  += 1
        if agrees:       agreement_count += 1

        review_details.append({
            "title":           review.get("title", ""),
            "rating":          star_rating,
            "verified":        is_verified,
            "compound_score":  round(compound, 3),
            "sentiment_label": label,
            "star_text_agree": agrees,
        })

    total = len(review_details)
    if total == 0:
        return {"error": "All reviews lacked text content."}

    verified_ratio    = verified_count  / total
    consistency_ratio = agreement_count / total
    avg_compound      = sum(compound_scores) / total

    raw_integrity       = (verified_ratio * 0.60) + (consistency_ratio * 0.40)
    integrity_score_pct = round(raw_integrity * 100)

    if integrity_score_pct >= 80:
        integrity_label = "Most reviews appear organic and verified."
    elif integrity_score_pct >= 60:
        integrity_label = "Some reviews may be unverified — read carefully."
    else:
        integrity_label = "Low review integrity — treat ratings with caution."

    flags = {}
    if verified_ratio < 0.50:
        flags["low_verified_ratio"] = True
    if consistency_ratio < 0.65:
        flags["star_text_mismatch"] = True
    if avg_compound < -0.1 and sum(r["rating"] for r in review_details) / total > 3.5:
        flags["inflated_ratings"] = True

    return {
        "integrity_score_pct":         integrity_score_pct,
        "integrity_label":             integrity_label,
        "verified_purchase_ratio":     round(verified_ratio, 2),
        "sentiment_consistency_ratio": round(consistency_ratio, 2),
        "avg_compound_score":          round(avg_compound, 3),
        "sentiment_breakdown":         sentiment_counts,
        "review_details":              review_details,
        "flags":                       flags,
        "commonKeywords":              extract_common_keywords(reviews),
    }