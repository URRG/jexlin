from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from gensim import corpora
from gensim.models import CoherenceModel
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import jaccard_score

nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

sid = SentimentIntensityAnalyzer()
model = SentenceTransformer('all-MiniLM-L6-v2')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)
    message = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@app.route('/evaluate', methods=['GET'])
def evaluate():
    ten_minutes_ago = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
    messages = Message.query.filter(Message.timestamp >= ten_minutes_ago).all()
    
    peer_messages = [msg.message for msg in messages if msg.role == 'peer']
    mentor_messages = [msg.message for msg in messages if msg.role == 'mentor']
    
    # Sentiment Analysis
    peer_sentiments = [sid.polarity_scores(msg)['compound'] for msg in peer_messages]
    mentor_sentiments = [sid.polarity_scores(msg)['compound'] for msg in mentor_messages]
    avg_peer_sentiment = sum(peer_sentiments) / len(peer_sentiments) if peer_sentiments else 0
    avg_mentor_sentiment = sum(mentor_sentiments) / len(mentor_sentiments) if mentor_sentiments else 0
    
    # Topic Modeling
    texts = [msg.split() for msg in (peer_messages + mentor_messages)]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    coherence_model = CoherenceModel(corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence = coherence_model.get_coherence()

    # Coherence Analysis
    peer_embeddings = model.encode(peer_messages, convert_to_tensor=True)
    mentor_embeddings = model.encode(mentor_messages, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(peer_embeddings.mean(dim=0), mentor_embeddings.mean(dim=0)).item()

    # BLEU Score Calculation
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []
    for peer_msg, mentor_msg in zip(peer_messages, mentor_messages):
        reference = [peer_msg.split()]
        hypothesis = mentor_msg.split()
        bleu_score = sentence_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score)
    
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    jaccard_similarities = []
    for peer_msg, mentor_msg in zip(peer_messages, mentor_messages):
        peer_tokens = set(word_tokenize(peer_msg.lower()))
        mentor_tokens = set(word_tokenize(mentor_msg.lower()))
        intersection = peer_tokens.intersection(mentor_tokens)
        union = peer_tokens.union(mentor_tokens)
        jaccard_similarity = len(intersection) / len(union) if union else 0
        jaccard_similarities.append(jaccard_similarity)
    
    avg_jaccard_similarity = sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0
    
    return jsonify({
        'avg_peer_sentiment': avg_peer_sentiment,
        'avg_mentor_sentiment': avg_mentor_sentiment,
        'coherence': coherence,
        'similarity': similarity,
        'avg_bleu_score': avg_bleu_score,
        'avg_jaccard_similarity': avg_jaccard_similarity
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
