from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pytz
import language_tool_python
import textstat
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
db = SQLAlchemy(app)

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()
tool = language_tool_python.LanguageTool('en-US')
nlp = spacy.load('en_core_web_sm')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mentor')
def mentor():
    return render_template('mentor.html')

@app.route('/peer')
def peer():
    return render_template('peer.html')

@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    role = data['role']
    message = data['message']
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    timestamp_ist = utc_now.astimezone(ist)
    new_message = Chat(role=role, message=message, timestamp=timestamp_ist)
    db.session.add(new_message)
    db.session.commit()
    return jsonify({'status': 'Message received'})

@app.route('/get_messages', methods=['GET'])
def get_messages():
    messages = Chat.query.all()
    chat_log = []
    ist = pytz.timezone('Asia/Kolkata')
    for msg in messages:
        timestamp_ist = msg.timestamp.astimezone(ist)
        chat_log.append({
            'role': msg.role,
            'message': msg.message,
            'timestamp': timestamp_ist.strftime('%d/%m/%Y, %I:%M:%S %p')
        })
    return jsonify(chat_log)

@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    try:
        db.session.query(Chat).delete()
        db.session.commit()
        return jsonify({'status': 'Chat cleared'})
    except Exception as e:
        return jsonify({'status': 'Failed to clear chat', 'error': str(e)})

def analyze_sentiment(message):
    scores = sia.polarity_scores(message)
    label = 'Neutral'
    if scores['compound'] >= 0.05:
        label = 'Positive'
    elif scores['compound'] <= -0.05:
        label = 'Negative'
    return scores, label

def check_grammar(message):
    matches = tool.check(message)
    return len(matches), [match.message for match in matches]

def check_readability(message):
    flesch_score = textstat.flesch_reading_ease(message)
    return flesch_score

def text_analysis(message):
    doc = nlp(message)
    tokens = [token.text for token in doc if not token.is_stop]
    pos_tags = [(token.text, token.pos_) for token in doc if not token.is_stop]
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    return tokens, pos_tags, entities, dependencies

def topic_modeling(messages, model_type='lda', num_topics=5):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(messages)
    if model_type == 'lda':
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    else:
        model = NMF(n_components=num_topics, random_state=42)
    model.fit(X)
    topics = model.transform(X)
    return topics

def quality_and_coherence_analysis(message):
    doc = nlp(message)
    sentences = list(doc.sents)
    coherence_score = len(sentences) / len(doc) if len(doc) > 0 else 0
    return {'coherence_score': coherence_score}

def automated_feedback_analysis(message):
    doc = nlp(message)
    feedback = {}
    aspects = set(['service', 'price', 'quality', 'experience'])
    for aspect in aspects:
        if aspect in message.lower():
            feedback[aspect] = analyze_sentiment(aspect)
    return feedback

def calculate_jaccard_similarity(mentor_message, peer_message):
    mentor_tokens = set(word_tokenize(mentor_message.lower()))
    peer_tokens = set(word_tokenize(peer_message.lower()))
    intersection = mentor_tokens.intersection(peer_tokens)
    union = mentor_tokens.union(peer_tokens)
    return len(intersection) / len(union)

@app.route('/perform_evaluation', methods=['GET'])
def perform_evaluation():
    ten_minutes_ago = datetime.utcnow() - timedelta(minutes=10)
    messages = Chat.query.filter(Chat.timestamp >= ten_minutes_ago).all()
    evaluations = []
    messages_text = [msg.message for msg in messages]

    peer_messages = [msg.message for msg in messages if msg.role == 'peer']
    mentor_messages = [msg.message for msg in messages if msg.role == 'mentor']

    bleu_scores = []
    smoothing_function = SmoothingFunction().method1
    for peer_msg, mentor_msg in zip(peer_messages, mentor_messages):
        peer_tokens = [word for word in word_tokenize(peer_msg) if word.lower() not in stop_words]
        mentor_tokens = [word for word in word_tokenize(mentor_msg) if word.lower() not in stop_words]
        reference = [peer_tokens]
        hypothesis = mentor_tokens
        bleu_score = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5), smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    jaccard_similarities = []
    for peer_msg, mentor_msg in zip(peer_messages, mentor_messages):
        jaccard_similarity = calculate_jaccard_similarity(mentor_msg, peer_msg)
        jaccard_similarities.append(jaccard_similarity)
    
    avg_jaccard_similarity = sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0


    for msg in messages:
        scores, label = analyze_sentiment(msg.message)
        grammar_errors, grammar_details = check_grammar(msg.message)
        readability_score = check_readability(msg.message)
        tokens, pos_tags, entities, dependencies = text_analysis(msg.message)
        coherence_scores = quality_and_coherence_analysis(msg.message)
        feedback = automated_feedback_analysis(msg.message)

        evaluations.append({
            'role': msg.role,
            'message': msg.message,
            'sentiment': {
                'scores': scores,
                'label': label
            },
            'grammar': {
                'errors': grammar_errors,
                'details': grammar_details
            },
            'readability': {
                'score': readability_score
            },
            'text_analysis': {
                'tokens': tokens,
                'pos_tags': pos_tags,
                'entities': entities,
                'dependencies': dependencies
            },
            'coherence': coherence_scores,
            'feedback': feedback
        })

    topics = topic_modeling(messages_text)
    return jsonify({
        "evaluations": evaluations,
        "topics": topics.tolist(),
        "avg_bleu_score": avg_bleu_score,
        "avg_jaccard_similarity": avg_jaccard_similarity
    })

if __name__ == '__main__':
    app.run(debug=True)
