from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with hardcoded configuration
app = Flask(__name__)
app.secret_key = 'development_key_123'

# StockGeist API configuration
STOCKGEIST_API_KEY = 'EB577JAy0ombl1t2AvXjIow1GRrAJTIQ'
STOCKGEIST_BASE_URL = 'https://api.stockgeist.ai'

# Database functions remain the same...
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users
    (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)
    ''')
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

# Initialize database on startup
init_db()

# New function to get stock sentiment data using StockGeist API
def get_stock_sentiment(ticker):
    """
    Generate sample text about the stock and analyze its sentiment using the StockGeist API
    """
    # Generate some sample messages about the stock to analyze
    messages = [
        f"I think {ticker} is going to perform well this quarter",
        f"Investors are excited about {ticker}'s future prospects",
        f"There are some concerns about {ticker}'s recent performance",
        f"{ticker} has been volatile lately, causing some investor anxiety"
    ]
    
    # Endpoint for sentiment analysis
    endpoint = f"{STOCKGEIST_BASE_URL}/models/sentiment"
    
    # Headers with API token according to documentation
    headers = {
        'token': STOCKGEIST_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        # Call the API with our messages
        logger.info(f"Calling StockGeist API for ticker {ticker}")
        response = requests.post(endpoint, headers=headers, json=messages)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        logger.info(f"Response text: {response.text[:500]}...")
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                # Process the sentiment data
                sentiment_results = data['data']
                
                # Calculate average sentiment
                avg_positive = sum(item.get('positive_conf', 0) for item in sentiment_results) / len(sentiment_results)
                avg_negative = sum(item.get('negative_conf', 0) for item in sentiment_results) / len(sentiment_results)
                avg_neutral = sum(item.get('neutral_conf', 0) for item in sentiment_results) / len(sentiment_results)
                avg_emotionality = sum(item.get('emotionality_conf', 0) for item in sentiment_results) / len(sentiment_results)
                
                # Determine the dominant sentiment
                sentiments = {
                    'positive': avg_positive,
                    'neutral': avg_neutral,
                    'negative': avg_negative
                }
                dominant_sentiment = max(sentiments, key=sentiments.get)
                
                # Create a sentiment score (-1 to 1)
                sentiment_score = avg_positive - avg_negative
                
                return {
                    'success': True,
                    'sentiment': {
                        'score': round(sentiment_score, 2),
                        'dominant': dominant_sentiment,
                        'positive': round(avg_positive, 2),
                        'negative': round(avg_negative, 2),
                        'neutral': round(avg_neutral, 2),
                        'emotionality': round(avg_emotionality, 2),
                        'messages': messages  # Include the analyzed messages
                    }
                }
            else:
                logger.error(f"Unexpected API response format: {data}")
                return {
                    'success': False,
                    'error': 'Unexpected API response format'
                }
        else:
            logger.error(f"API request failed with status {response.status_code}")
            return {
                'success': False,
                'error': f"API request failed: {response.status_code} - {response.text}"
            }
    except Exception as e:
        logger.error(f"Exception in API request: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Fallback function to generate simulated sentiment data
def get_simulated_sentiment(ticker):
    """
    Generate simulated sentiment data for a given ticker
    """
    import random
    
    # Use ticker to seed the random generator for consistency
    random.seed(hash(ticker))
    
    # Generate random sentiment values
    positive = round(random.uniform(0.2, 0.8), 2)
    negative = round(random.uniform(0.0, 1.0 - positive), 2)
    neutral = round(1.0 - positive - negative, 2)
    emotionality = round(random.uniform(0.3, 0.9), 2)
    
    # Determine dominant sentiment
    sentiments = {
        'positive': positive,
        'neutral': neutral,
        'negative': negative
    }
    dominant_sentiment = max(sentiments, key=sentiments.get)
    
    # Calculate sentiment score (-1 to 1)
    sentiment_score = positive - negative
    
    # Generate some sample messages that might have been analyzed
    messages = [
        f"I think {ticker} is going to perform well this quarter",
        f"Investors are excited about {ticker}'s future prospects",
        f"There are some concerns about {ticker}'s recent performance",
        f"{ticker} has been volatile lately, causing some investor anxiety"
    ]
    
    return {
        'success': True,
        'sentiment': {
            'score': round(sentiment_score, 2),
            'dominant': dominant_sentiment,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'emotionality': emotionality,
            'messages': messages,
            'source': 'Simulated data (API connection failed)'
        }
    }

# Login decorator
def login_required(view):
    def wrapped_view(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    wrapped_view.__name__ = view.__name__
    return wrapped_view

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Username and password are required')
            return render_template('register.html')
        
        success = register_user(username, password)
        
        if success:
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        else:
            flash('Username already exists. Please choose another one.')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = get_user(username)
        
        if user and user['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('stocks'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/stocks')
@login_required
def stocks():
    return render_template('stocks.html')

@app.route('/stocks/<ticker>')
@login_required
def stock_detail(ticker):
    # Try to get sentiment data from the API
    sentiment_data = get_stock_sentiment(ticker.upper())
    
    # If the API call fails, fall back to simulated data
    if not sentiment_data['success']:
        logger.warning(f"Using simulated data for {ticker} due to API failure: {sentiment_data['error']}")
        sentiment_data = get_simulated_sentiment(ticker.upper())
    
    return render_template(
        'stock_detail.html', 
        ticker=ticker.upper(),
        sentiment=sentiment_data['sentiment']
    )

if __name__ == '__main__':
    app.run(debug=True)