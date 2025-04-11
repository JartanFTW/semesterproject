from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import requests
import json
import logging
import os
import time
import yfinance as yf
from ai_helpers import StockAI
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app with hardcoded configuration
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_replace_in_production')

# API configuration
STOCKGEIST_API_KEY = 'EB577JAy0ombl1t2AvXjIow1GRrAJTIQ'
STOCKGEIST_BASE_URL = 'https://api.stockgeist.ai'
ALPHAVANTAGE_API_KEY = 'AOML1FY1BFGG5JYC'
ALPHAVANTAGE_BASE_URL = 'https://www.alphavantage.co/query'
OPENROUTER_API_KEY = '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'
POLYGON_API_KEY = 'BpKCtdhu6oQsG2VqITeyipcCslWpnmTW'

# Initialize StockAI helper
stock_ai = StockAI(
    alpha_vantage_key=os.environ.get('ALPHA_VANTAGE_KEY', ALPHAVANTAGE_API_KEY),
    stockgeist_key=os.environ.get('STOCKGEIST_KEY', STOCKGEIST_API_KEY),
    openrouter_key=os.environ.get('OPENROUTER_API_KEY', OPENROUTER_API_KEY),
    polygon_key=os.environ.get('POLYGON_API_KEY', POLYGON_API_KEY)
)

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


init_db()

# Function to get company overview using YFinance
def get_company_overview(ticker):
    """
    Fetch company overview data using YFinance library
    """
    try:
        logger.info(f"Getting company overview from YFinance for {ticker}")
        # Get stock info using yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and len(info) > 0:
            # Map YFinance data to the same format we were using with AlphaVantage
            company_data = {
                'Symbol': info.get('symbol', ticker),
                'Name': info.get('shortName', 'N/A'),
                'Description': info.get('longBusinessSummary', 'No description available'),
                'Exchange': info.get('exchange', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'MarketCapitalization': info.get('marketCap', 'N/A'),
                'PERatio': info.get('trailingPE', 'N/A'),
                'PEGRatio': info.get('pegRatio', 'N/A'),
                'BookValue': info.get('bookValue', 'N/A'),
                'DividendYield': info.get('dividendYield', 'N/A'),
                'EPS': info.get('trailingEps', 'N/A'),
                'RevenuePerShareTTM': info.get('revenuePerShare', 'N/A'),
                'ProfitMargin': info.get('profitMargins', 'N/A'),
                'OperatingMarginTTM': info.get('operatingMargins', 'N/A'),
                'ReturnOnAssetsTTM': info.get('returnOnAssets', 'N/A'),
                'ReturnOnEquityTTM': info.get('returnOnEquity', 'N/A'),
                'RevenueTTM': info.get('totalRevenue', 'N/A'),
                'GrossProfitTTM': info.get('grossProfits', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                '52WeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52WeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
                '50DayMovingAverage': info.get('fiftyDayAverage', 'N/A'),
                '200DayMovingAverage': info.get('twoHundredDayAverage', 'N/A'),
                'AnalystTargetPrice': info.get('targetMeanPrice', 'N/A')
            }
            
            # Convert any numerical values to proper format
            for key, value in company_data.items():
                if value == 'N/A':
                    continue
                if isinstance(value, float):
                    # Format as percentage if appropriate
                    if 'Margin' in key or 'Yield' in key or 'Return' in key:
                        company_data[key] = f"{value * 100:.2f}%"
                    # Format large numbers
                    elif 'Market' in key or 'Revenue' in key or 'Profit' in key:
                        company_data[key] = f"{value:,.2f}"
                    else:
                        company_data[key] = f"{value:.2f}"
            
            return {
                'success': True,
                'company_data': company_data
            }
        else:
            logger.error(f"No data returned from YFinance for {ticker}")
            return {
                'success': False,
                'error': f"No data found for ticker {ticker}"
            }
    except Exception as e:
        logger.error(f"Exception in YFinance API request: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Function to get stock sentiment data using StockGeist API
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
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if username already exists
        existing_user = get_user(username)
        if existing_user:
            flash('Username already exists. Please choose another one.', 'danger')
            return render_template('register.html')
        
        # Register the user in the database
        if register_user(username, password):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Please try again.', 'danger')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # In a real app, you would verify credentials against a database
        # For demo purposes, accept any login
        session['user_id'] = username
        flash('Login successful!', 'success')
        return redirect(url_for('stocks'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/stocks')
@login_required
def stocks():
    return render_template('stocks.html')

@app.route('/stocks/<ticker>')
@login_required
def stock_detail(ticker):
    ticker = ticker.upper()
    
    # Get company overview data
    company_data_response = get_company_overview(ticker)
    
    # Try to get sentiment data from the API
    sentiment_data = get_stock_sentiment(ticker)
    
    # If the sentiment API call fails, fall back to simulated data
    if not sentiment_data['success']:
        logger.warning(f"Using simulated sentiment data for {ticker} due to API failure: {sentiment_data['error']}")
        sentiment_data = get_simulated_sentiment(ticker)
    
    return render_template(
        'stock_detail.html', 
        ticker=ticker,
        company_data=company_data_response.get('company_data', None) if company_data_response['success'] else None,
        company_error=None if company_data_response['success'] else company_data_response['error'],
        sentiment=sentiment_data['sentiment']
    )

# API Routes for chatbot and AI features
@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'No message provided'
        }), 400
    
    # Set a timeout for the response generation
    try:
        # Use a thread pool to run with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start the API call in a separate thread
            future = executor.submit(stock_ai.generate_chatbot_response, data['message'])
            try:
                # Wait for up to 15 seconds for a response
                response = future.result(timeout=15)
                return jsonify(response)
            except TimeoutError:
                # If it takes too long, return a timeout error
                logger.warning("Chatbot response generation timed out")
                return jsonify({
                    'success': True,
                    'response': "I'm taking too long to think about this. Could you ask a simpler question or try again later?"
                })
    except Exception as e:
        logger.error(f"Error in chatbot API: {str(e)}")
        return jsonify({
            'success': False,
            'error': "An error occurred while processing your request."
        }), 500

@app.route('/api/predict/<ticker>')
def predict_stock_api(ticker):
    prediction = stock_ai.get_price_prediction(ticker)
    return jsonify(prediction)

@app.route('/api/similar/<ticker>')
def similar_stocks_api(ticker):
    similar = stock_ai.get_similar_stocks(ticker)
    return jsonify(similar)

if __name__ == '__main__':
    app.run(debug=True)