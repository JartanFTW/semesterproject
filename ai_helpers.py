"""
AI Helper functions for Stock Ticker Application

This module provides AI-powered features for stock analysis and chatbot functionality.
"""

import requests
import json
import logging
import re
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAI:
    """
    AI features for stock analysis and chatbot interactions
    """
    
    def __init__(self, alpha_vantage_key, stockgeist_key):
        """Initialize with API keys"""
        self.alpha_vantage_key = alpha_vantage_key
        self.stockgeist_key = stockgeist_key
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.stockgeist_base = "https://api.stockgeist.ai"
        
        # Financial terms dictionary for quick reference
        self.financial_terms = {
            "stock": "A stock represents ownership in a company. When you buy a stock, you're purchasing a small piece of that company.",
            "pe ratio": "The Price-to-Earnings (P/E) ratio is a valuation metric that compares a company's stock price to its earnings per share. It helps investors determine if a stock might be overvalued or undervalued.",
            "dividend": "A dividend is a portion of a company's earnings that is paid to shareholders. It's usually distributed in cash on a quarterly basis.",
            "market cap": "Market capitalization (market cap) is the total value of a company's outstanding shares of stock. It's calculated by multiplying the current share price by the number of shares outstanding.",
            "eps": "Earnings Per Share (EPS) is a company's profit divided by the outstanding shares of its common stock. It's an indicator of a company's profitability.",
            "bull market": "A bull market is when prices are rising or expected to rise in the stock market. It typically indicates investor confidence.",
            "bear market": "A bear market is when prices are falling or expected to fall in the stock market. It typically indicates investor pessimism.",
            "volatility": "Volatility is a measure of how much the price of a stock fluctuates. High volatility means the price can change dramatically over a short time period.",
            "dividend yield": "Dividend yield is a financial ratio that shows how much a company pays out in dividends each year relative to its stock price.",
            "etf": "An Exchange-Traded Fund (ETF) is a type of investment fund that trades on stock exchanges. ETFs typically track an index, sector, commodity, or other asset.",
            "moving average": "A moving average is a technical analysis indicator that helps smooth out price data by creating a constantly updated average price.",
            "day trading": "Day trading is the practice of buying and selling securities within the same trading day, hoping to profit from short-term price movements.",
            "blue chip": "Blue chip stocks are shares of large, well-established, and financially sound companies with a history of reliable performance.",
            "ipo": "An Initial Public Offering (IPO) is the process where a private company offers shares to the public in a new stock issuance.",
            "short selling": "Short selling is an investment strategy where an investor borrows a security and sells it on the open market, planning to buy it back later for less money.",
            "portfolio": "A portfolio is a collection of financial investments like stocks, bonds, cash, and other assets.",
            "hedge fund": "A hedge fund is an actively managed investment pool that uses various strategies to earn returns for its investors.",
            "ROI": "Return on Investment (ROI) is a performance measure used to evaluate the efficiency of an investment. It's calculated by dividing the benefit of an investment by its cost.",
            "index fund": "An index fund is a type of mutual fund or ETF that tracks a specific market index, like the S&P 500.",
            "leverage": "Leverage involves using borrowed capital to invest, with the expectation that the profits will be greater than the interest paid.",
            "fibonacci": "Fibonacci retracement levels are horizontal lines that indicate where support and resistance are likely to occur, based on the Fibonacci sequence."
        }
        
        # Initialize conversation context
        self.conversation_context = {}
        
    def get_price_prediction(self, ticker):
        """
        Generate a simple price prediction for a stock
        
        This is a placeholder for a real ML-based prediction model.
        In a production environment, this would use a trained model.
        """
        try:
            # Get current stock data
            stock_data = self._get_stock_data(ticker)
            
            if not stock_data['success']:
                return {
                    'success': False,
                    'error': stock_data['error']
                }
            
            # Generate a simple prediction
            # This is just for demonstration; a real system would use ML
            current_price = float(stock_data['price'])
            sentiment = self._get_simple_sentiment(ticker)
            
            # Generate a random prediction with some bias based on sentiment
            sentiment_factor = 0
            if sentiment == 'positive':
                sentiment_factor = 0.03  # Positive bias
            elif sentiment == 'negative':
                sentiment_factor = -0.03  # Negative bias
                
            # Random prediction with sentiment bias
            prediction_factor = random.uniform(-0.05, 0.05) + sentiment_factor
            prediction = current_price * (1 + prediction_factor)
            
            # Calculate confidence interval (again, simplified)
            confidence_low = prediction * 0.97
            confidence_high = prediction * 1.03
            
            return {
                'success': True,
                'current_price': current_price,
                'prediction': round(prediction, 2),
                'confidence_interval': {
                    'low': round(confidence_low, 2),
                    'high': round(confidence_high, 2)
                },
                'time_frame': '7 days',
                'sentiment_influence': sentiment
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction for {ticker}: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to generate prediction: {str(e)}"
            }
    
    def get_similar_stocks(self, ticker):
        """
        Find similar stocks based on industry and market cap
        
        This is a simplified implementation for demonstration purposes.
        A production system would use more sophisticated clustering or similarity metrics.
        """
        try:
            # Common stock sectors and some of their tickers
            # This is a simplified mapping
            stock_sectors = {
                'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'ADBE', 'CRM', 'INTC', 'CSCO'],
                'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'BLK'],
                'healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'BMY', 'ABT', 'TMO', 'LLY', 'AMGN'],
                'consumer': ['PG', 'KO', 'PEP', 'WMT', 'MCD', 'SBUX', 'NKE', 'DIS', 'HD', 'LOW'],
                'energy': ['XOM', 'CVX', 'BP', 'RDS.A', 'TOT', 'COP', 'SLB', 'EOG', 'MPC', 'PSX'],
                'industrial': ['GE', 'HON', 'MMM', 'CAT', 'DE', 'BA', 'LMT', 'RTX', 'UPS', 'FDX']
            }
            
            # Sector names for better display
            sector_names = {
                'tech': 'Technology',
                'finance': 'Financial Services',
                'healthcare': 'Healthcare',
                'consumer': 'Consumer Goods',
                'energy': 'Energy',
                'industrial': 'Industrial'
            }
            
            # Find which sector contains the ticker
            ticker = ticker.upper()
            found_sector = None
            for sector, stocks in stock_sectors.items():
                if ticker in stocks:
                    found_sector = sector
                    break
            
            if not found_sector:
                # If ticker not found, return tech stocks as default
                found_sector = 'tech'
                
            # Get stocks from the same sector, excluding the input ticker
            similar_stocks = [stock for stock in stock_sectors[found_sector] if stock != ticker]
            
            # Randomly select 4 to return
            if len(similar_stocks) > 4:
                similar_stocks = random.sample(similar_stocks, 4)
                
            return {
                'success': True,
                'similar_stocks': similar_stocks,
                'sector': sector_names.get(found_sector, found_sector.capitalize())
            }
            
        except Exception as e:
            logger.error(f"Error finding similar stocks for {ticker}: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to find similar stocks: {str(e)}"
            }
    
    def generate_chatbot_response(self, user_message, current_ticker=None):
        """
        Generate a response for the chatbot
        
        This is a rule-based implementation. In production, this would use 
        a more sophisticated NLP model like a fine-tuned LLM.
        
        Returns:
            dict: A dictionary with success flag and response message
        """
        try:
            # Convert to lowercase for easier matching
            message = user_message.lower().strip()
            
            # Extract ticker if mentioned - improved regex for ticker detection
            # This looks for 1-5 capital letters that could be a stock ticker
            ticker_match = re.search(r'\b([A-Za-z]{1,5})\b', user_message)
            mentioned_ticker = None
            if ticker_match:
                potential_ticker = ticker_match.group(0).upper()
                # Avoid common words being treated as tickers
                common_words = ['A', 'I', 'THE', 'FOR', 'AND', 'OR', 'IT', 'IS', 'BE', 'TO', 'IN', 'THAT', 'HAVE', 'WHAT', 'HOW', 'WHY', 'WHO', 'WHERE', 'WHEN']
                if potential_ticker not in common_words:
                    mentioned_ticker = potential_ticker
            
            # Check for ticker in context if none mentioned
            ticker = mentioned_ticker or current_ticker
            
            # Store this ticker in context for future responses
            if ticker:
                self.conversation_context['last_ticker'] = ticker
            elif 'last_ticker' in self.conversation_context:
                ticker = self.conversation_context.get('last_ticker')
            
            # Greetings
            if re.search(r'\b(hello|hi|hey|greetings|howdy)\b', message):
                response = "Hello! I'm your AI stock assistant. How can I help you with investments or financial information today?"
                
            # Help requests
            elif re.search(r'\b(help|guide|assist|support|what can you do)\b', message):
                response = """I can help with:
• Stock information and price data
• Price predictions and market sentiment
• Similar stocks to diversify your portfolio
• Explaining financial terms and concepts
• Basic investment advice

Try asking about a specific stock like "Tell me about AAPL" or explain concepts like "What is a P/E ratio?"
"""
            
            # Stock price requests
            elif re.search(r'\b(price|worth|value|trading at|how much is|cost|quote)\b', message) and ticker:
                stock_data = self._get_stock_data(ticker)
                if stock_data['success']:
                    change_direction = "up" if "+" in stock_data['change'] else "down"
                    response = f"{ticker} is currently trading at ${stock_data['price']}. The stock has moved {stock_data['change']} ({change_direction}) today."
                else:
                    response = f"I couldn't find price information for {ticker}. Are you sure that's a valid ticker symbol?"
            
            # Stock prediction requests
            elif re.search(r'\b(predict|prediction|forecast|outlook|future|where will|estimate|projection|will go)\b', message) and ticker:
                prediction = self.get_price_prediction(ticker)
                if prediction['success']:
                    direction = "up" if prediction['prediction'] > prediction['current_price'] else "down"
                    percent_change = abs(prediction['prediction'] - prediction['current_price']) / prediction['current_price'] * 100
                    response = (f"Based on my analysis, {ticker} could move {direction} by approximately {percent_change:.2f}% to around "
                            f"${prediction['prediction']} in the next {prediction['time_frame']}. "
                            f"The confidence interval is between ${prediction['confidence_interval']['low']} and ${prediction['confidence_interval']['high']}. "
                            f"Current sentiment is {prediction['sentiment_influence']}. Remember, this is just a simulation - not financial advice!")
                else:
                    response = f"I couldn't generate a prediction for {ticker} at this time."
            
            # Similar stocks
            elif re.search(r'\b(similar|like|related|alternatives|comparable|other companies like)\b', message) and ticker:
                similar = self.get_similar_stocks(ticker)
                if similar['success']:
                    stocks_list = ', '.join(similar['similar_stocks'])
                    response = f"If you're interested in {ticker}, you might also want to look at these stocks in the {similar['sector']} sector: {stocks_list}. These companies operate in similar markets."
                else:
                    response = f"I couldn't find similar stocks to {ticker} at this time."
            
            # Financial term explanation
            elif 'what is' in message or 'explain' in message or 'definition of' in message or 'meaning of' in message:
                # Check for any financial term in the message
                found_term = None
                for term in self.financial_terms.keys():
                    if term in message:
                        found_term = term
                        break
                
                if found_term:
                    response = self.financial_terms[found_term]
                elif ticker:
                    stock_data = self._get_stock_data(ticker)
                    if stock_data['success']:
                        response = f"{ticker} is a publicly traded company. It's currently priced at ${stock_data['price']}. You can ask me more specific questions about this stock."
                    else:
                        response = f"I'm not sure what you're asking about. If you're interested in learning about {ticker}, try asking about its price, predictions, or similar companies."
                else:
                    response = "I'm not sure what specific term you're asking about. I can explain concepts like P/E ratio, dividends, market cap, volatility, and many other financial terms. Just ask 'What is [term]?'"
            
            # Investment advice
            elif re.search(r'\b(invest|buy|sell|hold|good stock|recommendation|should I)\b', message):
                if ticker:
                    prediction = self.get_price_prediction(ticker)
                    if prediction['success']:
                        sentiment = prediction['sentiment_influence']
                        direction = "positive" if prediction['prediction'] > prediction['current_price'] else "negative"
                        response = f"Regarding {ticker}, current market sentiment is {sentiment} and my price prediction is {direction}. However, I can't provide personalized investment advice. Consider consulting a financial advisor and doing your own research before making investment decisions."
                    else:
                        response = "I can't provide personalized investment advice. Consider factors like your financial goals, risk tolerance, and time horizon. It's often wise to diversify your investments and consult with a financial advisor."
                else:
                    response = "I can provide information about stocks, but I can't offer personalized investment advice. A diversified portfolio that matches your risk tolerance and time horizon is generally recommended. Consider talking to a financial advisor for personalized guidance."
            
            # Market sentiment
            elif re.search(r'\b(sentiment|feeling|mood|outlook|bullish|bearish)\b', message):
                if ticker:
                    sentiment = self._get_simple_sentiment(ticker)
                    response = f"The current market sentiment for {ticker} appears to be {sentiment}. Remember that sentiment can change quickly based on news and market conditions."
                else:
                    sentiments = ["cautiously optimistic", "mixed", "generally positive", "somewhat uncertain", "trending positive"]
                    response = f"The overall market sentiment today seems {random.choice(sentiments)}. Would you like me to check the sentiment for a specific stock?"
            
            # News or events
            elif re.search(r'\b(news|events|headlines|announcement|report|earnings)\b', message):
                if ticker:
                    response = f"I don't have real-time news capabilities, but you can check recent news about {ticker} on financial news websites like Yahoo Finance, Bloomberg, or CNBC. These sources provide up-to-date information about company announcements and market events."
                else:
                    response = "I don't have real-time news capabilities. For the latest market news, check financial news websites like Yahoo Finance, Bloomberg, or CNBC."
            
            # Basic informational queries about a stock
            elif ticker and (re.search(r'\b(about|info|information|details|tell me about)\b', message) or message == ticker.lower()):
                sectors = {
                    'AAPL': 'Technology (Consumer Electronics)',
                    'MSFT': 'Technology (Software)',
                    'GOOGL': 'Technology (Internet Services)',
                    'META': 'Technology (Social Media)',
                    'AMZN': 'Consumer Cyclical (Online Retail)',
                    'TSLA': 'Automotive (Electric Vehicles)',
                    'NVDA': 'Technology (Semiconductors)',
                    'JPM': 'Financial Services (Banking)'
                }
                company_names = {
                    'AAPL': 'Apple Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'GOOGL': 'Alphabet Inc. (Google)',
                    'META': 'Meta Platforms, Inc. (formerly Facebook)',
                    'AMZN': 'Amazon.com Inc.',
                    'TSLA': 'Tesla, Inc.',
                    'NVDA': 'NVIDIA Corporation',
                    'JPM': 'JPMorgan Chase & Co.'
                }
                
                stock_data = self._get_stock_data(ticker)
                company_name = company_names.get(ticker, f"{ticker} Corporation")
                sector = sectors.get(ticker, "Various sectors")
                
                if stock_data['success']:
                    response = f"{company_name} ({ticker}) is a company operating in the {sector} sector. It's currently trading at ${stock_data['price']}. You can ask me about its price predictions, similar stocks, or other specific information."
                else:
                    response = f"I have limited information about {ticker}. It appears to be a publicly traded company, but I don't have current price data. You can still ask me to find similar stocks or make predictions based on historical patterns."
            
            # Thanks
            elif re.search(r'\b(thanks|thank you|appreciate|helpful|great|awesome)\b', message):
                response = "You're welcome! I'm happy to help with your financial and stock market questions. Is there anything else you'd like to know?"
                
            # Default response
            else:
                if ticker:
                    response = f"I see you mentioned {ticker}. What would you like to know? I can tell you about its current price, make a price prediction, find similar stocks, or provide other information about this company."
                else:
                    response = "I'm your stock market assistant. You can ask me about specific stocks (like AAPL or MSFT), get explanations of financial terms, find similar stocks to ones you're interested in, or get simple price predictions. How can I help you today?"
            
            return {
                'success': True,
                'response': response
            }
            
        except Exception as e:
            logger.error(f"Error generating chatbot response: {str(e)}")
            return {
                'success': False,
                'error': f"I'm having trouble processing your request right now. Could you try asking in a different way?"
            }
    
    def _get_stock_data(self, ticker):
        """
        Get basic stock data from Alpha Vantage
        """
        try:
            # In a real implementation, you would call the Alpha Vantage API
            # For this demo, we'll simulate a response
            
            # Map of some tickers to simulated prices
            stock_prices = {
                'AAPL': {'price': '175.34', 'change': '+0.67'},
                'MSFT': {'price': '342.88', 'change': '+1.23'},
                'GOOGL': {'price': '142.17', 'change': '-0.89'},
                'AMZN': {'price': '130.44', 'change': '+0.45'},
                'META': {'price': '312.22', 'change': '+1.78'},
                'TSLA': {'price': '245.65', 'change': '-2.34'},
                'NVDA': {'price': '445.78', 'change': '+3.21'},
                'JPM': {'price': '198.43', 'change': '-0.52'},
                'BAC': {'price': '39.27', 'change': '+0.81'},
                'WMT': {'price': '59.82', 'change': '+0.36'},
                'DIS': {'price': '105.38', 'change': '-1.12'},
                'KO': {'price': '61.74', 'change': '+0.28'},
                'PFE': {'price': '27.63', 'change': '-0.45'},
                'NKE': {'price': '93.76', 'change': '+1.24'},
                'INTC': {'price': '35.92', 'change': '-0.78'}
            }
            
            ticker = ticker.upper()
            
            # If we have the stock in our mock data
            if ticker in stock_prices:
                return {
                    'success': True,
                    'ticker': ticker,
                    'price': stock_prices[ticker]['price'],
                    'change': stock_prices[ticker]['change']
                }
            
            # Generate a random price for unknown tickers
            random_price = round(random.uniform(50, 500), 2)
            random_change = round(random.uniform(-3, 3), 2)
            change_str = f"+{random_change}" if random_change >= 0 else f"{random_change}"
            
            return {
                'success': True,
                'ticker': ticker,
                'price': str(random_price),
                'change': change_str
            }
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to get stock data: {str(e)}"
            }
    
    def _get_simple_sentiment(self, ticker):
        """
        Get a simple sentiment assessment for a stock
        
        This is a placeholder. In a real system, this would analyze recent news and social media.
        """
        # Some predefined sentiments for well-known stocks
        sentiment_map = {
            'AAPL': ['positive', 'very positive', 'neutral'],
            'MSFT': ['positive', 'very positive'],
            'GOOGL': ['positive', 'neutral'],
            'AMZN': ['positive', 'neutral'],
            'META': ['neutral', 'positive', 'negative'],
            'TSLA': ['volatile', 'very positive', 'very negative'],
            'NVDA': ['very positive', 'positive'],
            'JPM': ['positive', 'neutral'],
            'BAC': ['neutral', 'negative', 'positive'],
            'DIS': ['neutral', 'positive']
        }
        
        ticker = ticker.upper()
        
        # If we have predefined sentiment for this stock
        if ticker in sentiment_map:
            return random.choice(sentiment_map[ticker])
            
        # For demo purposes, return a weighted random sentiment for other stocks
        sentiments = ['very positive', 'positive', 'neutral', 'negative', 'very negative']
        weights = [0.1, 0.3, 0.4, 0.15, 0.05]  # Weighted towards neutral/positive
        return random.choices(sentiments, weights=weights, k=1)[0] 