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
from openai import OpenAI
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAI:
    """
    AI features for stock analysis and chatbot interactions
    """
    
    def __init__(self, alpha_vantage_key, stockgeist_key, openrouter_key=None, polygon_key=None):
        """Initialize with API keys"""
        self.alpha_vantage_key = alpha_vantage_key
        self.stockgeist_key = stockgeist_key
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.stockgeist_base = "https://api.stockgeist.ai"
        self.polygon_key = polygon_key or "BpKCtdhu6oQsG2VqITeyipcCslWpnmTW"
        self.polygon_base = "https://api.polygon.io"
        
        # OpenRouter setup
        self.openrouter_key = openrouter_key or "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"
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_key,
        )
        # Using Mistral-7B-Instruct which is available on OpenRouter
        self.ai_model = "mistralai/mistral-7b-instruct:free"
        
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
        # Track message history for the AI
        self.message_history = []
        
    def generate_chatbot_response(self, user_message, current_ticker=None):
        """
        Generate a response for the chatbot using OpenRouter API with Mistral-7B-Instruct model
        
        This version uses the Mistral-7B-Instruct model via OpenRouter with the OpenAI client
        
        Returns:
            dict: A dictionary with success flag and response message
        """
        try:
            # Extract ticker if mentioned
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
                
            # Add user message to history
            self.message_history.append({"role": "user", "content": user_message})
            
            # Keep only the last 10 messages to avoid token limits
            if len(self.message_history) > 10:
                self.message_history = self.message_history[-10:]
            
            # Add system message for context
            system_message = {
                "role": "system", 
                "content": """You are StockAI, a concise stock market assistant. Keep responses brief and direct. 
                Current stock context: {ticker_info}
                Rules:
                - Be direct and concise
                - Keep answers to 1-2 sentences max
                - Focus on key information
                - Reference current stock when relevant
                - No bullet points or lists
                - Avoid unnecessary explanations""".format(
                    ticker_info=f"You are currently viewing {ticker} stock information." if ticker else "No specific stock is currently selected."
                )
            }
            
            # Add stock information if a ticker is mentioned
            if ticker:
                stock_data = self._get_stock_data(ticker)
                if stock_data['success']:
                    system_message["content"] += f"\nCurrent {ticker} data: Price ${stock_data['price']}, Change {stock_data['change']}"
            
            # Prepare messages for API request
            messages = [system_message] + self.message_history
            
            try:
                logger.info(f"Sending request to OpenRouter API with model: {self.ai_model}")
                logger.info(f"Messages being sent: {json.dumps(messages, indent=2)}")
                
                # Call OpenRouter API with Mistral-7B-Instruct model
                completion = self.openai_client.chat.completions.create(
                    model=self.ai_model,
                    messages=messages,
                    max_tokens=100,  # Reduced further to encourage very short responses
                    temperature=0.7,
                    timeout=15
                )
                
                logger.info(f"Received response from OpenRouter API: {completion}")
                
                # Check for provider errors
                if hasattr(completion, 'error') and completion.error:
                    logger.error(f"Provider error: {completion.error}")
                    raise Exception(f"Provider error: {completion.error.get('message', 'Unknown error')}")
                
                # Extract response
                if completion and hasattr(completion, 'choices') and completion.choices:
                    ai_response = completion.choices[0].message.content
                    logger.info(f"Successfully extracted AI response: {ai_response}")
                    
                    # Add AI response to history
                    self.message_history.append({"role": "assistant", "content": ai_response})
                    
                    return {
                        'success': True,
                        'response': ai_response
                    }
                else:
                    logger.error(f"Invalid response format from OpenRouter API. Response: {completion}")
                    raise Exception("Invalid response format from OpenRouter API")
                
            except Exception as api_error:
                logger.error(f"OpenRouter API error: {str(api_error)}")
                logger.error(f"Error type: {type(api_error)}")
                logger.error(f"Error details: {api_error.__dict__ if hasattr(api_error, '__dict__') else 'No details available'}")
                
                # Fallback to rule-based response
                fallback_response = self._generate_fallback_response(user_message, ticker)
                
                return {
                    'success': True,
                    'response': fallback_response
                }
            
        except Exception as e:
            logger.error(f"Error generating chatbot response: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            return {
                'success': False,
                'error': f"I'm having trouble processing your request right now. Could you try asking in a different way?"
            }
    
    def _generate_fallback_response(self, user_message, ticker=None):
        """
        Generate a fallback response when the API fails
        Uses the old rule-based implementation for fallback
        """
        message = user_message.lower().strip()
        
        # Greetings
        if re.search(r'\b(hello|hi|hey|greetings|howdy)\b', message):
            return "Hello! I'm your AI stock assistant. How can I help you with investments or financial information today?"
            
        # Help requests
        elif re.search(r'\b(help|guide|assist|support|what can you do)\b', message):
            return """I can help with:
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
                return f"{ticker} is currently trading at ${stock_data['price']}. The stock has moved {stock_data['change']} ({change_direction}) today."
            else:
                return f"I couldn't find price information for {ticker}. Are you sure that's a valid ticker symbol?"
        
        # Financial term explanation
        elif 'what is' in message or 'explain' in message or 'definition of' in message or 'meaning of' in message:
            # Check for any financial term in the message
            found_term = None
            for term in self.financial_terms.keys():
                if term in message:
                    found_term = term
                    break
            
            if found_term:
                return self.financial_terms[found_term]
        
        # Default response
        if ticker:
            return f"I see you mentioned {ticker}. What would you like to know? I can tell you about its current price, make predictions, find similar stocks, or provide other information about this company."
        else:
            return "I'm your stock market assistant. You can ask me about specific stocks (like AAPL or MSFT), get explanations of financial terms, find similar stocks to ones you're interested in, or get price predictions. How can I help you today?"
    
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
            
            return {
                'success': True,
                'current_price': current_price,
                'prediction': round(prediction, 2),
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
    
    def _get_stock_data(self, ticker):
        """
        Get basic stock data using the yfinance library (completely free, no API key needed)
        """
        try:
            logger.info(f"Fetching stock data from YFinance for {ticker}")
            
            # Get stock data using yfinance
            stock = yf.Ticker(ticker)
            
            # Get current data (most recent price)
            try:
                current_data = stock.history(period="1d")
                
                if not current_data.empty:
                    # Get most recent price (last row, Close column)
                    current_price = current_data['Close'].iloc[-1]
                    
                    # Get previous close for calculating change
                    previous_data = stock.history(period="2d")
                    if len(previous_data) > 1:
                        previous_close = previous_data['Close'].iloc[-2]
                        change = current_price - previous_close
                        change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
                    else:
                        change_str = "0.00"
                    
                    return {
                        'success': True,
                        'ticker': ticker,
                        'price': str(round(current_price, 2)),
                        'change': change_str
                    }
                else:
                    logger.warning(f"No data returned from YFinance for {ticker}")
                    return {
                        'success': False,
                        'error': f"Could not get data for {ticker} from YFinance"
                    }
            except Exception as e:
                logger.error(f"Error processing YFinance data for {ticker}: {str(e)}")
                return {
                    'success': False,
                    'error': f"Error processing data for {ticker}: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error getting stock data from YFinance for {ticker}: {str(e)}")
            
            # Try Alpha Vantage as fallback (with limited daily requests)
            logger.info(f"Trying Alpha Vantage as fallback for {ticker}")
            return self._get_alpha_vantage_data(ticker)
    
    def _get_alpha_vantage_data(self, ticker):
        """Get stock data from Alpha Vantage as a fallback method"""
        try:
            url = f"{self.alpha_vantage_base}?function=GLOBAL_QUOTE&symbol={ticker}&apikey={self.alpha_vantage_key}"
            logger.info(f"Fetching stock data from Alpha Vantage for {ticker}")
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                price = quote.get("05. price", "N/A")
                change = quote.get("09. change", "N/A")
                
                return {
                    'success': True,
                    'ticker': ticker,
                    'price': price,
                    'change': change
                }
            else:
                logger.warning(f"No data returned from Alpha Vantage for {ticker}: {data}")
                # Use simulated data as ultimate fallback
                return self._get_simulated_data(ticker)
                
        except Exception as e:
            logger.error(f"Error getting stock data from Alpha Vantage for {ticker}: {str(e)}")
            # Use simulated data as ultimate fallback
            return self._get_simulated_data(ticker)
            
    def _get_simulated_data(self, ticker):
        """Generate simulated data as final fallback method"""
        logger.warning(f"Using simulated data for {ticker} as fallback")
        
        # Use ticker to seed random generator for consistency
        random.seed(hash(ticker))
        
        # Generate random price based on ticker length (just for variety)
        base_price = 50 + (len(ticker) * 10)
        random_price = round(random.uniform(base_price * 0.8, base_price * 1.2), 2)
        
        # Generate random change
        random_change = round(random.uniform(-5, 5), 2)
        change_str = f"+{random_change}" if random_change >= 0 else f"{random_change}"
        
        return {
            'success': True,
            'ticker': ticker,
            'price': str(random_price),
            'change': change_str,
            'source': 'Simulated data (API connections failed)'
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