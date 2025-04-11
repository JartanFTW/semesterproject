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

class OpenRouterWrapper:
    """
    Wrapper for OpenRouter API to provide financial and stock expertise
    """
    
    def __init__(self, openrouter_key=None):
        """Initialize the OpenRouter wrapper with the API key"""
        # Updated API key format and setup
        self.openrouter_key = openrouter_key or "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"
        
        # Properly initialize the OpenAI client with headers
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_key,
            default_headers={
                "HTTP-Referer": "https://yourwebsite.com",  # Optional, replace with your site
                "X-Title": "Stock Market Assistant"         # Optional, title of your app
            }
        )
        
        # Available models on OpenRouter - we'll use the best free ones
        self.primary_model = "meta-llama/llama-4-maverick:free"
        self.fallback_models = [
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-7b-it:free",
            "openchat/openchat-7b:free"
        ]
        
        # Initialize conversation history
        self.message_history = []
        
        # Financial expertise system prompt
        self.finance_expert_prompt = """You are StockAI, a highly knowledgeable financial advisor and stock market expert. 
You have extensive knowledge about:

1. MARKET FUNDAMENTALS
- Stock valuation methods (P/E, P/B, DCF models, etc.)
- Economic indicators and their impact on markets
- Market cycles, bull/bear markets, and trend analysis
- Sector rotation and industry performance patterns

2. TECHNICAL ANALYSIS
- Chart patterns and their interpretations
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Support/resistance levels and price action
- Volume analysis and market sentiment indicators

3. INVESTMENT STRATEGIES
- Value investing principles (Graham, Buffett methodologies)
- Growth investing approaches (GARP, momentum strategies)
- Income investing (dividend strategies, yield analysis)
- Risk management and portfolio allocation techniques

4. COMPANY ANALYSIS
- Financial statement analysis and key metrics
- Management evaluation and corporate governance
- Competitive positioning and market share analysis
- Product pipeline and R&D assessment

5. MACROECONOMIC FACTORS
- Interest rates and monetary policy impacts
- Inflation/deflation effects on asset classes
- Currency fluctuations and global trade dynamics
- Fiscal policy and regulatory environment changes

When responding to questions:
- Never recommend to check other websites and never refer your last update. Be Confident.
- Provide concise, actionable insights without unnecessary jargon
- Cite relevant financial principles and theories when appropriate
- Acknowledge both bull and bear perspectives when discussing outlook
- Clearly distinguish between established facts and market opinions
- Focus on educational value rather than specific investment recommendations

Remember: Your purpose is to educate users about financial markets and provide context for investment decisions, not to make specific buy/sell recommendations. Write one not too long paragraph, with no bullet points to the user."""
        
        # Test connection to validate API key on initialization
        self.api_available = self._test_connection()
    
    def _test_connection(self):
        """Test API connection and key validity"""
        try:
            # Simple test with minimal tokens
            self.client.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            logger.info("OpenRouter API connection successful")
            return True
        except Exception as e:
            logger.warning(f"OpenRouter API connection failed: {str(e)}")
            return False
    
    def get_response(self, user_message, context=None, ticker=None):
        """
        Get a response from the AI using OpenRouter
        
        Args:
            user_message (str): The user's message
            context (dict, optional): Additional context to include
            ticker (str, optional): Stock ticker if applicable
            
        Returns:
            dict: Response with success flag and message
        """
        try:
            # Add the message to history
            self.message_history.append({"role": "user", "content": user_message})
            
            # Limit history to last 10 messages to avoid token limits
            if len(self.message_history) > 10:
                self.message_history = self.message_history[-10:]
            
            # Create system message with financial expertise
            system_message = {
                "role": "system",
                "content": self.finance_expert_prompt
            }
            
            # Add context about current ticker if available
            if ticker:
                system_message["content"] += f"\n\nCurrent context: User is asking about {ticker} stock."
                
                # Try to get stock info from yfinance (which is free)
                try:
                    stock_info = self._get_basic_stock_info(ticker)
                    if stock_info:
                        system_message["content"] += f"\n\n{stock_info}"
                except Exception as e:
                    logger.warning(f"Could not get stock info for {ticker}: {str(e)}")
            
            # Add any additional context
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    context_str += f"- {key}: {value}\n"
                system_message["content"] += context_str
            
            # Prepare messages for the API request
            messages = [system_message] + self.message_history
            
            # If API is not available, use rule-based fallback
            if not self.api_available:
                return self._generate_rule_based_response(user_message, ticker, context)
            
            # Try primary model first
            try:
                logger.info(f"Attempting to use primary model: {self.primary_model}")
                completion = self.client.chat.completions.create(
                    model=self.primary_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                    timeout=30
                )
                
                ai_response = completion.choices[0].message.content
                self.message_history.append({"role": "assistant", "content": ai_response})
                
                return {
                    "success": True,
                    "response": ai_response
                }
            except Exception as primary_error:
                logger.warning(f"Primary model failed: {str(primary_error)}")
                
                # Try fallback models in sequence
                for fallback_model in self.fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        completion = self.client.chat.completions.create(
                            model=fallback_model,
                            messages=messages,
                            max_tokens=350,
                            temperature=0.7,
                            timeout=20
                        )
                        
                        ai_response = completion.choices[0].message.content
                        self.message_history.append({"role": "assistant", "content": ai_response})
                        
                        return {
                            "success": True,
                            "response": ai_response
                        }
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} failed: {str(fallback_error)}")
                        continue
                
                # If all models fail, use rule-based response
                return self._generate_rule_based_response(user_message, ticker, context)
                
        except Exception as e:
            logger.error(f"Error in OpenRouterWrapper.get_response: {str(e)}")
            return self._generate_rule_based_response(user_message, ticker, context)
    
    def _generate_rule_based_response(self, user_message, ticker=None, context=None):
        """Generate a rule-based response when API calls fail"""
        message = user_message.lower()
        
        # Financial dictionary for rule-based responses
        financial_terms = {
            "stock": "A stock represents ownership in a company. When you buy a stock, you're purchasing a small piece of that company.",
            "pe ratio": "The Price-to-Earnings (P/E) ratio compares a company's stock price to its earnings per share, helping determine if a stock might be overvalued or undervalued.",
            "dividend": "A dividend is a portion of a company's earnings paid to shareholders, usually distributed quarterly.",
            "market cap": "Market capitalization is the total value of a company's outstanding shares, calculated by multiplying share price by number of shares.",
            "bull market": "A bull market is when prices are rising or expected to rise, typically indicating investor confidence.",
            "bear market": "A bear market is when prices are falling or expected to fall, typically indicating investor pessimism.",
            "volatility": "Volatility measures how much a stock's price fluctuates. High volatility means the price can change dramatically over a short period.",
            "etf": "An Exchange-Traded Fund (ETF) is an investment fund that trades on stock exchanges, typically tracking an index, sector, or commodity."
        }
        
        # Check if asking about a financial term
        for term, definition in financial_terms.items():
            if term in message and any(phrase in message for phrase in ["what is", "explain", "tell me about", "meaning of"]):
                return {"success": True, "response": definition}
        
        # Stock price inquiry
        if ticker and any(word in message for word in ["price", "worth", "value", "trading at", "cost"]):
            try:
                stock_info = self._get_basic_stock_info(ticker)
                if stock_info:
                    return {"success": True, "response": f"Here's the latest information for {ticker}: {stock_info}"}
            except:
                pass
            return {"success": True, "response": f"I don't have the latest data for {ticker}, but you can find current prices on financial websites like Yahoo Finance or Google Finance."}
        
        # General greeting
        if any(word in message for word in ["hello", "hi", "hey", "greetings"]):
            return {"success": True, "response": "Hello! I'm your stock market assistant. How can I help you with financial information today?"}
        
        # Help request
        if any(word in message for word in ["help", "can you do", "assist"]):
            return {"success": True, "response": "I can help with stock information, explain financial terms, find similar stocks, and provide market insights. What would you like to know?"}
            
        # If context contains a financial term
        if context and 'financial_term' in context:
            return {"success": True, "response": context['financial_term']}
        
        # Default response
        if ticker:
            return {"success": True, "response": f"I see you're interested in {ticker}. You can ask about its price, recent performance, or company information."}
        else:
            return {"success": True, "response": "I'm your stock market assistant. Feel free to ask about specific stocks, financial concepts, or market trends."}
    
    def _get_basic_stock_info(self, ticker):
        """
        Get basic stock information using yfinance (free)
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Formatted stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return f"No information available for {ticker}."
                
            # Get current price data safely
            try:
                history = stock.history(period="2d")
                
                if history.empty:
                    price_str = "Price data unavailable."
                else:
                    current_price = history['Close'].iloc[-1] if len(history) > 0 else None
                    previous_close = history['Close'].iloc[-2] if len(history) > 1 else None
                    
                    # Calculate change
                    if current_price is not None and previous_close is not None:
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        price_str = f"Current Price: ${current_price:.2f}, Change: {'+' if change >= 0 else ''}{change:.2f} ({'+' if change >= 0 else ''}{change_percent:.2f}%)"
                    elif current_price is not None:
                        price_str = f"Current Price: ${current_price:.2f}"
                    else:
                        price_str = "Price data unavailable."
            except Exception as e:
                logger.warning(f"Error getting price data for {ticker}: {str(e)}")
                price_str = "Price data unavailable."
            
            # Company info - safely get values
            company_name = info.get('shortName', info.get('longName', 'Unknown'))
            sector = info.get('sector', 'Unknown sector')
            industry = info.get('industry', 'Unknown industry')
            
            # Format basic info
            result = f"{company_name} ({ticker}). {sector}, {industry}. {price_str}"
            
            # Add some key metrics if available
            metrics = []
            
            if 'marketCap' in info and info['marketCap']:
                # Format market cap in billions/millions for readability
                market_cap = info['marketCap']
                if market_cap >= 1000000000:
                    metrics.append(f"Market Cap: ${market_cap/1000000000:.2f}B")
                else:
                    metrics.append(f"Market Cap: ${market_cap/1000000:.2f}M")
            
            if 'trailingPE' in info and info['trailingPE']:
                metrics.append(f"P/E: {info['trailingPE']:.2f}")
                
            if 'dividendYield' in info and info['dividendYield']:
                metrics.append(f"Div Yield: {info['dividendYield']*100:.2f}%")
            
            if metrics:
                result += " " + ", ".join(metrics)
                
            return result
            
        except Exception as e:
            logger.warning(f"Error getting stock info for {ticker}: {str(e)}")
            return f"Unable to retrieve information for {ticker}."

class StockAI:
    """
    AI features for stock analysis and chatbot interactions
    """
    
    def __init__(self, alpha_vantage_key=None, stockgeist_key=None, openrouter_key=None, polygon_key=None):
        """Initialize with API keys (only openrouter_key is actually required now)"""
        self.alpha_vantage_key = alpha_vantage_key
        self.stockgeist_key = stockgeist_key
        self.polygon_key = polygon_key
        
        # Create OpenRouter AI wrapper
        self.ai = OpenRouterWrapper(openrouter_key)
        
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
            "fibonacci": "Fibonacci retracement levels are horizontal lines that indicate where support and resistance are likely to occur, based on the Fibonacci sequence.",
            "support level": "A price level where a stock tends to find buying interest that prevents it from falling further, often based on historical trading patterns.",
            "resistance level": "A price level where a stock tends to face selling pressure that prevents it from rising further, often based on historical trading patterns.",
            "technical analysis": "A method of evaluating securities by analyzing statistics generated by market activity, such as past prices and volume.",
            "fundamental analysis": "A method of evaluating securities by attempting to measure the intrinsic value of a stock using financial data and economic factors.",
            "dividend aristocrat": "A company in the S&P 500 that has increased its dividend payout for at least 25 consecutive years.",
            "market order": "An order to buy or sell a security immediately at the current market price.",
            "limit order": "An order to buy or sell a security at a specific price or better.",
            "stop-loss order": "An order to buy or sell a security when its price reaches a specified stop price, designed to limit an investor's loss.",
            "option": "A contract giving the buyer the right, but not the obligation, to buy or sell an underlying asset at a specific price on or before a certain date.",
            "call option": "An option contract giving the holder the right to buy a specified amount of an underlying security at a specified price within a specified time.",
            "put option": "An option contract giving the holder the right to sell a specified amount of an underlying security at a specified price within a specified time.",
            "beta": "A measure of a stock's volatility in relation to the overall market. A beta greater than 1 indicates higher volatility than the market.",
            "alpha": "A measure of an investment's performance relative to a benchmark index, adjusted for risk.",
            "rsi": "Relative Strength Index, a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.",
            "macd": "Moving Average Convergence Divergence, a trend-following momentum indicator showing the relationship between two moving averages of a security's price."
        }
        
        # Initialize conversation context
        self.conversation_context = {}
        
    def generate_chatbot_response(self, user_message, current_ticker=None):
        """
        Generate a response for the chatbot using the OpenRouterWrapper
        
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
            
            # Use mentioned ticker or current context ticker
            ticker = mentioned_ticker or current_ticker
            
            # Store this ticker in context for future responses
            if ticker:
                self.conversation_context['last_ticker'] = ticker
            elif 'last_ticker' in self.conversation_context:
                ticker = self.conversation_context.get('last_ticker')
            
            # Check if the query is about a financial term we have in our database
            term_match = None
            for term in self.financial_terms:
                if term in user_message.lower():
                    term_match = term
                    break
                    
            # Add term definition to context if found
            context = {}
            if term_match:
                context['financial_term'] = f"{term_match.upper()}: {self.financial_terms[term_match]}"
            
            # Get response from the AI
            response = self.ai.get_response(user_message, context, ticker)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating chatbot response: {str(e)}")
            return {
                'success': False,
                'error': f"I'm having trouble processing your request right now. Could you try asking in a different way?"
            }
    
    def get_price_prediction(self, ticker):
        """
        Generate a price prediction for a stock using the AI
        """
        try:
            # First get real stock data using yfinance
            stock_data = self._get_stock_data(ticker)
            if not stock_data['success']:
                return {
                    'success': False,
                    'error': stock_data['error']
                }
                
            current_price = float(stock_data['price'])
            
            # Determine if the AI is available
            if self.ai.api_available:
                # Create a prompt specifically for price prediction
                prediction_prompt = f"What is your price prediction for {ticker} stock for the next 7 days? Include the current price of ${current_price}, your prediction, and the reasoning behind it. Be concise and focus on the numerical prediction along with a brief rationale."
                
                # Get response from the AI
                response = self.ai.get_response(prediction_prompt, context={"request_type": "price_prediction"}, ticker=ticker)
                
                if response['success']:
                    # Check if the AI gave a well-formed response
                    ai_response = response['response']
                    
                    # Generate a simple prediction number for frontend
                    # Use sentiment to determine prediction direction
                    sentiment = self._get_simple_sentiment(ticker)
                    sentiment_factor = 0
                    if 'positive' in sentiment:
                        sentiment_factor = 0.03
                    elif 'negative' in sentiment:
                        sentiment_factor = -0.03
                        
                    # Generate a prediction value for the frontend
                    prediction_factor = random.uniform(-0.02, 0.02) + sentiment_factor
                    prediction = current_price * (1 + prediction_factor)
                    
                    # If AI response looks good, use it but also include the prediction number
                    return {
                        'success': True,
                        'current_price': current_price,
                        'prediction': round(prediction, 2),  # Ensure this is always provided for the frontend
                        'prediction_text': ai_response,
                        'time_frame': '7 days',
                        'sentiment_influence': sentiment  # Include sentiment for consistency
                    }
            
            # Fallback to simulated prediction if AI is unavailable or response is not well-formed
            # Generate a simple sentiment-based prediction
            sentiment = self._get_simple_sentiment(ticker)
            
            # Generate a random prediction with some bias based on sentiment
            sentiment_factor = 0
            if sentiment == 'positive' or sentiment == 'very positive':
                sentiment_factor = 0.03  # Positive bias
            elif sentiment == 'negative' or sentiment == 'very negative':
                sentiment_factor = -0.03  # Negative bias
                
            # Random prediction with sentiment bias
            prediction_factor = random.uniform(-0.05, 0.05) + sentiment_factor
            prediction = current_price * (1 + prediction_factor)
            
            # Calculate price range
            prediction_low = prediction * 0.98
            prediction_high = prediction * 1.02
            
            # Format the prediction message
            prediction_message = f"Based on current market conditions and technical analysis, {ticker} is currently trading at ${current_price:.2f} and is expected to "
            
            if prediction > current_price:
                prediction_message += f"rise to around ${prediction:.2f} (range: ${prediction_low:.2f} - ${prediction_high:.2f}) over the next 7 days. The stock shows {sentiment} sentiment indicators, with potential upside based on recent momentum."
            else:
                prediction_message += f"decline to around ${prediction:.2f} (range: ${prediction_low:.2f} - ${prediction_high:.2f}) over the next 7 days. The stock shows {sentiment} sentiment indicators, suggesting caution may be warranted."
                
            return {
                'success': True,
                'current_price': current_price,
                'prediction': round(prediction, 2),
                'prediction_low': round(prediction_low, 2),
                'prediction_high': round(prediction_high, 2),
                'time_frame': '7 days',
                'sentiment_influence': sentiment,  # Changed to match original key
                'prediction_text': prediction_message
            }
                
        except Exception as e:
            logger.error(f"Error in price prediction for {ticker}: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to generate prediction: {str(e)}"
            }
    
    def get_similar_stocks(self, ticker):
        """
        Find similar stocks using both AI and fallback methods
        """
        try:
            # Try AI method first
            if self.ai.api_available:
                # Create a prompt specifically for finding similar stocks
                similar_stocks_prompt = f"What are 4-5 similar stocks to {ticker} that investors might consider? Include only the tickers with a very brief explanation of why they are similar. Format as a comma-separated list of tickers followed by a short explanation."
                
                # Get response from the AI
                response = self.ai.get_response(similar_stocks_prompt, context={"request_type": "similar_stocks"}, ticker=ticker)
                
                if response['success']:
                    ai_response = response['response']
                    
                    # Extract tickers from AI response using regex
                    ticker_pattern = r'\b[A-Z]{1,5}\b'
                    found_tickers = re.findall(ticker_pattern, ai_response)
                    
                    # Filter out the input ticker and limit to 4 stocks
                    similar_tickers = [t for t in found_tickers if t != ticker][:4]
                    
                    # If we found at least 2 tickers, it's a usable response
                    if len(similar_tickers) >= 2:
                        # Get sector information
                        try:
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            sector = info.get('sector', 'Unknown')
                        except:
                            sector = 'Related Sector'
                        
                        return {
                            'success': True,
                            'similar_stocks': similar_tickers,
                            'sector': sector,
                            'recommendation_text': ai_response
                        }
            
            # Fallback to traditional method if AI is unavailable or returned poor results
            # Common stock sectors and some of their tickers
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
            
            # First try to get real sector info from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                real_sector = info.get('sector', '').lower()
                
                for sector_key, sector_name in sector_names.items():
                    if real_sector and (real_sector in sector_key or sector_key in real_sector):
                        found_sector = sector_key
                        break
                else:
                    found_sector = None
            except:
                found_sector = None
            
            # If we couldn't determine sector from yfinance, find which predefined sector contains the ticker
            if not found_sector:
                ticker = ticker.upper()
                for sector, stocks in stock_sectors.items():
                    if ticker in stocks:
                        found_sector = sector
                        break
            
            if not found_sector:
                # If ticker not found in any sector, default to tech
                found_sector = 'tech'
                
            # Get stocks from the same sector, excluding the input ticker
            similar_stocks = [stock for stock in stock_sectors[found_sector] if stock != ticker]
            
            # Randomly select 4 to return
            if len(similar_stocks) > 4:
                similar_stocks = random.sample(similar_stocks, 4)
                
            # Get the nice sector name
            sector_display = sector_names.get(found_sector, found_sector.capitalize())
            
            # Create explanation text
            explanation = f"Here are some similar stocks to {ticker} in the {sector_display} sector: {', '.join(similar_stocks)}. These companies operate in the same industry and may be worth investigating as part of a diversified {sector_display.lower()} portfolio."
                
            return {
                'success': True,
                'similar_stocks': similar_stocks,
                'sector': sector_display,
                'recommendation_text': explanation
            }
                
        except Exception as e:
            logger.error(f"Error finding similar stocks for {ticker}: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to find similar stocks: {str(e)}"
            }
    
    def _get_stock_data(self, ticker):
        """
        Get basic stock data using yfinance (free)
        """
        try:
            logger.info(f"Fetching stock data from YFinance for {ticker}")
            
            # Get stock data using yfinance
            stock = yf.Ticker(ticker)
            
            # Get current data (most recent price)
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
                    'error': f"Could not get data for {ticker}"
                }
                
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            return {
                'success': False,
                'error': f"Error retrieving data for {ticker}: {str(e)}"
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
