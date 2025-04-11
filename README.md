# Stock Ticker Application

> **Project Overview**: A minimalist web application that allows users to register, login, search for stock tickers, and view comprehensive financial data alongside social sentiment analysis.

This application combines quantitative financial metrics with qualitative sentiment indicators to provide a holistic view of stock market assets. The dual-perspective approach integrates official company data with social sentiment analysis to offer insights beyond traditional financial indicators.

## Features

- **User Authentication**: Secure registration and login system
- **Stock Search**: Simple interface to look up stocks by ticker symbol
- **Search History**: Persistent record of recently viewed stocks
- **Financial Data**: Comprehensive company metrics via AlphaVantage API
- **Sentiment Analysis**: Social media sentiment indicators via StockGeist API
- **Chatbot**: AI Assistant that assists the user in Stocks and Finance Questions

## Setup Process

### Prerequisites

- Python 3.7+ installed on your system
- Internet connection for API access

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd CIS4004-project
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   *On Windows:*
   ```bash
   venv\Scripts\activate
   ```

   *On macOS/Linux:*
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure API keys**

   The application comes pre-configured with test API keys. For production use, replace these in `app.py`:
   
   ```python
   STOCKGEIST_API_KEY = 'your_stockgeist_api_key'
   ALPHAVANTAGE_API_KEY = 'your_alphavantage_api_key'
   OPENROUTER_API_KEY = 'your_openrouter_api_key'
   ```

6. **Run the application**

   ```bash
   python app.py
   ```

7. **Access the application**

   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Register a new account or login with existing credentials
2. Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA) in the search box
3. View financial data and sentiment analysis for the selected stock
4. Use the "Back to Search" button to look up additional stocks

## Technical Architecture

The application follows a **layered architecture** with clear separation of concerns:

- **Frontend**: HTML templates with CSS styling for responsive user interface
- **Backend**: Flask server handling authentication and API integration
- **Data Layer**: External APIs for financial and sentiment data
- **Persistence**: SQLite database for user management and local storage for search history

## Limitations

- **Security**: This MVP does not implement password hashing or other security best practices
- **API Rate Limits**: Free API keys have usage limitations (5 requests/minute for AlphaVantage)
- **Error Handling**: Basic implementation without comprehensive fallback mechanisms

This application is intended as a demonstration and learning tool rather than a production-ready system.

## AI Generation

All code from this project has been generated using highly customized Claude AI prompting. Credit to Anthropic.
