import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup

# Set up sentiment analysis tool using BERT
sentiment_pipeline = pipeline("sentiment-analysis")

# Define a function to scrape news from multiple sources
def scrape_news(stock):
    ticker = yf.Ticker(stock)
    news = ticker.news
    # Example of adding additional scraping from Google News (you can integrate other sources similarly)
    headers = {'User-Agent': 'Mozilla/5.0'}
    google_news_url = f"https://www.google.com/search?q={stock}+stock+news&tbm=nws"
    response = requests.get(google_news_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    for result in soup.find_all('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}):
        title = result.get_text()
        link = result.find_parent('a')['href']
        news.append({'title': title, 'link': link})
    return news[:20]  # Return the first 10 news articles

# Define a function to perform sentiment analysis
def sentiment_analysis(articles):
    aggregate_sentiment = 0
    sentiments = []
    for article in articles:
        title = article['title']
        sentiment = sentiment_pipeline(title)[0]
        if sentiment['label'] == 'POSITIVE':
            score = sentiment['score']
        else:
            score = -sentiment['score']
        aggregate_sentiment += score
        sentiments.append((title, score))
    return aggregate_sentiment / len(articles), sentiments

# Create a Streamlit app
st.title('SENTIMENTai KS V0.3')

# Create a sidebar with buttons for each stock
stocks = {
    'INFY.NS': 'Infosys Limited is an Indian multinational corporation that provides business consulting, information technology, and outsourcing services.',
    'TCS.NS': 'Tata Consultancy Services Limited is an Indian multinational information technology services and consulting company.',
    'RELIANCE.NS': 'Reliance Industries Limited is an Indian multinational conglomerate company, and one of the largest companies in India.',
    'TSLA': 'Tesla, Inc. is an American electric vehicle and clean energy company.',
    'NVDA': 'NVIDIA Corporation is an American multinational technology company, which designs graphics processing units for the gaming and professional markets.',
    'AAPL': 'Apple Inc. is an American multinational technology company that designs, manufactures, and markets consumer electronics, software, and online services.'
}

for stock, description in stocks.items():
    if st.sidebar.button(stock):
        with st.spinner('Analyzing...'):
            time.sleep(6)  # Simulate a delay for the analysis

        # Get current price and time
        ticker = yf.Ticker(stock)
        current_price = ticker.info["currentPrice"]
        previous_close = ticker.info["previousClose"]
        delta = (current_price - previous_close) / previous_close * 100

        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}" if stock in ['TSLA', 'NVDA', 'AAPL'] else f"₹{current_price:.2f}",
            delta=f"{delta:.2f}%"
        )
        st.write(f'**{ticker.info["longName"]}**')
        st.write(description)
        st.write(f'Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # Plot historical price chart
        hist = ticker.history(period='1y')
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        st.plotly_chart(fig)

        # Plot moving averages
        fig2 = go.Figure(data=[
            go.Scatter(x=hist.index, y=hist['Close'].rolling(window=50).mean(), name='50-day MA'),
            go.Scatter(x=hist.index, y=hist['Close'].rolling(window=200).mean(), name='200-day MA')
        ])
        st.plotly_chart(fig2)

        # Scrape news
        articles = scrape_news(stock)

        # Perform sentiment analysis
        aggregate_sentiment, sentiments = sentiment_analysis(articles)

        # Display aggregate sentiment
        st.write('Aggregate Sentiment:')
        if aggregate_sentiment >= 0.05:
            st.write('<font size="5" color="green">**Positive**</font>', unsafe_allow_html=True)
        elif aggregate_sentiment <= -0.05:
            st.write('<font size="5" color="red">**Negative**</font>', unsafe_allow_html=True)
        else:
            st.write('<font size="5" color="blue">**Neutral**</font>', unsafe_allow_html=True)

        # Display news articles with sentiment and reasons
        st.write('News Articles:')
        for i, (title, score) in enumerate(sentiments):
            st.write(f'Article {i+1}:')
            st.write(title)
            st.write(f'Sentiment: {score:.2f}')
            st.write(f"Reason: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} sentiment detected based on the content of the title.")
            st.write(articles[i]['link'])
