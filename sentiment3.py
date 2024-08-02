import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline, BertTokenizer
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# Initialize the BERT tokenizer and sentiment analysis pipeline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_pipeline = pipeline("sentiment-analysis")

def truncate_text(text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def split_text(text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def extract_article_content(link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception:
        return ""

def extract_real_link(google_news_url):
    try:
        parsed_url = urlparse(google_news_url)
        if parsed_url.netloc != 'www.google.com':
            return google_news_url
        
        query_params = parse_qs(parsed_url.query)
        return query_params.get('url', [None])[0]
    except Exception:
        return google_news_url

def sentiment_analysis(articles):
    aggregate_sentiment = 0
    sentiments = []
    if not articles:
        return aggregate_sentiment, sentiments
    
    for article in articles:
        title = article.get('title', '')
        link = extract_real_link(article.get('link', ''))
        if not title or not link:
            continue
        
        # Extract article content
        article_content = extract_article_content(link)
        if not article_content:
            continue

        # Split and analyze the article content
        chunks = split_text(article_content)
        
        try:
            chunk_sentiments = []
            for chunk in chunks:
                sentiment = sentiment_pipeline(chunk)[0]
                score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
                chunk_sentiments.append(score)
            
            # Average the sentiment scores for the chunks
            avg_score = sum(chunk_sentiments) / len(chunk_sentiments)
            aggregate_sentiment += avg_score
            sentiments.append((title, avg_score))
        except Exception:
            continue
    
    if articles:
        aggregate_sentiment /= len(articles)
    
    return aggregate_sentiment, sentiments

def scrape_news(stock):
    ticker = yf.Ticker(stock)
    news = ticker.news
    headers = {'User-Agent': 'Mozilla/5.0'}
    google_news_url = f"https://www.google.com/search?q={stock}+stock+news&tbm=nws"
    try:
        response = requests.get(google_news_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for result in soup.find_all('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}):
            title = result.get_text()
            link = result.find_parent('a')['href']
            # Google News links need extraction
            link = extract_real_link(link)
            if link:
                news.append({'title': title, 'link': link})
    except Exception:
        pass
    return news[:20]  # Return the first 20 news articles

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
        current_price = ticker.info.get("currentPrice", "N/A")
        previous_close = ticker.info.get("previousClose", "N/A")
        delta = (current_price - previous_close) / previous_close * 100 if previous_close else 0

        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}" if stock in ['TSLA', 'NVDA', 'AAPL'] else f"₹{current_price:.2f}",
            delta=f"{delta:.2f}%"
        )
        st.write(f'**{ticker.info.get("longName", "Unknown")}**')
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
            st.write(f"Reason: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} sentiment detected based on the content of the article.")
            st.write(articles[i]['link'])
