import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline, BertTokenizer
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# Initialize the BERT tokenizer and summarization pipeline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_pipeline = pipeline("sentiment-analysis")
summarization_pipeline = pipeline("summarization")

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
        return content, soup
    except Exception:
        return "", None

def extract_article_date(soup):
    try:
        date_meta = soup.find('meta', {'property': 'article:published_time'})
        if date_meta:
            return datetime.fromisoformat(date_meta['content']).strftime('%Y-%m-%d')
        date_meta = soup.find('time')
        if date_meta:
            return date_meta.get_text().strip()
    except Exception:
        return "Unknown"
    return "Unknown"

def extract_real_link(google_news_url):
    try:
        parsed_url = urlparse(google_news_url)
        if parsed_url.netloc != 'www.google.com':
            return google_news_url
        
        query_params = parse_qs(parsed_url.query)
        return query_params.get('url', [None])[0]
    except Exception:
        return google_news_url

def summarize_text(text):
    try:
        # Summarize the text and return the summary
        summary = summarization_pipeline(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return "Summary not available."

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
        article_content, soup = extract_article_content(link)
        if not article_content or not soup:
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
            article_date = extract_article_date(soup)
            summary = summarize_text(article_content)
            sentiments.append((title, avg_score, article_date, summary, link))
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

# Get stock descriptions from Yahoo Finance dynamically
stocks = ['INFY.NS', 'TCS.NS', 'RELIANCE.NS', 'TSLA', 'NVDA', 'AAPL']
descriptions = {}

for stock in stocks:
    ticker = yf.Ticker(stock)
    description = ticker.info.get("longBusinessSummary", "Description not available")
    descriptions[stock] = description

# Main content
st.write("Welcome to SENTIMENTai KS V0.3! Select a stock from the buttons below to view detailed analysis and sentiment.")

# Stock buttons in one row
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    if st.button('INFY.NS'):
        selected_stock = 'INFY.NS'
with col2:
    if st.button('TCS.NS'):
        selected_stock = 'TCS.NS'
with col3:
    if st.button('RELIANCE.NS'):
        selected_stock = 'RELIANCE.NS'
with col4:
    if st.button('TSLA'):
        selected_stock = 'TSLA'
with col5:
    if st.button('NVDA'):
        selected_stock = 'NVDA'
with col6:
    if st.button('AAPL'):
        selected_stock = 'AAPL'

if 'selected_stock' in locals():
    with st.spinner('Analyzing...'):
        time.sleep(6)  # Simulate a delay for the analysis

    # Get current price and time
    ticker = yf.Ticker(selected_stock)
    current_price = ticker.info.get("currentPrice", "N/A")
    previous_close = ticker.info.get("previousClose", "N/A")
    delta = (current_price - previous_close) / previous_close * 100 if previous_close else 0

    st.metric(
        label="Current Price",
        value=f"${current_price:.2f}" if selected_stock in ['TSLA', 'NVDA', 'AAPL'] else f"₹{current_price:.2f}",
        delta=f"{delta:.2f}%"
    )
    st.write(f'**{ticker.info.get("longName", "Unknown")}**')
    st.write(descriptions[selected_stock])
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
    articles = scrape_news(selected_stock)

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

    # Display news articles with sentiment, reasons, and summaries
    st.write('News Articles:')
    for i, (title, score, date, summary, link) in enumerate(sentiments):
        st.write(f'Article {i+1}:')
        st.write(f'Title: {title}')
        st.write(f'Sentiment: {score:.2f}')
        st.write(f"Date: {date}")
        st.write(f"Summary: <font color='orange'>{summary}</font>", unsafe_allow_html=True)
       
        st.write(f"[Read Article]({link})")
