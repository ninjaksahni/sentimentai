import yfinance as yf
import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
from datetime import datetime
import time  # Import the time module

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Define descriptions
descriptions = {
    'INFY.NS': 'Infosys Limited is an Indian multinational corporation that provides business consulting, information technology, and outsourcing services.',
    'TCS.NS': 'Tata Consultancy Services Limited is an Indian multinational information technology services and consulting company.',
    'RELIANCE.NS': 'Reliance Industries Limited is an Indian multinational conglomerate company, and one of the largest companies in India.',
    'TSLA': 'Tesla, Inc. is an American electric vehicle and clean energy company.',
    'NVDA': 'NVIDIA Corporation is an American multinational technology company, which designs graphics processing units for the gaming and professional markets.',
    'AAPL': 'Apple Inc. is an American multinational technology company that designs, manufactures, and markets consumer electronics, software, and online services.'
}

def scrape_news(stock):
    ticker = yf.Ticker(stock)
    try:
        news = ticker.news
        if not news:
            st.write("No news available.")
            return []
    except Exception as e:
        st.write(f"Error fetching news: {e}")
        return []

    headers = {'User-Agent': 'Mozilla/5.0'}
    google_news_url = f"https://www.google.com/search?q={stock}+stock+news&tbm=nws"
    try:
        response = requests.get(google_news_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        for result in soup.find_all('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}):
            title = result.get_text()
            link = result.find_parent('a')['href']
            link = extract_real_link(link)
            if link:
                news_items.append({'title': title, 'link': link})
    except Exception as e:
        st.write(f"Error scraping news: {e}")
        news_items = []

    return news_items[:7]  # Return the first 20 news articles

def extract_real_link(link):
    # Helper function to clean and fix the link if necessary
    if link.startswith('/'):
        return f"https://www.google.com{link}"
    return link

def extract_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content, soup
    except Exception as e:
        st.write(f"Error extracting article content: {e}")
        return "", None

def split_text(text, max_length=1000):
    # Split text into chunks for sentiment analysis
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def summarize_text(text):
    # Placeholder summary function, should be replaced with an actual summarizer
    return text[:500] + "..." if len(text) > 500 else text

def extract_article_date(soup):
    # Extract the publication date of the article if available
    date_tag = soup.find('time')
    return date_tag.get('datetime') if date_tag else 'Unknown'

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
        try:
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
            except Exception as e:
                st.write(f"Error analyzing sentiment: {e}")
                continue
        except Exception as e:
            st.write(f"Error extracting article content: {e}")
            continue
    
    if articles:
        aggregate_sentiment /= len(articles)
    
    return aggregate_sentiment, sentiments

def fetch_news_with_retries(stock, retries=3, delay=5):
    for _ in range(retries):
        try:
            return scrape_news(stock)
        except Exception as e:
            st.write(f"Attempt failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    st.write("Failed to fetch news after multiple attempts.")
    return []

# Streamlit app
st.title("Stock Analysis App")

selected_stock = st.selectbox(
    "Select a stock",
    ['TSLA', 'NVDA', 'AAPL', 'ZOMATO.NS', 'INFY.NS', 'TCS.NS', 'RELIANCE.NS']
)

if selected_stock:
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
    articles = fetch_news_with_retries(selected_stock)

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
        st.write(f"Reason: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} sentiment detected based on the content of the article.")
        st.write(f"[Read more]({link})")
