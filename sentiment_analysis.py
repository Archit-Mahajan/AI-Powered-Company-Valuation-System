import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
from google_play_scraper import reviews_all
from app_store_scraper import AppStore
import time
import random
from fake_useragent import UserAgent
import logging

# New/Corrected Imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from github import Github, RateLimitExceededException
from linkedin_api import Linkedin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create global instances
ua = UserAgent()
vader_analyzer = SentimentIntensityAnalyzer()

# Ensure all NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading missing NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# -----------------------------
# Helper Functions
# -----------------------------

def tokenize_text(text):
    """Tokenize text with NLTK with regex fallback"""
    try:
        return nltk.word_tokenize(text)
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        return text.split()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# --- Functions for Textual Data Collection ---

def get_ios_app_id(company_name):
    known_ios_ids = {
        "makemytrip": 530488359, "tesla": 582007913, "uber": 368677368, "airbnb": 401626263, "amazon": 297606951,
        "netflix": 363590051, "spotify": 324684580, "starbucks": 331177714, "walmart": 338137227, "target": 297430070,
        "flipkart": 742044692, "ola": 539179405, "zomato": 434613896, "swiggy": 989540920, "bookmyshow": 404553958,
        "irctc": 6443557980, "paytm": 473941634, "phonepe": 1170055821, "googlepay": 1193357048, "whatsapp": 310633997,
        "instagram": 389801252, "facebook": 284882215, "twitter": 333903271, "linkedin": 288429040, "youtube": 544007664,
        "gmail": 422689480, "hotstar": 934459219, "sonyliv": 587294331, "primevideo": 545519333, "microsoft": 1214786518,
        "zoom": 546505307, "slack": 618783545, "dropbox": 327630330, "adobe": 331014734, "salesforce": 404249815,
        "atlassian": 1535318269, "shopify": 371294472, "wordpress": 335703880, "byjus": 1106230230
    }
    lower_name = company_name.lower()
    if lower_name in known_ios_ids: return known_ios_ids[lower_name]
    for name, app_id in known_ios_ids.items():
        if name in lower_name: return app_id
    return None

def find_play_app_id(company_name):
    known_ids = {
        "tesla": "com.teslamotors.tesla", "makemytrip": "com.makemytrip", "uber": "com.ubercab", "airbnb": "com.airbnb.android",
        "amazon": "com.amazon.mShop.android.shopping", "netflix": "com.netflix.mediaclient", "spotify": "com.spotify.music",
        "starbucks": "com.starbucks.mobilecard", "walmart": "com.walmart.android", "target": "com.target.ui", "flipkart": "com.flipkart.android",
        "ola": "com.olacabs.customer", "zomato": "com.application.zomato", "swiggy": "in.swiggy.android", "bookmyshow": "com.bt.bms",
        "irctc": "cris.org.in.prs.ima", "paytm": "net.one97.paytm", "phonepe": "com.phonepe.app", "googlepay": "com.google.android.apps.nbu.paisa.user",
        "whatsapp": "com.whatsapp", "instagram": "com.instagram.android", "facebook": "com.facebook.katana", "twitter": "com.twitter.android",
        "linkedin": "com.linkedin.android", "youtube": "com.google.android.youtube", "gmail": "com.google.android.gm", "hotstar": "in.startv.hotstar",
        "sonyliv": "com.sonyliv", "primevideo": "com.amazon.avod.thirdpartyclient", "microsoft": "com.microsoft.office.outlook", "zoom": "us.zoom.videomeetings",
        "slack": "com.Slack", "dropbox": "com.dropbox.android", "adobe": "com.adobe.reader", "salesforce": "com.salesforce.chatter",
        "atlassian": "com.atlassian.android.jira.core", "shopify": "com.shopify.mobile", "wordpress": "org.wordpress.android", "byjus": "com.byjus.thelearningapp"
    }
    lower_name = company_name.lower()
    if lower_name in known_ids: return known_ids[lower_name]
    for name, app_id in known_ids.items():
        if name in lower_name: return app_id
    return None

def get_textual_data(company_name):
    logger.info("🔍 Collecting textual data (reviews, news)...")
    all_data = []

    # 1. App Store Reviews
    ios_app_id = get_ios_app_id(company_name)
    if ios_app_id:
        try:
            ios_app = AppStore(country='us', app_name=None, app_id=ios_app_id)
            ios_app.review(how_many=30, sleep=random.randint(1, 3))
            if ios_app.reviews:
                all_data.extend([r['review'] for r in ios_app.reviews])
                logger.info(f"✅ Collected {len(ios_app.reviews)} iOS reviews")
        except Exception as e:
            logger.error(f"❌ iOS review error: {str(e)[:100]}")

    # 2. Google Play Reviews
    play_app_id = find_play_app_id(company_name)
    if play_app_id:
        try:
            gplay_reviews = reviews_all(play_app_id, lang='en', country='us', count=50, sleep_milliseconds=1500)
            all_data.extend([r['content'] for r in gplay_reviews])
            logger.info(f"✅ Collected {len(gplay_reviews)} Android reviews")
        except Exception as e:
            logger.error(f"❌ Play Store scraping error: {e}")

    return all_data

# --- Functions for Traction & Team Strength ---

def get_github_data(company_name):
    logger.info("🔍 Collecting GitHub data for traction metrics...")
    try:
        g = Github(os.getenv("GITHUB_TOKEN"))
        # CORRECTED: Use search_users with type:org to find organizations
        orgs = g.search_users(f'{company_name} type:org')
        if orgs.totalCount > 0:
            org = orgs[0]
            stars, forks = 0, 0
            repos = org.get_repos(sort='stargazers', direction='desc')
            for repo in repos[:5]: # Analyze top 5 repos
                stars += repo.stargazers_count
                forks += repo.forks_count
            logger.info(f"✅ Found GitHub org '{org.login}' with {stars} stars and {forks} forks across top repos.")
            return {"stars": stars, "forks": forks}
    except RateLimitExceededException:
        logger.error("❌ GitHub API rate limit exceeded. Please use a token or wait.")
    except Exception as e:
        logger.error(f"❌ Could not fetch GitHub data: {e}")
    return None

def get_linkedin_data(company_name):
    logger.info("🔍 Collecting LinkedIn data for team metrics...")
    logger.warning("⚠️ Using unofficial LinkedIn API. Use with caution.")
    try:
        api = Linkedin(os.getenv("LINKEDIN_USER"), os.getenv("LINKEDIN_PASSWORD"))
        companies = api.search_companies(keywords=company_name, limit=1)
        if companies:
            company_urn = companies[0]['entityUrn']
            company_details = api.get_company(company_urn)
            employee_count = company_details.get('staffCount', 0)
            logger.info(f"✅ Found LinkedIn company with {employee_count} employees.")
            return {"employee_count": employee_count}
    except Exception as e:
        logger.error(f"❌ Could not fetch LinkedIn data: {e}")
    return None

# -----------------------------
# Analysis & Reporting
# -----------------------------

def label_sentiment(text):
    """UPGRADED: Uses VADER for more nuanced sentiment analysis."""
    if not text or not isinstance(text, str) or not text.strip():
        return 'Neutral'
    compound_score = vader_analyzer.polarity_scores(text)['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_bigrams(texts):
    stop_words = set(stopwords.words("english"))
    custom_stopwords = {'app', 'company', 'service', 'use', 'get'}
    stop_words.update(custom_stopwords)
    bigram_list = []
    for text in texts:
        words = [word for word in tokenize_text(text) if word not in stop_words and len(word) > 2]
        bigrams = ngrams(words, 2)
        bigram_list.extend([' '.join(bg) for bg in bigrams])
    return Counter(bigram_list).most_common(10)

def generate_qualitative_report(company_name):
    """Main function to generate the full qualitative report."""
    # --- 1. Public Perception Analysis ---
    text_data = get_textual_data(company_name)
    if not text_data:
        logger.warning("⚠️ No textual data collected, skipping sentiment analysis.")
    else:
        logger.info(f"\n📊 Analyzing {len(text_data)} text samples for Public Perception...")
        cleaned = [clean_text(d) for d in text_data]
        sentiments = [label_sentiment(d) for d in cleaned]
        df = pd.DataFrame({'text': cleaned, 'sentiment': sentiments})
        
        # Sentiment Distribution
        plt.figure(figsize=(10, 5))
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#4CAF50' if s == 'Positive' else '#F44336' if s == 'Negative' else '#9E9E9E' for s in sentiment_counts.index]
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.title(f"Sentiment Distribution for {company_name}", fontsize=14)
        plt.ylabel('Count', fontsize=12)
        plt.savefig('sentiment_distribution.png', bbox_inches='tight')
        plt.close()

        logger.info("\n📈 Sentiment Analysis Report:")
        pos_count = len(df[df['sentiment'] == 'Positive'])
        neg_count = len(df[df['sentiment'] == 'Negative'])
        neu_count = len(df[df['sentiment'] == 'Neutral'])
        logger.info(f"- Positive: {pos_count} samples")
        logger.info(f"- Negative: {neg_count} samples")
        logger.info(f"- Neutral: {neu_count} samples")

    # --- 2. Innovation & Traction Analysis ---
    github_data = get_github_data(company_name)
    if github_data:
        stars = github_data.get('stars', 0)
        forks = github_data.get('forks', 0)
        traction_score = min(100, (stars / 500) + (forks / 250))
        logger.info("\n🚀 Innovation & Traction Report (GitHub):")
        logger.info(f"- Stars (Top Repos): {stars}")
        logger.info(f"- Forks (Top Repos): {forks}")
        logger.info(f"- Estimated Traction Score: {traction_score:.1f} / 100")

    # --- 3. Team Strength Analysis ---
    linkedin_data = get_linkedin_data(company_name)
    if linkedin_data:
        employees = linkedin_data.get('employee_count', 0)
        team_score = min(100, employees / 100)
        logger.info("\n👥 Team Strength Report (LinkedIn):")
        logger.info(f"- Estimated Employee Count: {employees}")
        logger.info(f"- Estimated Team Strength Score: {team_score:.1f} / 100")

# -----------------------------
# Run Analysis
# -----------------------------
if __name__ == "__main__":
    company = input("Enter company name: ").strip()
    if company:
        start_time = time.time()
        generate_qualitative_report(company)
        logger.info(f"\n\n⏱️  Analysis completed in {time.time() - start_time:.2f} seconds")
    else:
        logger.error("Please provide a valid company name")