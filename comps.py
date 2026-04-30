import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
import warnings
from datetime import datetime
import time
import json
from urllib.parse import quote
import yfinance as yf
import google.generativeai as genai
# Suppress warnings
warnings.filterwarnings('ignore')
# Initialize NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Set plotting style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 8)
# Initialize Gemini API from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set. Please configure it in your .env file.")
genai.configure(api_key=API_KEY)
# Try different model names in order of preference
model_names = [
    'gemini-2.5-flash-preview-09-2025', # Specific new version for stability
    'gemini-flash-latest',               # Convenient alias for the newest Flash
    'gemini-1.5-flash',                  # Fallback to the older flash
    'gemini-1.5-pro',                    # Fallback to pro
    'gemini-pro'
]
model = None
for model_name in model_names:
    try:
        model = genai.GenerativeModel(model_name)
        # Test with a simple prompt
        test_response = model.generate_content("Hello")
        print(f"✅ Successfully connected to Gemini model: {model_name}")
        break
    except Exception as e:
        print(f"⚠️ Failed to connect to {model_name}: {e}")
        continue
if model is None:
    print("❌ Could not connect to any Gemini model. Falling back to web scraping only.")
    use_gemini = False
else:
    use_gemini = True
class UniversalCompsEngine:
    def __init__(self, target_company_name):
        self.target_company_name = target_company_name
        self.target_ticker = None
        self.is_public_company = False
        self.companies = []
        self.company_data = {}  # Store all company information
        self.descriptions = {}
        self.similarity_scores = None
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.target_sector = None
        self.target_industry = None
        self.target_location = None
    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = [lemmatizer.lemmatize(word.lower()) 
                    for word in text.split() 
                    if word.lower() not in stop_words and len(word) > 2]
        return " ".join(tokens)
    def _is_valid_company_name(self, name):
        """Check if a name is a valid company name"""
        if not name or len(name) < 3:
            return False
        # Convert to lowercase for easier checking
        name_lower = name.lower()
        # Common non-company words/phrases to exclude
        invalid_terms = [
            'google', 'facebook', 'youtube', 'wikipedia', 'amazon', 'microsoft',
            'feedback', 'similar', 'images', 'news', 'short', 'videos', 'forum',
            'overview', 'listen', 'pause', 'here', 'there', 'you', 'we', 'they',
            'search', 'results', 'about', 'contact', 'privacy', 'terms',
            'skip', 'content', 'mobile', 'english', 'hindi', 'bengali',
            'all', 'more', 'shopping', 'flights', 'travel', 'tools',
            'sign', 'in', 'up', 'log', 'register', 'help', 'support'
        ]
        # Check if name contains any invalid terms
        for term in invalid_terms:
            if term in name_lower:
                return False
        # Check if name is just a common word
        if name_lower in ['inc', 'corp', 'llc', 'ltd', 'co', 'company', 'technologies', 'tech']:
            return False
        # Check if name starts with a capital letter (proper noun)
        if not name[0].isupper():
            return False
        # Check if name contains mostly letters and spaces
        if not re.match(r'^[A-Za-z\s&.-]+$', name):
            return False
        # Check if name has at least 3 characters
        if len(name.replace(' ', '')) < 3:
            return False
        return True
    def _find_competitors_with_gemini(self, company_name):
        """Use Gemini API to find competitors"""
        if not use_gemini:
            return []
        try:
            # Create prompt for Gemini
            prompt = f"""
            I need to find competitors for the company "{company_name}". 
            Please provide a list of 10-15 direct competitors or similar companies in the same industry.
            Return only the company names, one per line, without any additional text, numbers, or formatting.
            Do not include the company "{company_name}" itself in the list.
            """
            # Generate response
            response = model.generate_content(prompt)
            competitors_text = response.text
            # Parse the response to extract company names
            competitors = []
            lines = competitors_text.strip().split('\n')
            for line in lines:
                # Clean up each line
                company = line.strip().strip('"').strip("'").strip("*").strip("-")
                # Remove any numbering
                company = re.sub(r'^\d+\.\s*', '', company)
                if company and self._is_valid_company_name(company):
                    competitors.append(company)
            return competitors[:15]  # Limit to 15 competitors
        except Exception as e:
            print(f"Error using Gemini API: {e}")
            return []
    def _get_company_info_comprehensive(self, company_name):
        """Get comprehensive company information from multiple sources"""
        company_info = {
            'name': company_name,
            'ticker': None,
            'is_public': False,
            'description': '',
            'sector': '',
            'industry': '',
            'location': '',
            'founding_year': None,
            'employee_count': None,
            'revenue': None,
            'funding': None,
            'competitors': [],
            'website': '',
            'market_cap': None,
            'valuation': None,
            'business_model': '',
            'key_products': []
        }
        # Method 1: Try to get ticker and public company data
        ticker = self._find_ticker_from_name(company_name)
        if ticker and self._validate_ticker(ticker):
            company_info['ticker'] = ticker
            company_info['is_public'] = True
            public_data = self._get_public_company_data(ticker)
            company_info.update(public_data)
        # Method 2: Web scraping for general company information
        web_data = self._scrape_company_info(company_name)
        for key, value in web_data.items():
            if not company_info[key] or company_info[key] == '':
                company_info[key] = value
        # Method 3: Search for startup/private company information
        if not company_info['is_public']:
            startup_data = self._get_startup_info(company_name)
            for key, value in startup_data.items():
                if not company_info[key] or company_info[key] == '':
                    company_info[key] = value
        # Method 4: Use Gemini API to find competitors (if available)
        if use_gemini:
            competitors = self._find_competitors_with_gemini(company_name)
            if competitors:
                company_info['competitors'] = competitors
            else:
                # Fallback to web scraping if Gemini fails
                competitors = self._find_competitors_any_company(company_name, company_info)
                company_info['competitors'] = competitors
        else:
            # Use web scraping if Gemini is not available
            competitors = self._find_competitors_any_company(company_name, company_info)
            company_info['competitors'] = competitors
        return company_info
    def _find_ticker_from_name(self, company_name):
        """Find ticker symbol from company name"""
        try:
            search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={quote(company_name)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            data = response.json()
            if 'quotes' in data and len(data['quotes']) > 0:
                for quote_data in data['quotes']:
                    if quote_data.get('quoteType') == 'EQUITY':
                        ticker = quote_data.get('symbol', '')
                        name = quote_data.get('longname', '') or quote_data.get('shortname', '')
                        if company_name.lower() in name.lower() or name.lower() in company_name.lower():
                            return ticker
                return data['quotes'][0].get('symbol', '')
        except:
            pass
        return None
    def _validate_ticker(self, ticker):
        """Validate if ticker is real and active"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'marketCap' in info and info.get('marketCap', 0) > 0
        except:
            return False
    def _get_public_company_data(self, ticker):
        """Get data for public companies"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'description': info.get('longBusinessSummary', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'location': info.get('country', ''),
                'employee_count': info.get('fullTimeEmployees', None),
                'market_cap': info.get('marketCap', None),
                'website': info.get('website', ''),
                'revenue': info.get('totalRevenue', None)
            }
        except:
            return {}
    def _scrape_company_info(self, company_name):
        """Scrape general company information from web"""
        company_info = {
            'description': '',
            'sector': '',
            'industry': '',
            'location': '',
            'website': '',
            'employee_count': None,
            'founding_year': None
        }
        try:
            # Use multiple sources for better information
            sources = [
                self._search_google_for_company_info,
                self._search_wikipedia_for_company_info,
                self._search_bing_for_company_info
            ]
            for source in sources:
                try:
                    info = source(company_name)
                    for key, value in info.items():
                        if not company_info[key] or company_info[key] == '':
                            company_info[key] = value
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"Error with {source.__name__}: {e}")
                    continue
        except Exception as e:
            print(f"Error scraping company info: {e}")
        return company_info
    def _search_google_for_company_info(self, company_name):
        """Search Google for company information"""
        company_info = {}
        search_queries = [
            f"{company_name} company information",
            f"{company_name} about us",
            f"{company_name} business description",
            f'"{company_name}" company profile'
        ]
        for query in search_queries:
            try:
                search_url = f"https://www.google.com/search?q={quote(query)}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(search_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text content from search results
                search_results = []
                for result in soup.find_all('div', class_='g'):
                    result_text = result.get_text()
                    search_results.append(result_text)
                text_content = ' '.join(search_results)
                # Look for description patterns
                description_patterns = [
                    rf"{re.escape(company_name)}[^.]*?(?:is|was|operates|provides|offers|specializes|focuses)[^.]*?\.",
                    rf"About {re.escape(company_name)}[^.]*?\.",
                    rf"{re.escape(company_name)}[^.]*?(?:company|business|organization|firm|corporation)[^.]*?\."
                ]
                for pattern in description_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches and not company_info.get('description'):
                        company_info['description'] = matches[0][:500]  # Limit length
                # Look for location information
                location_patterns = [
                    rf"{re.escape(company_name)}[^.]*?(?:based in|located in|headquartered in|founded in)\s+([^.,]+)",
                    rf"(?:based in|located in|headquartered in)\s+([^.,]+)[^.]*?{re.escape(company_name)}"
                ]
                for pattern in location_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches and not company_info.get('location'):
                        company_info['location'] = matches[0]
                # Look for founding year
                year_patterns = [
                    rf"{re.escape(company_name)}[^.]*?(?:founded|established|started|launched|created)\s+(?:in\s+)?(\d{{4}})",
                    rf"(?:founded|established|started|launched|created)\s+(?:in\s+)?(\d{{4}})[^.]*?{re.escape(company_name)}"
                ]
                for pattern in year_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches and not company_info.get('founding_year'):
                        year = int(matches[0])
                        if 1800 <= year <= 2025:  # Reasonable year range
                            company_info['founding_year'] = year
                # Look for employee count
                employee_patterns = [
                    rf"{re.escape(company_name)}[^.]*?(?:employs|has|with)\s+(?:over\s+|more than\s+|approximately\s+)?(\d+(?:,\d+)*)\s+(?:employees|people|staff)",
                    rf"(\d+(?:,\d+)*)\s+(?:employees|people|staff)[^.]*?{re.escape(company_name)}"
                ]
                for pattern in employee_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches and not company_info.get('employee_count'):
                        try:
                            emp_count = int(matches[0].replace(',', ''))
                            if emp_count > 0 and emp_count < 10000000:  # Reasonable range
                                company_info['employee_count'] = emp_count
                        except:
                            pass
                # Look for website
                website_patterns = [
                    rf"{re.escape(company_name)}[^.]*?(?:website|site|url|visit us at)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}})",
                    rf"(?:website|site|url|visit us at)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}})[^.]*?{re.escape(company_name)}"
                ]
                for pattern in website_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches and not company_info.get('website'):
                        website = matches[0]
                        if not website.startswith('http'):
                            website = 'https://' + website
                        company_info['website'] = website
                # Look for sector/industry
                sector_patterns = [
                    rf"{re.escape(company_name)}[^.]*?(?:sector|industry)[^.]*?([A-Z][a-zA-Z\s&]+)",
                    rf"(?:sector|industry)[^.]*?([A-Z][a-zA-Z\s&]+)[^.]*?{re.escape(company_name)}"
                ]
                for pattern in sector_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches and not company_info.get('sector'):
                        sector = matches[0].strip()
                        if len(sector) > 3 and len(sector) < 50:
                            company_info['sector'] = sector
                if company_info.get('description') and company_info.get('location'):
                    break  # We have enough info
            except Exception as e:
                print(f"Error in Google search: {e}")
                continue
        return company_info
    def _search_wikipedia_for_company_info(self, company_name):
        """Search Wikipedia for company information"""
        company_info = {}
        try:
            # Try to find the Wikipedia page for the company
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(company_name)}&format=json"
            response = requests.get(wiki_url, timeout=10)
            data = response.json()
            if 'query' in data and 'search' in data['query'] and len(data['query']['search']) > 0:
                # Get the first search result
                page_title = data['query']['search'][0]['title']
                # Get the page content
                page_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={quote(page_title)}&prop=extracts&exintro&format=json"
                page_response = requests.get(page_url, timeout=10)
                page_data = page_response.json()
                if 'query' in page_data and 'pages' in page_data['query']:
                    for page_id, page_info in page_data['query']['pages'].items():
                        if 'extract' in page_info:
                            extract = page_info['extract']
                            # Extract description
                            if not company_info.get('description'):
                                # Get the first paragraph
                                first_paragraph = extract.split('\n')[0]
                                if len(first_paragraph) > 50:
                                    company_info['description'] = first_paragraph[:500]
                            # Look for founding year
                            year_match = re.search(r'founded in (\d{4})', extract, re.IGNORECASE)
                            if year_match and not company_info.get('founding_year'):
                                company_info['founding_year'] = int(year_match.group(1))
                            # Look for location
                            location_match = re.search(r'based in ([^.]+)', extract, re.IGNORECASE)
                            if location_match and not company_info.get('location'):
                                company_info['location'] = location_match.group(1)
                            # Look for industry
                            industry_match = re.search(r'industry\s*[:=]\s*([^.]+)', extract, re.IGNORECASE)
                            if industry_match and not company_info.get('industry'):
                                company_info['industry'] = industry_match.group(1)
                            break
        except Exception as e:
            print(f"Error in Wikipedia search: {e}")
        return company_info
    def _search_bing_for_company_info(self, company_name):
        """Search Bing for company information"""
        company_info = {}
        try:
            search_url = f"https://www.bing.com/search?q={quote(company_name + ' company information')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text content from search results
            search_results = []
            for result in soup.find_all('li', class_='b_algo'):
                result_text = result.get_text()
                search_results.append(result_text)
            text_content = ' '.join(search_results)
            # Look for description patterns
            description_patterns = [
                rf"{re.escape(company_name)}[^.]*?(?:is|was|operates|provides|offers)[^.]*?\.",
                rf"About {re.escape(company_name)}[^.]*?\.",
            ]
            for pattern in description_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches and not company_info.get('description'):
                    company_info['description'] = matches[0][:500]
            # Look for location information
            location_patterns = [
                rf"{re.escape(company_name)}[^.]*?(?:based in|located in|headquartered in)\s+([^.,]+)",
            ]
            for pattern in location_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches and not company_info.get('location'):
                    company_info['location'] = matches[0]
        except Exception as e:
            print(f"Error in Bing search: {e}")
        return company_info
    def _get_startup_info(self, company_name):
        """Get information specific to startups and private companies"""
        startup_info = {
            'funding': None,
            'valuation': None,
            'business_model': '',
            'key_products': []
        }
        try:
            # Search for startup/funding information
            funding_queries = [
                f"{company_name} funding raised investment",
                f"{company_name} Series A B C funding",
                f"{company_name} valuation startup",
                f'"{company_name}" private company'
            ]
            for query in funding_queries:
                try:
                    search_url = f"https://www.google.com/search?q={quote(query)}"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(search_url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract text content from search results
                    search_results = []
                    for result in soup.find_all('div', class_='g'):
                        result_text = result.get_text()
                        search_results.append(result_text)
                    text_content = ' '.join(search_results)
                    # Look for funding information
                    funding_patterns = [
                        rf"{re.escape(company_name)}[^.]*?(?:raised|secured|received)\s+\$([0-9.]+)\s*(million|billion|M|B)",
                        rf"\$([0-9.]+)\s*(million|billion|M|B)[^.]*?{re.escape(company_name)}[^.]*?(?:funding|investment|round)"
                    ]
                    for pattern in funding_patterns:
                        matches = re.findall(pattern, text_content, re.IGNORECASE)
                        if matches and not startup_info['funding']:
                            amount, unit = matches[0]
                            multiplier = 1000000 if unit.lower() in ['million', 'm'] else 1000000000
                            startup_info['funding'] = float(amount) * multiplier
                    # Look for valuation
                    valuation_patterns = [
                        rf"{re.escape(company_name)}[^.]*?(?:valued at|valuation of)\s+\$([0-9.]+)\s*(million|billion|M|B)",
                        rf"valuation[^.]*?\$([0-9.]+)\s*(million|billion|M|B)[^.]*?{re.escape(company_name)}"
                    ]
                    for pattern in valuation_patterns:
                        matches = re.findall(pattern, text_content, re.IGNORECASE)
                        if matches and not startup_info['valuation']:
                            amount, unit = matches[0]
                            multiplier = 1000000 if unit.lower() in ['million', 'm'] else 1000000000
                            startup_info['valuation'] = float(amount) * multiplier
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    continue
        except Exception as e:
            print(f"Error getting startup info: {e}")
        return startup_info
    def _find_competitors_any_company(self, company_name, company_info):
        """Find competitors for any type of company using multiple sources"""
        competitors = []
        # Method 1: Use industry/sector to find similar companies
        if company_info.get('industry'):
            industry_comps = self._find_competitors_by_industry(company_info['industry'], company_name)
            competitors.extend(industry_comps)
        # Method 2: Use business description to find similar companies
        if company_info.get('description'):
            desc_comps = self._find_competitors_by_description(company_info['description'], company_name)
            competitors.extend(desc_comps)
        # Method 3: Search for direct competitors
        direct_comps = self._search_direct_competitors(company_name, company_info)
        competitors.extend(direct_comps)
        # Remove duplicates and invalid names
        competitors = list(set(competitors))
        if company_name in competitors:
            competitors.remove(company_name)
        # Validate company names
        valid_competitors = []
        for comp in competitors:
            if self._is_valid_company_name(comp):
                valid_competitors.append(comp)
        return valid_competitors[:15]  # Return top 15 competitors
    def _find_competitors_by_industry(self, industry, company_name):
        """Find competitors by industry using web search"""
        competitors = []
        try:
            # Search for companies in the same industry
            search_url = f"https://www.google.com/search?q={quote(industry + ' companies list')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text content from search results
            search_results = []
            for result in soup.find_all('div', class_='g'):
                result_text = result.get_text()
                search_results.append(result_text)
            text_content = ' '.join(search_results)
            # Look for company names with more specific patterns
            company_patterns = [
                r'([A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Co|Company|Technologies|Tech|Systems|Solutions|Group|International)\.?)',
                r'([A-Z][a-zA-Z\s&]+(?:\.com|\.net|\.org))',
                r'([A-Z][a-zA-Z\s&]{3,25})'  # General company name pattern
            ]
            potential_competitors = set()
            for pattern in company_patterns:
                matches = re.findall(pattern, text_content)
                for match in matches:
                    if self._is_valid_company_name(match):
                        potential_competitors.add(match.strip())
            # Add to competitors list
            for comp in list(potential_competitors)[:10]:  # Limit per query
                if comp not in competitors:
                    competitors.append(comp)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"Error finding competitors by industry: {e}")
        return competitors
    def _find_competitors_by_description(self, description, company_name):
        """Find competitors by analyzing business description"""
        competitors = []
        try:
            # Extract key terms from description
            key_terms = []
            words = description.split()
            for word in words:
                if len(word) > 5 and word.isalpha() and word.lower() not in stop_words:
                    key_terms.append(word)
            if key_terms:
                # Search for companies with similar terms
                search_terms = ' '.join(key_terms[:5])  # Use top 5 terms
                search_url = f"https://www.google.com/search?q={quote(search_terms + ' companies')}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(search_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text content from search results
                search_results = []
                for result in soup.find_all('div', class_='g'):
                    result_text = result.get_text()
                    search_results.append(result_text)
                text_content = ' '.join(search_results)
                # Look for company names with more specific patterns
                company_patterns = [
                    r'([A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Co|Company|Technologies|Tech|Systems|Solutions|Group|International)\.?)',
                    r'([A-Z][a-zA-Z\s&]+(?:\.com|\.net|\.org))',
                    r'([A-Z][a-zA-Z\s&]{3,25})'  # General company name pattern
                ]
                potential_competitors = set()
                for pattern in company_patterns:
                    matches = re.findall(pattern, text_content)
                    for match in matches:
                        if self._is_valid_company_name(match):
                            potential_competitors.add(match.strip())
                # Add to competitors list
                for comp in list(potential_competitors)[:8]:  # Limit per query
                    if comp not in competitors:
                        competitors.append(comp)
                time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"Error finding competitors by description: {e}")
        return competitors
    def _search_direct_competitors(self, company_name, company_info):
        """Search for direct competitors using multiple queries"""
        competitors = []
        try:
            # Multiple search strategies
            search_strategies = [
                f"{company_name} competitors alternatives",
                f"{company_name} vs competitors",
                f"companies like {company_name}",
                f"{company_name} similar companies",
                f'alternatives to "{company_name}"',
                f"{company_name} market competition"
            ]
            if company_info.get('industry'):
                search_strategies.append(f"{company_info['industry']} companies similar to {company_name}")
            if company_info.get('sector'):
                search_strategies.append(f"{company_info['sector']} companies similar to {company_name}")
            for query in search_strategies:
                try:
                    search_url = f"https://www.google.com/search?q={quote(query)}"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(search_url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract text content from search results
                    search_results = []
                    for result in soup.find_all('div', class_='g'):
                        result_text = result.get_text()
                        search_results.append(result_text)
                    text_content = ' '.join(search_results)
                    # Look for company names with more specific patterns
                    company_patterns = [
                        r'([A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Co|Company|Technologies|Tech|Systems|Solutions|Group|International)\.?)',
                        r'([A-Z][a-zA-Z\s&]+(?:\.com|\.net|\.org))',
                        r'([A-Z][a-zA-Z\s&]{3,25})'  # General company name pattern
                    ]
                    potential_competitors = set()
                    for pattern in company_patterns:
                        matches = re.findall(pattern, text_content)
                        for match in matches:
                            if self._is_valid_company_name(match):
                                potential_competitors.add(match.strip())
                    # Add to competitors list
                    for comp in list(potential_competitors)[:5]:  # Limit per query
                        if comp not in competitors:
                            competitors.append(comp)
                    if len(competitors) >= 15:  # Overall limit
                        break
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"Error in search strategy '{query}': {e}")
                    continue
        except Exception as e:
            print(f"Error searching direct competitors: {e}")
        return competitors
    def collect_data(self, num_companies=20):
        """Collect data for target company and competitors"""
        print(f"🔍 Collecting comprehensive data for '{self.target_company_name}'...")
        # Get target company information
        target_data = self._get_company_info_comprehensive(self.target_company_name)
        self.companies.append(self.target_company_name)
        self.company_data[self.target_company_name] = target_data
        self.descriptions[self.target_company_name] = self._clean_text(target_data['description'])
        # Set target company attributes
        self.target_sector = target_data['sector']
        self.target_industry = target_data['industry']
        self.target_location = target_data['location']
        self.is_public_company = target_data['is_public']
        print(f"✅ Target company: {self.target_company_name}")
        print(f"   Type: {'Public' if self.is_public_company else 'Private'}")
        print(f"   Sector: {self.target_sector or 'Unknown'}")
        print(f"   Industry: {self.target_industry or 'Unknown'}")
        print(f"   Location: {self.target_location or 'Unknown'}")
        # Get competitor data
        competitors = target_data['competitors']
        print(f"🔍 Found {len(competitors)} potential competitors")
        for competitor_name in competitors:
            if len(self.companies) >= num_companies:
                break
            if competitor_name not in self.companies:
                try:
                    print(f"   Analyzing: {competitor_name}")
                    competitor_data = self._get_company_info_comprehensive(competitor_name)
                    # Basic filtering - must have some description
                    if competitor_data['description']:
                        self.companies.append(competitor_name)
                        self.company_data[competitor_name] = competitor_data
                        self.descriptions[competitor_name] = self._clean_text(competitor_data['description'])
                        print(f"   ✅ Added: {competitor_name}")
                    else:
                        print(f"   ⚠️ Skipped: {competitor_name} (insufficient data)")
                except Exception as e:
                    print(f"   ❌ Error processing {competitor_name}: {e}")
                time.sleep(0.5)  # Rate limiting
        print(f"✅ Collected data for {len(self.companies)} companies")
    def _embed_descriptions(self):
        """Create document embeddings using Sentence Transformers"""
        descriptions = list(self.descriptions.values())
        embeddings = self.text_model.encode(descriptions)
        return np.array(embeddings)
    def analyze_comparables(self):
        """Perform comparable company analysis using text similarity and company attributes"""
        if len(self.companies) < 2:
            raise ValueError("Need at least 2 companies for analysis")
        # 1. Text similarity analysis
        text_embeddings = self._embed_descriptions()
        # 2. Create company attribute features
        attribute_features = []
        target_data = self.company_data[self.target_company_name]
        for company_name in self.companies:
            data = self.company_data[company_name]
            # Create feature vector based on available attributes
            features = []
            # Sector similarity
            sector_sim = 1 if (data['sector'] and target_data['sector'] and 
                                data['sector'].lower() == target_data['sector'].lower()) else 0
            features.append(sector_sim)
            # Industry similarity
            industry_sim = 1 if (data['industry'] and target_data['industry'] and 
                                data['industry'].lower() == target_data['industry'].lower()) else 0
            features.append(industry_sim)
            # Location similarity
            location_sim = 1 if (data['location'] and target_data['location'] and 
                                data['location'].lower() == target_data['location'].lower()) else 0
            features.append(location_sim)
            # Company type similarity (public vs private)
            type_sim = 1 if data['is_public'] == target_data['is_public'] else 0
            features.append(type_sim)
            # Size similarity (based on employees or market cap/valuation)
            size_sim = 0
            if data['employee_count'] and target_data['employee_count']:
                ratio = min(data['employee_count'], target_data['employee_count']) / max(data['employee_count'], target_data['employee_count'])
                size_sim = ratio
            elif data.get('market_cap') and target_data.get('market_cap'):
                ratio = min(data['market_cap'], target_data['market_cap']) / max(data['market_cap'], target_data['market_cap'])
                size_sim = ratio
            elif data.get('valuation') and target_data.get('valuation'):
                ratio = min(data['valuation'], target_data['valuation']) / max(data['valuation'], target_data['valuation'])
                size_sim = ratio
            features.append(size_sim)
            # Direct competitor flag
            competitor_flag = 1 if company_name in target_data['competitors'] else 0
            features.append(competitor_flag)
            attribute_features.append(features)
        attribute_features = np.array(attribute_features)
        # 3. Combine text and attribute features
        text_weight = 0.7  # Higher weight for text similarity
        attr_weight = 0.3
        # Normalize features
        text_scaler = StandardScaler()
        attr_scaler = StandardScaler()
        scaled_text = text_scaler.fit_transform(text_embeddings) * text_weight
        scaled_attr = attr_scaler.fit_transform(attribute_features) * attr_weight
        combined_features = np.hstack((scaled_text, scaled_attr))
        # 4. Calculate similarity scores
        target_idx = 0  # Target company is always first
        target_vector = combined_features[target_idx].reshape(1, -1)
        self.similarity_scores = cosine_similarity(target_vector, combined_features)[0]
        # Normalize similarity scores
        min_score = np.min(self.similarity_scores)
        max_score = np.max(self.similarity_scores)
        if max_score > min_score:
            self.similarity_scores = (self.similarity_scores - min_score) / (max_score - min_score)
        # 5. Create results dataframe
        results_data = []
        for i, company_name in enumerate(self.companies):
            data = self.company_data[company_name]
            results_data.append({
                'Company': company_name,
                'Similarity_Score': self.similarity_scores[i],
                'Type': 'Public' if data['is_public'] else 'Private',
                'Ticker': data.get('ticker', 'N/A'),
                'Sector': data['sector'],
                'Industry': data['industry'],
                'Location': data['location'],
                'Employees': data['employee_count'],
                'Market_Cap': data.get('market_cap'),
                'Valuation': data.get('valuation'),
                'Funding': data.get('funding'),
                'Founded': data['founding_year'],
                'Website': data['website'],
                'Is_Direct_Competitor': company_name in self.company_data[self.target_company_name]['competitors']
            })
        results = pd.DataFrame(results_data)
        results = results.sort_values('Similarity_Score', ascending=False)
        return results
    def visualize_results(self, results):
        """Generate visualizations for any type of company analysis"""
        # 1. Company similarity network
        plt.figure(figsize=(14, 10))
        G = nx.Graph()
        # Add nodes
        for _, row in results.iterrows():
            if row['Company'] == self.target_company_name:
                size = 800
                color = 'red'
            elif row['Is_Direct_Competitor']:
                size = 400
                color = 'gold'
            elif row['Type'] == 'Public':
                size = 300
                color = 'lightblue'
            else:
                size = 250
                color = 'lightgreen'
            G.add_node(row['Company'], size=size, color=color, 
                        similarity=row['Similarity_Score'], type=row['Type'])
        # Add edges to top similar companies
        top_similar = results[results['Company'] != self.target_company_name].head(8)
        for _, row in top_similar.iterrows():
            G.add_edge(self.target_company_name, row['Company'], 
                        weight=row['Similarity_Score'])
        # Layout and draw
        pos = nx.spring_layout(G, k=1.2, seed=42)
        node_colors = [G.nodes[node]['color'] for node in G.nodes]
        node_sizes = [G.nodes[node]['size'] for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                alpha=0.8, edgecolors='black')
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.4, edge_color='gray')
        # Labels
        labels = {}
        for node in G.nodes:
            # Truncate long company names
            label = node if len(node) <= 15 else node[:12] + "..."
            labels[node] = label
        nx.draw_networkx_labels(G, pos, labels, font_size=8,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        plt.title(f'Company Similarity Network: {self.target_company_name}\n'
                    f'(Red=Target, Gold=Direct Competitor, Blue=Public, Green=Private)', 
                    fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('universal_comps_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        # 2. Company type and similarity distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Company type distribution
        type_counts = results['Type'].value_counts()
        colors = ['lightblue', 'lightgreen']
        ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Company Type Distribution')
        # Similarity score distribution
        ax2.hist(results[results['Company'] != self.target_company_name]['Similarity_Score'],
                bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(results[results['Company'] != self.target_company_name]['Similarity_Score'].mean(),
                    color='red', linestyle='--', label='Average Similarity')
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Number of Companies')
        ax2.set_title('Similarity Score Distribution')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('universal_comps_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated visualizations: universal_comps_network.png, universal_comps_analysis.png")
    def generate_dashboard(self, results):
        """Generate comprehensive dashboard for any type of company"""
        # Prepare summary data
        top_comps = results[results['Company'] != self.target_company_name].head(10)
        target_data = self.company_data[self.target_company_name]
        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Universal Company Analysis: {self.target_company_name}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                .header {{ 
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 40px 0; margin-bottom: 30px; 
                }}
                .card {{ margin-bottom: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-card {{ text-align: center; padding: 25px; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                .company-type-public {{ background: linear-gradient(45deg, #3498db, #2980b9); color: white; }}
                .company-type-private {{ background: linear-gradient(45deg, #2ecc71, #27ae60); color: white; }}
                .competitor-badge {{ background-color: #f39c12; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; }}
                .similarity-high {{ background-color: #e8f5e8; }}
                .similarity-medium {{ background-color: #fff3cd; }}
                .similarity-low {{ background-color: #f8d7da; }}
                .description-box {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="header text-center">
                <div class="container">
                    <h1><i class="fas fa-building"></i> Universal Company Comparison Engine</h1>
                    <h2>{self.target_company_name}</h2>
                    <div class="row justify-content-center mt-3">
                        <div class="col-md-8">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="{'company-type-public' if target_data['is_public'] else 'company-type-private'} p-3 rounded">
                                        <i class="fas fa-{'chart-line' if target_data['is_public'] else 'seedling'}"></i>
                                        <br><strong>{'Public Company' if target_data['is_public'] else 'Private Company'}</strong>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="bg-white text-dark p-3 rounded">
                                        <i class="fas fa-industry"></i>
                                        <br><strong>{target_data['sector'] or 'Unknown Sector'}</strong>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="bg-white text-dark p-3 rounded">
                                        <i class="fas fa-map-marker-alt"></i>
                                        <br><strong>{target_data['location'] or 'Unknown Location'}</strong>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="bg-white text-dark p-3 rounded">
                                        <i class="fas fa-calendar-alt"></i>
                                        <br><strong>{target_data['founding_year'] if target_data['founding_year'] else 'Unknown'}</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="container">
                <div class="row">
                    <div class="col-md-4">
                        <div class="card border-primary">
                            <div class="card-header bg-primary text-white">
                                <h5><i class="fas fa-chart-bar"></i> Analysis Summary</h5>
                            </div>
                            <div class="card-body">
                                <div class="metric-card bg-light mb-3 rounded">
                                    <div class="metric-value">{len(results) - 1}</div>
                                    <div class="metric-label">Comparable Companies Found</div>
                                </div>
                                <div class="metric-card bg-light mb-3 rounded">
                                    <div class="metric-value">{top_comps.iloc[0]['Similarity_Score']:.3f}</div>
                                    <div class="metric-label">Highest Similarity Score</div>
                                </div>
                                <div class="metric-card bg-light mb-3 rounded">
                                    <div class="metric-value">{len(results[results['Type'] == 'Public'])}</div>
                                    <div class="metric-label">Public Companies</div>
                                </div>
                                <div class="metric-card bg-warning text-dark rounded">
                                    <div class="metric-value">{len(results[results['Type'] == 'Private'])}</div>
                                    <div class="metric-label">Private Companies</div>
                                </div>
                            </div>
                        </div>
                        <div class="card border-info">
                            <div class="card-header bg-info text-white">
                                <h5><i class="fas fa-info-circle"></i> Target Company Info</h5>
                            </div>
                            <div class="card-body">
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item"><strong>Industry:</strong> {target_data['industry'] or 'Not specified'}</li>
        """
        if target_data.get('employee_count'):
            html_content += f"<li class='list-group-item'><strong>Employees:</strong> {target_data['employee_count']:,}</li>"
        if target_data.get('market_cap'):
            html_content += f"<li class='list-group-item'><strong>Market Cap:</strong> ${target_data['market_cap']:,.0f}</li>"
        elif target_data.get('valuation'):
            html_content += f"<li class='list-group-item'><strong>Valuation:</strong> ${target_data['valuation']:,.0f}</li>"
        if target_data.get('funding'):
            html_content += f"<li class='list-group-item'><strong>Funding:</strong> ${target_data['funding']:,.0f}</li>"
        if target_data.get('website'):
            html_content += f"<li class='list-group-item'><strong>Website:</strong> <a href='{target_data['website']}' target='_blank'>{target_data['website']}</a></li>"
        html_content += f"""
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <div class="card border-success">
                            <div class="card-header bg-success text-white">
                                <h5><i class="fas fa-users"></i> Most Similar Companies</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
                                    <table class="table table-hover">
                                        <thead class="table-dark">
                                            <tr>
                                                <th>Company</th>
                                                <th>Similarity</th>
                                                <th>Type</th>
                                                <th>Industry</th>
                                                <th>Location</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
        """
        for _, row in top_comps.iterrows():
            similarity_class = 'similarity-high' if row['Similarity_Score'] > 0.7 else 'similarity-medium' if row['Similarity_Score'] > 0.4 else 'similarity-low'
            competitor_badge = '<span class="competitor-badge">Direct Competitor</span>' if row['Is_Direct_Competitor'] else ''
            html_content += f"""
                                            <tr class="{similarity_class}">
                                                <td><strong>{row['Company']}</strong></td>
                                                <td>{row['Similarity_Score']:.3f}</td>
                                                <td><span class="badge {'bg-primary' if row['Type'] == 'Public' else 'bg-success'}">{row['Type']}</span></td>
                                                <td>{row['Industry'] or 'N/A'}</td>
                                                <td>{row['Location'] or 'N/A'}</td>
                                                <td>{competitor_badge}</td>
                                            </tr>
            """
        html_content += f"""
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h5><i class="fas fa-project-diagram"></i> Company Network Analysis</h5>
                            </div>
                            <div class="card-body text-center">
                                <img src="universal_comps_network.png" class="img-fluid rounded shadow" style="max-height: 600px;">
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h5><i class="fas fa-chart-pie"></i> Analysis Distribution</h5>
                            </div>
                            <div class="card-body text-center">
                                <img src="universal_comps_analysis.png" class="img-fluid rounded shadow">
                            </div>
                        </div>
                    </div>
                </div>
        """
        if target_data['description']:
            html_content += f"""
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card border-info">
                            <div class="card-header bg-info text-white">
                                <h5><i class="fas fa-file-alt"></i> Company Description</h5>
                            </div>
                            <div class="card-body">
                                <div class="description-box">
                                    {target_data['description'][:1000]}{'...' if len(target_data['description']) > 1000 else ''}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        ai_status = "✅ Enabled" if use_gemini else "⚠️ Disabled (using web scraping fallback)"
        html_content += f"""
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card border-warning">
                            <div class="card-header bg-warning text-dark">
                                <h5><i class="fas fa-cogs"></i> Analysis Methodology</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6><i class="fas fa-search"></i> Data Sources:</h6>
                                        <ul>
                                            <li>Gemini AI for competitor discovery ({ai_status})</li>
                                            <li>Web scraping for company information</li>
                                            <li>Yahoo Finance for public companies</li>
                                            <li>Business description analysis</li>
                                            <li>Industry and sector classification</li>
                                        </ul>
                                    </div>
                                    <div class="col-md-6">
                                        <h6><i class="fas fa-brain"></i> Analysis Features:</h6>
                                        <ul>
                                            <li>Natural Language Processing (NLP) similarity</li>
                                            <li>Company attribute matching</li>
                                            <li>Sector and industry alignment</li>
                                            <li>Geographic proximity</li>
                                            <li>Company size and type compatibility</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="alert alert-success mt-3">
                                    <i class="fas fa-check-circle"></i> <strong>Universal Coverage:</strong> This engine works with any company - public, private, startups, or established businesses. No ticker symbol required!
                                </div>
                                <div class="alert alert-info mt-3">
                                    <i class="fas fa-robot"></i> <strong>AI-Powered:</strong> Competitor identification powered by Google's Gemini AI for accurate and comprehensive results.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <footer class="bg-dark text-light text-center py-4 mt-5">
                <div class="container">
                    <p><strong><i class="fas fa-rocket"></i> Universal Company Comparison Engine</strong> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><small>Works with ANY company - Public • Private • Startups • No ticker required</small></p>
                </div>
            </footer>
        </body>
        </html>
        """
        with open('universal_comps_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        # Save summary CSV
        summary_df = top_comps[['Company', 'Similarity_Score', 'Type', 'Sector', 'Industry', 'Location', 'Is_Direct_Competitor']].copy()
        summary_df.to_csv('universal_comps_summary.csv', index=False)
        print("✅ Generated universal dashboard: universal_comps_dashboard.html")
        print("✅ Generated summary: universal_comps_summary.csv")
# Main execution
if __name__ == "__main__":
    print("🚀 Universal Company Comparison Engine")
    print("=" * 60)
    print("Works with ANY company - Public, Private, Startups, etc.")
    print("No ticker symbol required!")
    print("AI-Powered by Google Gemini")
    print("=" * 60)
    # Get company name from user
    target_company = input("\nEnter any company name (e.g., 'Airbnb', 'SpaceX', 'Apple', 'Local Restaurant'): ").strip()
    if not target_company:
        print("❌ Please enter a valid company name")
        exit(1)
    # Initialize engine
    engine = UniversalCompsEngine(target_company_name=target_company)
    # Collect data
    print(f"\n🔍 Starting comprehensive analysis for '{target_company}'...")
    print("This may take a few minutes as we research online...")
    try:
        engine.collect_data(num_companies=15)
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        exit(1)
    if len(engine.companies) < 2:
        print("❌ Could not find enough comparable companies. Try a more well-known company name.")
        exit(1)
    # Perform analysis
    print("\n🧠 Analyzing company similarities...")
    try:
        results = engine.analyze_comparables()
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        exit(1)
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    engine.visualize_results(results)
    # Generate dashboard
    print("\n🎨 Building comprehensive dashboard...")
    engine.generate_dashboard(results)
    # Display results
    print(f"\n🏆 Most Similar Companies to {target_company}:")
    print("=" * 80)
    top_results = results[results['Company'] != target_company].head(8)
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        company_type = f"{'📈' if row['Type'] == 'Public' else '🏢'} {row['Type']}"
        competitor_flag = " 🎯" if row['Is_Direct_Competitor'] else ""
        print(f"{i:2d}. {row['Company']:<25} | Similarity: {row['Similarity_Score']:.3f} | {company_type}{competitor_flag}")
        if row['Industry']:
            print(f"     Industry: {row['Industry']}")
        print()
    print("✅ Universal analysis complete!")
    print("\n📁 Generated files:")
    print("   • universal_comps_dashboard.html (comprehensive dashboard)")
    print("   • universal_comps_summary.csv (data export)")
    print("   • universal_comps_network.png (company network)")
    print("   • universal_comps_analysis.png (analysis charts)")
    print(f"\n🌟 This engine analyzed {target_company} without needing a ticker symbol!")
    if use_gemini:
        print("🤖 Competitor discovery powered by Google's Gemini AI")
    else:
        print("🔍 Competitor discovery using web scraping (Gemini unavailable)")
    print("🔍 Detailed company information gathered through web research")
    if engine.is_public_company:
        print(f"📈 Detected as public company (ticker: {engine.company_data[target_company]['ticker']})")
    else:
        print("🏢 Detected as private company - analysis based on business intelligence")