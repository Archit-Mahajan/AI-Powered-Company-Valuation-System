import os
import json
import threading
import io
import sys
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import re
# Import your analysis modules
from forecast import EnhancedMakeMyTripForecaster, main as enhanced_main
from dcf_value import main
# Import Gemini-related modules with error handling
try:
    from sentiment_analysis import generate_qualitative_report
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sentiment_analysis: {e}")
    GEMINI_AVAILABLE = False
try:
    from comps import UniversalCompsEngine
except ImportError as e:
    print(f"Warning: Could not import comps: {e}")
    UniversalCompsEngine = None
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
# Global variables to store analysis results
analysis_results = {
    'dcf': None,
    'enhanced': None,
    'qualitative': None,
    'comps': None
}
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_dcf', methods=['POST'])
def run_dcf():
    data = request.json
    ticker = data.get('ticker', 'MMYT')
    
    # Run DCF analysis in a thread to avoid blocking
    def dcf_thread():
        original_dir = os.getcwd()
        try:
            # Change to the script directory to ensure proper file paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            # Set up the environment with API keys
            env = os.environ.copy()
            env['FRED_API_KEY'] = 'ef818507901a73b2dc8cb3b1fe3e3184'
            env['FMP_API_KEY'] = 'UrGqphhSmbOap5fdmRaHtU8Nt83V5US6'
            env['TIINGO_API_KEY'] = '2c7f5a1d68183990481a93d1a880010da411a873'
            env['ALPHAVANTAGE_API_KEY'] = 'AOVG7UW53408H5IC'
            env['FINNHUB_API_KEY'] = 'd2dp2k1r01qjrul4cnk0d2dp2k1r01qjrul4cnk'
            
            # Use the specific Python interpreter that has yfinance installed
            python_path = '/Users/architmahajan/.pyenv/versions/3.10.12/bin/python3.10'
            
            # Run DCF analysis as a subprocess
            cmd = [
                python_path, 'dcf_value.py',
                '--ticker', ticker,
                '--indir', './out',
                '--outdir', './out',
                '--fred_key', env['FRED_API_KEY'],
                '--fmp_key', env['FMP_API_KEY'],
                '--tiingo_key', env['TIINGO_API_KEY'],
                '--alphav_key', env['ALPHAVANTAGE_API_KEY'],
                '--finnhub_key', env['FINNHUB_API_KEY']
            ]
            
            print(f"Running DCF command: {' '.join(cmd)}")
            
            # Run the subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            
            print(f"DCF process completed with return code: {result.returncode}")
            print(f"DCF stdout: {result.stdout[:500]}...")  # First 500 chars
            if result.stderr:
                print(f"DCF stderr: {result.stderr[:500]}...")  # First 500 chars
            
            if result.returncode != 0:
                analysis_results['dcf'] = {'error': f"DCF analysis failed: {result.stderr}"}
                return
            
            # Load results
            dcf_file = f'./out/dcf_valuation_{ticker}_scenarios.json'
            if os.path.exists(dcf_file):
                with open(dcf_file, 'r') as f:
                    dcf_data = json.load(f)
                
                # Parse the stdout to extract market cap and other key metrics
                stdout_lines = result.stdout.split('\n')
                market_cap = None
                implied_upside = None
                
                for line in stdout_lines:
                    if 'Market Cap (normalized):' in line:
                        try:
                            market_cap = float(line.split(':')[1].strip().replace(',', ''))
                        except:
                            pass
                    elif 'Implied Upside vs Mkt Cap:' in line:
                        try:
                            implied_upside_str = line.split(':')[1].strip().replace('%', '')
                            implied_upside = float(implied_upside_str) / 100
                        except:
                            pass
                
                # Add the extracted information to the DCF data
                if market_cap:
                    dcf_data['market_cap'] = market_cap
                if implied_upside is not None:
                    dcf_data['implied_upside_vs_mcap'] = implied_upside
                
                # Also parse other key metrics from stdout
                for line in stdout_lines:
                    if 'Weighted EV:' in line:
                        try:
                            weighted_ev = float(line.split(':')[1].strip().split('|')[0].replace(',', ''))
                            dcf_data['ev_weighted'] = weighted_ev
                        except:
                            pass
                    elif 'Confidence:' in line:
                        try:
                            confidence = line.split(':')[1].strip().split('%')[0]
                            dcf_data['confidence'] = f"{confidence}%"
                        except:
                            pass
                
                analysis_results['dcf'] = dcf_data
                print(f"Successfully loaded DCF results from {dcf_file}")
            else:
                analysis_results['dcf'] = {'error': f'DCF results file not found: {dcf_file}'}
                print(f"DCF results file not found: {dcf_file}")
            
        except subprocess.TimeoutExpired:
            error_msg = 'DCF analysis timed out (5 minutes)'
            print(error_msg)
            analysis_results['dcf'] = {'error': error_msg}
        except Exception as e:
            error_msg = f"Unexpected error in DCF analysis: {str(e)}"
            print(error_msg)
            analysis_results['dcf'] = {'error': error_msg}
        finally:
            # Always restore the original directory
            os.chdir(original_dir)
    
    thread = threading.Thread(target=dcf_thread)
    thread.start()
    thread.join()
    
    return jsonify(analysis_results['dcf'])
@app.route('/run_enhanced', methods=['POST'])
def run_enhanced():
    data = request.json
    ticker = data.get('ticker', 'MMYT')
    def enhanced_thread():
        original_dir = os.getcwd()
        try:
            # Change to the script directory to ensure proper file paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            # Create analyzer instance
            analyzer = EnhancedMakeMyTripForecaster("AIzaSyCT5fJO0UlpLv-HuYp7JFlC2U1XmGGMK4A")
            success = analyzer.run_complete_enhanced_analysis()
            if success:
                # Load results
                enhanced_file = f'./mmyt_enhanced_revenue_report.txt'
                if os.path.exists(enhanced_file):
                    with open(enhanced_file, 'r') as f:
                        report_content = f.read()
                    # Parse the report to extract key metrics
                    key_metrics = parse_enhanced_report(report_content)
                    analysis_results['enhanced'] = {
                        'report': report_content,
                        'key_metrics': key_metrics
                    }
                else:
                    analysis_results['enhanced'] = {'error': 'Enhanced analysis report not found'}
            else:
                analysis_results['enhanced'] = {'error': 'Enhanced analysis failed'}
        except Exception as e:
            analysis_results['enhanced'] = {'error': str(e)}
        finally:
            # Always restore the original directory
            os.chdir(original_dir)
    thread = threading.Thread(target=enhanced_thread)
    thread.start()
    thread.join()
    return jsonify(analysis_results['enhanced'])
def parse_enhanced_report(report_content):
    """Parse the enhanced report to extract key metrics"""
    key_metrics = {}
    # Extract forecast values
    forecast_pattern = r'• (\d{4}): \$([\d\.]+)M'
    forecasts = re.findall(forecast_pattern, report_content)
    if forecasts:
        key_metrics['forecasts'] = {year: float(value) for year, value in forecasts}
    # Extract CAGR
    cagr_pattern = r'• 5-Year CAGR: ([\d\.]+)%'
    cagr_match = re.search(cagr_pattern, report_content)
    if cagr_match:
        key_metrics['cagr'] = float(cagr_match.group(1))
    # Extract Monte Carlo results
    mc_patterns = {
        'mean': r'• Mean: \$([\d\.]+)M',
        'std': r'• Standard Deviation: \$([\d\.]+)M',
        'percentile_10': r'• 80% Confidence Interval: \$([\d\.]+)M - \$([\d\.]+)M',
        'percentile_90': r'• 80% Confidence Interval: \$[\d\.]+M - \$([\d\.]+)M'
    }
    for key, pattern in mc_patterns.items():
        if key == 'percentile_10':
            matches = re.search(pattern, report_content)
            if matches:
                key_metrics['mc_low'] = float(matches.group(1))
                key_metrics['mc_high'] = float(matches.group(2))
        else:
            match = re.search(pattern, report_content)
            if match:
                key_metrics[f'mc_{key}'] = float(match.group(1))
    # Extract period-specific growth rates
    period_patterns = {
        'full_period': r'Full Period:.*?Mean Growth: ([\d\.]+)%',
        'covid_period': r'Covid Period:.*?Mean Growth: ([\d\.\-]+)%',
        'recovery_period': r'Recovery Period:.*?Mean Growth: ([\d\.]+)%'
    }
    for key, pattern in period_patterns.items():
        match = re.search(pattern, report_content, re.DOTALL)
        if match:
            key_metrics[key] = float(match.group(1))
    # Extract model performance
    model_pattern = r'🏆 BEST PERFORMING MODEL: ([A-Z\s]+)'
    model_match = re.search(model_pattern, report_content)
    if model_match:
        key_metrics['best_model'] = model_match.group(1).strip()
    # Extract MAPE
    mape_pattern = r'• MAPE: ([\d\.]+)%'
    mape_match = re.search(mape_pattern, report_content)
    if mape_match:
        key_metrics['mape'] = float(mape_match.group(1))
    return key_metrics
@app.route('/run_qualitative', methods=['POST'])
def run_qualitative():
    data = request.json
    company = data.get('company', 'MakeMyTrip')
    def qualitative_thread():
        original_dir = os.getcwd()
        try:
            # Change to the script directory to ensure proper file paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            if not GEMINI_AVAILABLE:
                analysis_results['qualitative'] = {
                    'report': f"Qualitative analysis unavailable due to API limitations.\n\nCompany: {company}\nPlease check your Gemini API quota or try again later.",
                    'api_error': True
                }
                return
            # Import and run the sentiment analysis
            from sentiment_analysis import generate_qualitative_report
            # Capture the logging output by temporarily redirecting the logger
            import logging
            from io import StringIO
            # Create a string buffer to capture log messages
            log_capture_string = StringIO()
            ch = logging.StreamHandler(log_capture_string)
            ch.setLevel(logging.INFO)
            # Get the sentiment_analysis logger
            sentiment_logger = logging.getLogger('sentiment_analysis')
            sentiment_logger.addHandler(ch)
            sentiment_logger.setLevel(logging.INFO)
            # Also capture the root logger if needed
            root_logger = logging.getLogger()
            root_logger.addHandler(ch)
            # Run the analysis
            generate_qualitative_report(company)
            # Get the captured log content
            log_contents = log_capture_string.getvalue()
            # Remove the handler
            sentiment_logger.removeHandler(ch)
            root_logger.removeHandler(ch)
            # Parse the log contents to extract structured data
            sentiment_data = {}
            # Extract sentiment counts from logs
            if "Positive:" in log_contents:
                pos_match = re.search(r'Positive: (\d+) samples', log_contents)
                neg_match = re.search(r'Negative: (\d+) samples', log_contents)
                neu_match = re.search(r'Neutral: (\d+) samples', log_contents)
                if pos_match and neg_match and neu_match:
                    positive = int(pos_match.group(1))
                    negative = int(neg_match.group(1))
                    neutral = int(neu_match.group(1))
                    total = positive + negative + neutral
                    sentiment_data = {
                        'positive': positive,
                        'negative': negative,
                        'neutral': neutral,
                        'total': total,
                        'sentiment_score': ((positive - negative) / total * 100) if total > 0 else 0
                    }
            # Check if sentiment distribution image was created
            chart_available = os.path.exists('sentiment_distribution.png')
            # Create a comprehensive report
            report_sections = []
            if sentiment_data:
                report_sections.append("=== SENTIMENT ANALYSIS RESULTS ===")
                report_sections.append(f"Total samples analyzed: {sentiment_data['total']:,}")
                report_sections.append(f"Positive sentiment: {sentiment_data['positive']:,} ({sentiment_data['positive']/sentiment_data['total']*100:.1f}%)")
                report_sections.append(f"Negative sentiment: {sentiment_data['negative']:,} ({sentiment_data['negative']/sentiment_data['total']*100:.1f}%)")
                report_sections.append(f"Neutral sentiment: {sentiment_data['neutral']:,} ({sentiment_data['neutral']/sentiment_data['total']*100:.1f}%)")
                report_sections.append(f"Overall sentiment score: {sentiment_data['sentiment_score']:.1f}")
                report_sections.append("")
            # Extract GitHub data from logs
            if "Innovation & Traction Report" in log_contents:
                report_sections.append("=== INNOVATION & TRACTION (GITHUB) ===")
                stars_match = re.search(r'Stars \(Top Repos\): (\d+)', log_contents)
                forks_match = re.search(r'Forks \(Top Repos\): (\d+)', log_contents)
                traction_match = re.search(r'Estimated Traction Score: ([\d.]+)', log_contents)
                if stars_match:
                    report_sections.append(f"GitHub Stars (Top Repositories): {stars_match.group(1)}")
                if forks_match:
                    report_sections.append(f"GitHub Forks (Top Repositories): {forks_match.group(1)}")
                if traction_match:
                    report_sections.append(f"Traction Score: {traction_match.group(1)}/100")
                report_sections.append("")
            # Extract LinkedIn data from logs if available
            if "Team Strength Report" in log_contents:
                report_sections.append("=== TEAM STRENGTH (LINKEDIN) ===")
                emp_match = re.search(r'Estimated Employee Count: (\d+)', log_contents)
                team_match = re.search(r'Estimated Team Strength Score: ([\d.]+)', log_contents)
                if emp_match:
                    report_sections.append(f"Employee Count: {emp_match.group(1)}")
                if team_match:
                    report_sections.append(f"Team Strength Score: {team_match.group(1)}/100")
                report_sections.append("")
            # Add any error messages or warnings
            if "❌" in log_contents or "⚠️" in log_contents:
                report_sections.append("=== NOTES & WARNINGS ===")
                for line in log_contents.split('\n'):
                    if "❌" in line or "⚠️" in line:
                        # Clean up the log line
                        clean_line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - ', '', line)
                        if clean_line.strip():
                            report_sections.append(clean_line.strip())
                report_sections.append("")
            final_report = "\n".join(report_sections)
            if not final_report.strip():
                final_report = f"Analysis completed for {company}, but no structured data was extracted.\n\nRaw log output:\n{log_contents}"
            analysis_results['qualitative'] = {
                'report': final_report,
                'sentiment_data': sentiment_data if sentiment_data else None,
                'chart_available': chart_available,
                'raw_logs': log_contents  # For debugging
            }
        except Exception as e:
            error_msg = str(e)
            analysis_results['qualitative'] = {
                'error': error_msg,
                'api_error': 'Gemini' in error_msg or 'API' in error_msg or 'quota' in error_msg.lower()
            }
        finally:
            # Always restore the original directory
            os.chdir(original_dir)
    thread = threading.Thread(target=qualitative_thread)
    thread.start()
    thread.join()
    return jsonify(analysis_results['qualitative'])
@app.route('/run_comps', methods=['POST'])
def run_comps():
    data = request.json
    company = data.get('company', 'MakeMyTrip')
    def comps_thread():
        original_dir = os.getcwd()
        try:
            # Change to the script directory to ensure proper file paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            if UniversalCompsEngine is None:
                analysis_results['comps'] = {
                    'error': 'Comparable companies analysis unavailable due to missing dependencies',
                    'api_error': True
                }
                return
            engine = UniversalCompsEngine(company)
            engine.collect_data(num_companies=15)
            results = engine.analyze_comparables()
            engine.visualize_results(results)
            engine.generate_dashboard(results)
            # Load results
            comps_file = f'./universal_comps_summary.csv'
            if os.path.exists(comps_file):
                df = pd.read_csv(comps_file)
                analysis_results['comps'] = {
                    'summary': df.to_dict('records'),
                    'dashboard': 'universal_comps_dashboard.html'
                }
            else:
                analysis_results['comps'] = {'error': 'Comps analysis results not found'}
        except Exception as e:
            analysis_results['comps'] = {
                'error': str(e),
                'api_error': 'Gemini' in str(e) or 'API' in str(e) or 'quota' in str(e).lower()
            }
        finally:
            # Always restore the original directory
            os.chdir(original_dir)
    thread = threading.Thread(target=comps_thread)
    thread.start()
    thread.join()
    return jsonify(analysis_results['comps'])
@app.route('/get_results')
def get_results():
    return jsonify(analysis_results)
@app.route('/sentiment_distribution.png')
def serve_sentiment_image():
    """Serve the sentiment distribution image if it exists"""
    try:
        return send_file('sentiment_distribution.png', mimetype='image/png')
    except FileNotFoundError:
        return "Image not found", 404
@app.route('/enhanced_analysis.png')
def serve_enhanced_image():
    """Serve the enhanced analysis image if it exists"""
    try:
        return send_file('mmyt_enhanced_revenue_analysis.png', mimetype='image/png')
    except FileNotFoundError:
        return "Image not found", 404
# Add a simple fallback comps analysis
def simple_comps_analysis(company):
    """Fallback comparable companies analysis when Gemini is unavailable"""
    # This would typically use alternative data sources
    return [
        {
            'company': company,
            'metric': 'P/E Ratio',
            'value': 'N/A',
            'industry_avg': 'N/A',
            'note': 'API unavailable'
        },
        {
            'company': company,
            'metric': 'P/B Ratio',
            'value': 'N/A',
            'industry_avg': 'N/A',
            'note': 'API unavailable'
        }
    ]
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('./out', exist_ok=True)
    os.makedirs('./static', exist_ok=True)
    os.makedirs('./templates', exist_ok=True)
    # Create basic templates with improved DCF display
    with open('./templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --danger-gradient: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }
        .dashboard-header {
            background: var(--primary-gradient);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .analysis-section {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }
        .analysis-section:hover {
            transform: translateY(-5px);
        }
        .section-title {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e9ecef;
        }
        .control-panel {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            position: sticky;
            top: 20px;
        }
        .btn-analysis {
            width: 100%;
            margin-bottom: 0.8rem;
            padding: 0.8rem;
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .btn-analysis:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .metric-card {
            background: var(--primary-gradient);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .forecast-card {
            background: var(--info-gradient);
        }
        .monte-carlo-card {
            background: var(--warning-gradient);
        }
        .cagr-card {
            background: var(--success-gradient);
        }
        .sentiment-card {
            background: var(--warning-gradient);
        }
        .upside-positive {
            color: #28a745;
            font-weight: bold;
        }
        .upside-negative {
            color: #dc3545;
            font-weight: bold;
        }
        .results-container {
            min-height: 200px;
            padding: 1rem;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        .api-warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 2rem;
        }
        .nav-tabs .nav-link {
            color: #6c757d;
            font-weight: 500;
        }
        .nav-tabs .nav-link.active {
            color: #495057;
            font-weight: 600;
        }
        .tab-content {
            padding-top: 1.5rem;
        }
        .scenario-table {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }
        .scenario-table thead {
            background: var(--primary-gradient);
            color: white;
        }
        .scenario-table th {
            border: none;
            padding: 1rem;
            font-weight: 600;
        }
        .scenario-table td {
            padding: 1rem;
            vertical-align: middle;
        }
        .table-success td {
            background-color: rgba(40, 167, 69, 0.1);
        }
        .table-danger td {
            background-color: rgba(220, 53, 69, 0.1);
        }
        .table-info td {
            background-color: rgba(23, 162, 184, 0.1);
        }
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
        }
        .accordion-button:not(.collapsed) {
            background-color: #f8f9fa;
            color: #495057;
        }
        .accordion-body {
            background-color: #f8f9fa;
        }
        .forecast-highlight {
            background: linear-gradient(90deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
            border-left: 4px solid #4facfe;
            padding: 1rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 1rem;
        }
        .monte-carlo-highlight {
            background: linear-gradient(90deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
            border-left: 4px solid #f093fb;
            padding: 1rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 1rem;
        }
        .model-performance-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }
        .badge-best-model {
            background: var(--success-gradient);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
        }
        .period-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            border-left: 4px solid #667eea;
        }
        .period-title {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        .growth-positive {
            color: #28a745;
            font-weight: 600;
        }
        .growth-negative {
            color: #dc3545;
            font-weight: 600;
        }
        @media (max-width: 768px) {
            .control-panel {
                position: relative;
                margin-bottom: 2rem;
            }
            .metric-value {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <h1 class="text-center mb-0">
                <i class="fas fa-chart-line me-3"></i>
                Comprehensive Financial Analysis Dashboard
            </h1>
            <p class="text-center mb-0 opacity-75">Advanced DCF, Enhanced Forecasting, Qualitative & Comparative Analysis</p>
        </div>
    </div>
    <div class="container">
        <div class="alert alert-warning mb-4" role="alert" id="apiWarning" style="display: none;">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>API Limitations:</strong> Some features may be unavailable due to API quota restrictions. 
            Please try again later or check your API keys.
        </div>
        <div class="row">
            <div class="col-lg-3 col-md-4">
                <div class="control-panel">
                    <h5 class="mb-4">
                        <i class="fas fa-sliders-h me-2"></i>
                        Analysis Controls
                    </h5>
                    <div class="mb-3">
                        <label for="tickerInput" class="form-label">Ticker Symbol</label>
                        <input type="text" class="form-control" id="tickerInput" value="MMYT">
                    </div>
                    <div class="mb-3">
                        <label for="companyInput" class="form-label">Company Name</label>
                        <input type="text" class="form-control" id="companyInput" value="MakeMyTrip">
                    </div>
                    <button class="btn btn-primary btn-analysis" onclick="runAnalysis('dcf')">
                        <i class="fas fa-calculator me-2"></i>Run DCF Valuation
                    </button>
                    <button class="btn btn-success btn-analysis" onclick="runAnalysis('enhanced')">
                        <i class="fas fa-chart-area me-2"></i>Run Enhanced Analysis
                    </button>
                    <button class="btn btn-info btn-analysis" onclick="runAnalysis('qualitative')">
                        <i class="fas fa-comments me-2"></i>Run Qualitative Analysis
                    </button>
                    <button class="btn btn-warning btn-analysis" onclick="runAnalysis('comps')">
                        <i class="fas fa-balance-scale me-2"></i>Run Comps Analysis
                    </button>
                    <button class="btn btn-dark btn-analysis" onclick="runAllAnalyses()">
                        <i class="fas fa-play-circle me-2"></i>Run All Analyses
                    </button>
                </div>
            </div>
            <div class="col-lg-9 col-md-8">
                <div class="analysis-section">
                    <h3 class="section-title">
                        <i class="fas fa-calculator me-2"></i>
                        DCF Valuation
                    </h3>
                    <div id="dcfResults" class="results-container">
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-chart-pie fa-3x mb-3 opacity-25"></i>
                            <p>Click "Run DCF Valuation" to see results</p>
                        </div>
                    </div>
                </div>
                <div class="analysis-section">
                    <h3 class="section-title">
                        <i class="fas fa-chart-area me-2"></i>
                        Enhanced Financial Analysis
                    </h3>
                    <div id="enhancedResults" class="results-container">
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-chart-line fa-3x mb-3 opacity-25"></i>
                            <p>Click "Run Enhanced Analysis" to see results</p>
                        </div>
                    </div>
                </div>
                <div class="analysis-section">
                    <h3 class="section-title">
                        <i class="fas fa-comments me-2"></i>
                        Qualitative Analysis
                    </h3>
                    <div id="qualitativeResults" class="results-container">
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-comments fa-3x mb-3 opacity-25"></i>
                            <p>Click "Run Qualitative Analysis" to see results</p>
                        </div>
                    </div>
                </div>
                <div class="analysis-section">
                    <h3 class="section-title">
                        <i class="fas fa-balance-scale me-2"></i>
                        Comparable Companies Analysis
                    </h3>
                    <div id="compsResults" class="results-container">
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-balance-scale fa-3x mb-3 opacity-25"></i>
                            <p>Click "Run Comps Analysis" to see results</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        function runAnalysis(type) {
            const ticker = document.getElementById('tickerInput').value;
            const company = document.getElementById('companyInput').value;
            // Show loading state
            document.getElementById(`${type}Results`).innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
            fetch(`/run_${type}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker, company })
            })
            .then(response => response.json())
            .then(data => {
                displayResults(type, data);
            })
            .catch(error => {
                document.getElementById(`${type}Results`).innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error: ${error}
                    </div>
                `;
            });
        }
        function runAllAnalyses() {
            runAnalysis('dcf');
            runAnalysis('enhanced');
            runAnalysis('qualitative');
            runAnalysis('comps');
        }
        function displayResults(type, data) {
            let html = '';
            if (data && data.error) {
                const isApiError = data.api_error || data.error.includes('API') || data.error.includes('Gemini') || data.error.includes('quota');
                if (isApiError) {
                    document.getElementById('apiWarning').style.display = 'block';
                }
                html = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error: ${data.error}
                    </div>
                `;
            } else if (data && data.api_error) {
                document.getElementById('apiWarning').style.display = 'block';
                html = `
                    <div class="api-warning p-3">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${data.report || data.error}
                    </div>
                `;
            } else if (data) {
                switch(type) {
                    case 'dcf':
                        html = formatDCFResults(data);
                        break;
                    case 'enhanced':
                        html = formatEnhancedResults(data);
                        break;
                    case 'qualitative':
                        html = formatQualitativeResults(data);
                        break;
                    case 'comps':
                        html = formatCompsResults(data);
                        break;
                }
            } else {
                html = `
                    <div class="alert alert-warning">
                        <i class="fas fa-info-circle me-2"></i>
                        No data returned from analysis
                    </div>
                `;
            }
            document.getElementById(`${type}Results`).innerHTML = html;
            // Initialize any charts after rendering
            if (type === 'enhanced' && data && data.key_metrics) {
                initializeCharts(data.key_metrics);
            }
        }
        function formatDCFResults(data) {
            let html = '<h4 class="mb-4">DCF Valuation Results</h4>';
            // Key metrics cards at the top
            html += '<div class="row mb-4">';
            if (data.market_cap_observed) {
                let marketCapFormatted;
                if (data.market_cap_observed >= 1e9) {
                    marketCapFormatted = `$${(data.market_cap_observed / 1e9).toFixed(2)}B`;
                } else {
                    marketCapFormatted = `$${(data.market_cap_observed / 1e6).toFixed(0)}M`;
                }
                html += '<div class="col-md-4">';
                html += '<div class="metric-card text-center">';
                html += '<div class="metric-label">Current Market Cap</div>';
                html += `<div class="metric-value">${marketCapFormatted}</div>`;
                html += '</div></div>';
            }
            if (data.ev_weighted) {
                let evFormatted;
                if (data.ev_weighted >= 1e9) {
                    evFormatted = `$${(data.ev_weighted / 1e9).toFixed(2)}B`;
                } else {
                    evFormatted = `$${(data.ev_weighted / 1e6).toFixed(0)}M`;
                }
                html += '<div class="col-md-4">';
                html += '<div class="metric-card text-center" style="background: var(--success-gradient);">';
                html += '<div class="metric-label">Weighted Enterprise Value</div>';
                html += `<div class="metric-value">${evFormatted}</div>`;
                html += '</div></div>';
            }
            if (data.implied_upside_vs_mcap !== undefined) {
                const upside = data.implied_upside_vs_mcap * 100;
                const upsideClass = upside >= 0 ? 'upside-positive' : 'upside-negative';
                const bgColor = upside >= 0 ? 'var(--success-gradient)' : 'var(--danger-gradient)';
                html += '<div class="col-md-4">';
                html += `<div class="metric-card text-center" style="background: ${bgColor};">`;
                html += '<div class="metric-label">Implied Upside</div>';
                html += `<div class="metric-value ${upsideClass}">${upside.toFixed(1)}%</div>`;
                html += '</div></div>';
            }
            html += '</div>';
            // Scenario analysis table
            if (data.scenarios) {
                html += '<div class="row"><div class="col-md-8">';
                html += '<h5 class="mb-3">Scenario Analysis</h5>';
                html += '<div class="table-responsive scenario-table">';
                html += '<table class="table table-hover mb-0">';
                html += '<thead><tr><th>Scenario</th><th>Enterprise Value</th><th>Value (Millions)</th></tr></thead><tbody>';
                for (const [scenario, details] of Object.entries(data.scenarios)) {
                    const ev = details.enterprise_value || 0;
                    const scenarioClass = scenario === 'Bull' ? 'table-success' : scenario === 'Bear' ? 'table-danger' : 'table-info';
                    html += `<tr class="${scenarioClass}">`;
                    html += `<td><strong>${scenario}</strong></td>`;
                    html += `<td>$${ev.toLocaleString()}</td>`;
                    html += `<td>$${(ev / 1000000).toFixed(0)}M</td>`;
                    html += '</tr>';
                }
                html += '</tbody></table></div></div>';
                // Additional metrics panel
                html += '<div class="col-md-4">';
                html += '<div class="card">';
                html += '<div class="card-header bg-info text-white">';
                html += '<h6 class="mb-0">Analysis Details</h6>';
                html += '</div>';
                html += '<div class="card-body">';
                if (data.confidence_score) {
                    html += `<p><strong>Model Confidence:</strong><br><span class="badge bg-secondary">${(data.confidence_score * 100).toFixed(0)}%</span></p>`;
                }
                if (data.market_cap_observed && data.ev_weighted) {
                    const premium = ((data.ev_weighted - data.market_cap_observed) / data.market_cap_observed * 100);
                    html += `<p><strong>EV Premium:</strong><br><span class="badge ${premium >= 0 ? 'bg-success' : 'bg-danger'}">${premium.toFixed(1)}%</span></p>`;
                }
                html += '</div></div></div>';
                html += '</div>';
            } else if (data.error) {
                html += `<div class="alert alert-warning">${data.error}</div>`;
            } else {
                html += '<div class="alert alert-info">No DCF data available</div>';
            }
            return html;
        }
        function formatEnhancedResults(data) {
            let html = '<h4 class="mb-4">Enhanced Financial Analysis</h4>';
            if (data.key_metrics) {
                const metrics = data.key_metrics;
                // Create tabs for different sections
                html += `
                    <ul class="nav nav-tabs" id="enhancedTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="forecast-tab" data-bs-toggle="tab" data-bs-target="#forecast" type="button" role="tab">
                                <i class="fas fa-chart-line me-2"></i>Forecast
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="montecarlo-tab" data-bs-toggle="tab" data-bs-target="#montecarlo" type="button" role="tab">
                                <i class="fas fa-dice me-2"></i>Monte Carlo
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="performance-tab" data-bs-toggle="tab" data-bs-target="#performance" type="button" role="tab">
                                <i class="fas fa-tachometer-alt me-2"></i>Performance
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="periods-tab" data-bs-toggle="tab" data-bs-target="#periods" type="button" role="tab">
                                <i class="fas fa-calendar-alt me-2"></i>Periods
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="report-tab" data-bs-toggle="tab" data-bs-target="#report" type="button" role="tab">
                                <i class="fas fa-file-alt me-2"></i>Full Report
                            </button>
                        </li>
                    </ul>
                    <div class="tab-content" id="enhancedTabsContent">
                `;
                // Forecast Tab
                html += `
                    <div class="tab-pane fade show active" id="forecast" role="tabpanel">
                        <div class="row mb-4">
                `;
                if (metrics.cagr) {
                    html += '<div class="col-md-4">';
                    html += '<div class="metric-card cagr-card text-center">';
                    html += '<div class="metric-label">5-Year CAGR</div>';
                    html += `<div class="metric-value">${metrics.cagr.toFixed(1)}%</div>`;
                    html += '</div></div>';
                }
                if (metrics.forecasts) {
                    const lastYear = Object.keys(metrics.forecasts).pop();
                    const lastValue = metrics.forecasts[lastYear];
                    html += '<div class="col-md-4">';
                    html += '<div class="metric-card forecast-card text-center">';
                    html += '<div class="metric-label">2029 Forecast</div>';
                    html += `<div class="metric-value">$${lastValue.toFixed(0)}M</div>`;
                    html += '</div></div>';
                }
                if (metrics.best_model) {
                    html += '<div class="col-md-4">';
                    html += '<div class="metric-card text-center" style="background: var(--warning-gradient);">';
                    html += '<div class="metric-label">Best Model</div>';
                    html += `<div class="metric-value">${metrics.best_model}</div>`;
                    html += '</div></div>';
                }
                html += '</div>';
                // Forecast chart
                if (metrics.forecasts) {
                    html += '<div class="forecast-highlight">';
                    html += '<h5 class="mb-3"><i class="fas fa-chart-line me-2"></i>Revenue Forecast (2026-2029)</h5>';
                    html += '<div class="chart-container">';
                    html += '<canvas id="forecastChart"></canvas>';
                    html += '</div></div>';
                }
                html += '</div>';
                // Monte Carlo Tab
                html += `
                    <div class="tab-pane fade" id="montecarlo" role="tabpanel">
                        <div class="row mb-4">
                `;
                if (metrics.mc_mean) {
                    html += '<div class="col-md-4">';
                    html += '<div class="metric-card monte-carlo-card text-center">';
                    html += '<div class="metric-label">Monte Carlo Mean (2029)</div>';
                    html += `<div class="metric-value">$${metrics.mc_mean.toFixed(0)}M</div>`;
                    html += '</div></div>';
                }
                if (metrics.mc_std) {
                    html += '<div class="col-md-4">';
                    html += '<div class="metric-card text-center" style="background: var(--danger-gradient);">';
                    html += '<div class="metric-label">Standard Deviation</div>';
                    html += `<div class="metric-value">$${metrics.mc_std.toFixed(0)}M</div>`;
                    html += '</div></div>';
                }
                if (metrics.mc_low && metrics.mc_high) {
                    const range = metrics.mc_high - metrics.mc_low;
                    html += '<div class="col-md-4">';
                    html += '<div class="metric-card text-center" style="background: var(--info-gradient);">';
                    html += '<div class="metric-label">80% Range</div>';
                    html += `<div class="metric-value">$${range.toFixed(0)}M</div>`;
                    html += '</div></div>';
                }
                html += '</div>';
                // Monte Carlo chart
                if (metrics.mc_low && metrics.mc_high) {
                    html += '<div class="monte-carlo-highlight">';
                    html += '<h5 class="mb-3"><i class="fas fa-dice me-2"></i>Monte Carlo Simulation Results</h5>';
                    html += '<div class="chart-container">';
                    html += '<canvas id="monteCarloChart"></canvas>';
                    html += '</div></div>';
                }
                html += '</div>';
                // Performance Tab
                html += `
                    <div class="tab-pane fade" id="performance" role="tabpanel">
                        <div class="row mb-4">
                `;
                if (metrics.mape) {
                    html += '<div class="col-md-6">';
                    html += '<div class="model-performance-card">';
                    html += '<h5 class="mb-3">Model Accuracy</h5>';
                    html += '<div class="progress mb-2" style="height: 30px;">';
                    const accuracy = Math.max(0, 100 - metrics.mape);
                    html += `<div class="progress-bar bg-success" role="progressbar" style="width: ${accuracy}%" aria-valuenow="${accuracy}" aria-valuemin="0" aria-valuemax="100">${accuracy.toFixed(1)}%</div>`;
                    html += '</div>';
                    html += `<p class="mb-0"><strong>MAPE:</strong> ${metrics.mape.toFixed(2)}%</p>`;
                    html += '</div></div>';
                }
                if (metrics.best_model) {
                    html += '<div class="col-md-6">';
                    html += '<div class="model-performance-card">';
                    html += '<h5 class="mb-3">Best Performing Model</h5>';
                    html += `<span class="badge-best-model">${metrics.best_model}</span>`;
                    if (metrics.mape) {
                        html += `<p class="mb-0 mt-2"><strong>Accuracy:</strong> ${(100 - metrics.mape).toFixed(1)}%</p>`;
                    }
                    html += '</div></div>';
                }
                html += '</div></div>';
                // Periods Tab
                html += `
                    <div class="tab-pane fade" id="periods" role="tabpanel">
                        <div class="row mb-4">
                `;
                if (metrics.full_period !== undefined) {
                    html += '<div class="col-md-4">';
                    html += '<div class="period-card">';
                    html += '<div class="period-title">Full Period</div>';
                    html += `<p class="mb-0 ${metrics.full_period >= 0 ? 'growth-positive' : 'growth-negative'}">${metrics.full_period.toFixed(1)}%</p>`;
                    html += '</div></div>';
                }
                if (metrics.covid_period !== undefined) {
                    html += '<div class="col-md-4">';
                    html += '<div class="period-card">';
                    html += '<div class="period-title">COVID Period</div>';
                    html += `<p class="mb-0 ${metrics.covid_period >= 0 ? 'growth-positive' : 'growth-negative'}">${metrics.covid_period.toFixed(1)}%</p>`;
                    html += '</div></div>';
                }
                if (metrics.recovery_period !== undefined) {
                    html += '<div class="col-md-4">';
                    html += '<div class="period-card">';
                    html += '<div class="period-title">Recovery Period</div>';
                    html += `<p class="mb-0 ${metrics.recovery_period >= 0 ? 'growth-positive' : 'growth-negative'}">${metrics.recovery_period.toFixed(1)}%</p>`;
                    html += '</div></div>';
                }
                html += '</div></div>';
                // Report Tab
                html += `
                    <div class="tab-pane fade" id="report" role="tabpanel">
                        <div class="accordion" id="reportAccordion">
                `;
                if (data.report) {
                    // Split the report into sections
                    const sections = data.report.split(/(?=ENHANCED METHODOLOGY:|ENHANCED DATA SUMMARY:|PERIOD-SPECIFIC ANALYSIS:|ENHANCED MODEL PERFORMANCE:|ENHANCED FORECAST RESULTS:|ENHANCED MONTE CARLO RESULTS:|SYNTHETIC DATA VALIDATION:|ENHANCED LIMITATIONS|KEY ENHANCED INSIGHTS|STRATEGIC IMPLICATIONS|CONCLUSION)/);
                    sections.forEach((section, index) => {
                        if (section.trim()) {
                            const titleMatch = section.match(/^([A-Z][A-Z\s]+):/);
                            const title = titleMatch ? titleMatch[1] : `Section ${index + 1}`;
                            const content = section.replace(/^[A-Z][A-Z\s]+:/, '').trim();
                            html += `
                                <div class="accordion-item">
                                    <h2 class="accordion-header" id="heading${index}">
                                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                                            ${title}
                                        </button>
                                    </h2>
                                    <div id="collapse${index}" class="accordion-collapse collapse" aria-labelledby="heading${index}" data-bs-parent="#reportAccordion">
                                        <div class="accordion-body">
                                            <pre style="white-space: pre-wrap;">${content}</pre>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                    });
                }
                html += '</div></div></div>';
            } else if (data.error) {
                html += `<div class="alert alert-warning">${data.error}</div>`;
            } else {
                html += '<div class="alert alert-info">No enhanced analysis data available</div>';
            }
            return html;
        }
        function formatQualitativeResults(data) {
            let html = '<h4 class="mb-4">Qualitative Analysis Results</h4>';
            if (data.report) {
                // Check if we have structured sentiment data
                if (data.sentiment_data) {
                    const sentData = data.sentiment_data;
                    // Create sentiment overview cards
                    html += '<div class="row mb-4">';
                    html += '<div class="col-md-3">';
                    html += '<div class="metric-card text-center" style="background: var(--success-gradient);">';
                    html += '<div class="metric-label">Positive</div>';
                    html += `<p class="card-text display-6">${sentData.positive}</p>`;
                    html += `<small>${(sentData.positive/sentData.total*100).toFixed(1)}%</small>`;
                    html += '</div></div>';
                    html += '<div class="col-md-3">';
                    html += '<div class="metric-card text-center" style="background: var(--danger-gradient);">';
                    html += '<div class="metric-label">Negative</div>';
                    html += `<p class="card-text display-6">${sentData.negative}</p>`;
                    html += `<small>${(sentData.negative/sentData.total*100).toFixed(1)}%</small>`;
                    html += '</div></div>';
                    html += '<div class="col-md-3">';
                    html += '<div class="metric-card text-center" style="background: var(--info-gradient);">';
                    html += '<div class="metric-label">Neutral</div>';
                    html += `<p class="card-text display-6">${sentData.neutral}</p>`;
                    html += `<small>${(sentData.neutral/sentData.total*100).toFixed(1)}%</small>`;
                    html += '</div></div>';
                    html += '<div class="col-md-3">';
                    const scoreColor = sentData.sentiment_score > 20 ? 'bg-success' : 
                                        sentData.sentiment_score > 0 ? 'bg-info' : 
                                        sentData.sentiment_score > -20 ? 'bg-warning' : 'bg-danger';
                    html += `<div class="metric-card text-center" style="background: var(--${scoreColor.replace('bg-', '')}-gradient);">`;
                    html += '<div class="metric-label">Score</div>';
                    html += `<p class="card-text display-6">${sentData.sentiment_score}</p>`;
                    html += '<small>Overall Sentiment</small>';
                    html += '</div></div>';
                    html += '</div>';
                }
                // Add the sentiment distribution image if available
                if (data.chart_available) {
                    html += `<div class="text-center mb-3">`;
                    html += `<img src="/sentiment_distribution.png?${Date.now()}" alt="Sentiment Distribution" class="img-fluid rounded shadow" style="max-width: 600px;">`;
                    html += '</div>';
                }
                // Display the detailed report
                html += '<div class="card">';
                html += '<div class="card-header bg-info text-white">';
                html += '<h6 class="mb-0">Detailed Analysis Report</h6>';
                html += '</div>';
                html += '<div class="card-body">';
                html += `<pre style="white-space: pre-wrap; font-family: inherit; background: #f8f9fa; padding: 15px; border-radius: 5px;">${data.report}</pre>`;
                html += '</div></div>';
            } else if (data.no_data) {
                html += '<div class="alert alert-warning">';
                html += '<h5>No Data Available</h5>';
                html += data.report;
                html += '</div>';
            } else if (data.error) {
                const isApiError = data.api_error;
                if (isApiError) {
                    document.getElementById('apiWarning').style.display = 'block';
                }
                html += `<div class="alert ${isApiError ? 'alert-warning' : 'alert-danger'}">${data.error}</div>`;
            } else {
                html += '<div class="alert alert-info">No qualitative analysis data available</div>';
            }
            return html;
        }
        function formatCompsResults(data) {
            let html = '<h4 class="mb-4">Comparable Companies Analysis</h4>';
            if (data.summary) {
                html += '<div class="table-responsive">';
                html += '<table class="table table-striped table-hover">';
                html += '<thead class="table-dark"><tr>';
                // Create table headers
                if (data.summary.length > 0) {
                    Object.keys(data.summary[0]).forEach(key => {
                        html += `<th>${key.replace(/_/g, ' ').toUpperCase()}</th>`;
                    });
                    html += '</tr></thead><tbody>';
                    // Add table rows
                    data.summary.forEach(row => {
                        html += '<tr>';
                        Object.values(row).forEach(value => {
                            html += `<td>${value}</td>`;
                        });
                        html += '</tr>';
                    });
                }
                html += '</tbody></table></div>';
                if (data.dashboard) {
                    html += `<a href="/${data.dashboard}" target="_blank" class="btn btn-secondary">
                        <i class="fas fa-external-link-alt me-2"></i>View Full Dashboard
                    </a>`;
                }
            } else if (data.error) {
                html += `<div class="alert alert-warning">${data.error}</div>`;
            } else {
                html += '<div class="alert alert-info">No comparable companies data available</div>';
            }
            return html;
        }
        function initializeCharts(metrics) {
            // Initialize Forecast Chart
            if (metrics.forecasts) {
                const forecastCtx = document.getElementById('forecastChart').getContext('2d');
                const years = Object.keys(metrics.forecasts);
                const values = Object.values(metrics.forecasts);
                new Chart(forecastCtx, {
                    type: 'line',
                    data: {
                        labels: years,
                        datasets: [{
                            label: 'Revenue Forecast ($M)',
                            data: values,
                            borderColor: 'rgb(79, 172, 254)',
                            backgroundColor: 'rgba(79, 172, 254, 0.1)',
                            tension: 0.4,
                            fill: true,
                            pointBackgroundColor: 'rgb(79, 172, 254)',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 6,
                            pointHoverRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Revenue Forecast (2026-2029)',
                                font: {
                                    size: 16,
                                    weight: 'bold'
                                }
                            },
                            legend: {
                                display: false
                            },
                            tooltip: {
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: 'rgb(79, 172, 254)',
                                borderWidth: 1,
                                padding: 10,
                                displayColors: false,
                                callbacks: {
                                    label: function(context) {
                                        return `Revenue: $${context.parsed.y.toFixed(1)}M`;
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                title: {
                                    display: true,
                                    text: 'Revenue ($M)',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                },
                                ticks: {
                                    callback: function(value) {
                                        return '$' + value + 'M';
                                    }
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Year',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                }
                            }
                        }
                    }
                });
            }
            // Initialize Monte Carlo Chart
            if (metrics.mc_low && metrics.mc_high) {
                const monteCarloCtx = document.getElementById('monteCarloChart').getContext('2d');
                new Chart(monteCarloCtx, {
                    type: 'bar',
                    data: {
                        labels: ['2029 Revenue Distribution'],
                        datasets: [
                            {
                                label: '80% Confidence Interval',
                                data: [metrics.mc_high - metrics.mc_low],
                                backgroundColor: 'rgba(240, 147, 251, 0.6)',
                                borderColor: 'rgb(240, 147, 251)',
                                borderWidth: 1
                            },
                            {
                                label: 'Mean',
                                data: [metrics.mc_mean],
                                backgroundColor: 'rgba(245, 87, 108, 0.8)',
                                borderColor: 'rgb(245, 87, 108)',
                                borderWidth: 1,
                                type: 'line',
                                pointRadius: 8,
                                pointHoverRadius: 10
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Monte Carlo Simulation Results (2029)',
                                font: {
                                    size: 16,
                                    weight: 'bold'
                                }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: 'rgb(240, 147, 251)',
                                borderWidth: 1,
                                padding: 10,
                                displayColors: false,
                                callbacks: {
                                    label: function(context) {
                                        if (context.dataset.label === '80% Confidence Interval') {
                                            return `Range: $${metrics.mc_low.toFixed(1)}M - $${metrics.mc_high.toFixed(1)}M`;
                                        } else {
                                            return `Mean: $${context.parsed.y.toFixed(1)}M`;
                                        }
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Revenue ($M)',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                },
                                ticks: {
                                    callback: function(value) {
                                        return '$' + value + 'M';
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        ''')
    # Create static directory with a basic CSS file
    with open('./static/style.css', 'w') as f:
        f.write('''
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f7fa;
    color: #333;
}
.dashboard-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem 0;
    margin-bottom: 2rem;
    border-radius: 0 0 20px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.analysis-section {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    transition: transform 0.3s ease;
}
.analysis-section:hover {
    transform: translateY(-5px);
}
.control-panel {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    position: sticky;
    top: 20px;
}
.btn-analysis {
    width: 100%;
    margin-bottom: 0.8rem;
    padding: 0.8rem;
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.3s ease;
}
.btn-analysis:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}
.forecast-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}
.monte-carlo-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}
.cagr-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}
.sentiment-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}
.upside-positive {
    color: #28a745;
    font-weight: bold;
}
.upside-negative {
    color: #dc3545;
    font-weight: bold;
}
.results-container {
    min-height: 200px;
    padding: 1rem;
    border-radius: 10px;
    background-color: #f8f9fa;
}
.api-warning {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 5px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.chart-container {
    position: relative;
    height: 300px;
    margin-bottom: 2rem;
}
.nav-tabs .nav-link {
    color: #6c757d;
    font-weight: 500;
}
.nav-tabs .nav-link.active {
    color: #495057;
    font-weight: 600;
}
.tab-content {
    padding-top: 1.5rem;
}
.scenario-table {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}
.scenario-table thead {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.scenario-table th {
    border: none;
    padding: 1rem;
    font-weight: 600;
}
.scenario-table td {
    padding: 1rem;
    vertical-align: middle;
}
.table-success td {
    background-color: rgba(40, 167, 69, 0.1);
}
.table-danger td {
    background-color: rgba(220, 53, 69, 0.1);
}
.table-info td {
    background-color: rgba(23, 162, 184, 0.1);
}
.loading-spinner {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
}
.accordion-button:not(.collapsed) {
    background-color: #f8f9fa;
    color: #495057;
}
.accordion-body {
    background-color: #f8f9fa;
}
.forecast-highlight {
    background: linear-gradient(90deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
    border-left: 4px solid #4facfe;
    padding: 1rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: 1rem;
}
.monte-carlo-highlight {
    background: linear-gradient(90deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
    border-left: 4px solid #f093fb;
    padding: 1rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: 1rem;
}
.model-performance-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}
.badge-best-model {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
}
.period-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    border-left: 4px solid #667eea;
}
.period-title {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 0.5rem;
}
.growth-positive {
    color: #28a745;
    font-weight: 600;
}
.growth-negative {
    color: #dc3545;
    font-weight: 600;
}
@media (max-width: 768px) {
    .control-panel {
        position: relative;
        margin-bottom: 2rem;
    }
    .metric-value {
        font-size: 1.5rem;
    }
}
        ''')
    app.run(debug=True, host='0.0.0.0', port=5005)