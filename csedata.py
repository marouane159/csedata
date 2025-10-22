import streamlit as st
import random
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

  # Streamlit Page Setup
st.set_page_config(layout="wide", page_title="CSE DATA ANALYTICS", page_icon="🇲🇦")
    

# --- Configuration & Data ---
BASE_STOCKS = [
     {"symbol": "TGC", "name": "TRAVAUX GENERAUX DE CONSTRUCTIONS", "sector": "Construction"},
    {"symbol": "TMA", "name": "TOTALENERGIES MARKETING MAROC", "sector": "Énergie"},
    {"symbol": "TQM", "name": "TAQA MOROCCO", "sector": "Énergie"},
    {"symbol": "NKL", "name": "ENNAKL SA", "sector": "Transport"},
    {"symbol": "LHM", "name": "LAFARGEHOLCIM", "sector": "Construction"},
    {"symbol": "UMR", "name": "UNIMER", "sector": "Agroalimentaire"},
    {"symbol": "WAA", "name": "WAFA ASSURANCE", "sector": "Assurance"},
    {"symbol": "ZDJ", "name": "ZELLIDJA S.A", "sector": "Mines"},
    {"symbol": "MSA", "name": "SODEP MARSA MAROC", "sector": "Transport"},
    {"symbol": "RDS", "name": "RESIDENCE DAR SAADA", "sector": "Construction"},
    {"symbol": "CSR", "name": "COSUMAR", "sector": "Industrie"},
    {"symbol": "CFG", "name": "CFG BANK", "sector": "Banque"},
    {"symbol": "CMG", "name": "CMGP CAS", "sector": "Agriculture"},
    {"symbol": "HPS", "name": "HPS", "sector": "Paiment"},
    {"symbol": "S2M", "name": "S2M", "sector": "Paiment"},
    {"symbol": "RIS", "name": "RISMA", "sector": "Hotel Management"},
    {"symbol": "DHO", "name": "DELTA HOLDING", "sector": "Industrie"},
    {"symbol": "DWY", "name": "DISWAY", "sector": "Distribution éléctro"},
    {"symbol": "SNA", "name": "STOKVIS NORD AFRIQUE", "sector": "Distribution service"},
    {"symbol": "SNP", "name": "SNEP", "sector": "Process Industries"},
    {"symbol": "STR", "name": "STROC INDUSTRIE", "sector": "Service Industriel"},
    {"symbol": "INV", "name": "INVOLYS", "sector": "Service de Technologie"},
    {"symbol": "MIC", "name": "MICRODATA", "sector": "Service de Technologie"},
    {"symbol": "DYT", "name": "DISTY TECHNOLOGIES", "sector": "Service de destribution"},
    {"symbol": "ADH", "name": "DOUJA PROM ADDOHA", "sector": "Immobilier"},
    {"symbol": "IMO", "name": "IMMORENT INVEST", "sector": "Immobilier"},
    {"symbol": "ADI", "name": "ALLIANCES", "sector": "Divers"},
    {"symbol": "AFI", "name": "AFRIC INDUSTRIES", "sector": "Industrie"},
    {"symbol": "AFM", "name": "AFMA", "sector": "Finance"},
    {"symbol": "AKT", "name": "AKDITAL S.A", "sector": "Santé"},
    {"symbol": "ALM", "name": "ALUMINIUM DU MAROC", "sector": "Matériaux"},
    {"symbol": "ARD", "name": "ARADEI CAPITAL", "sector": "Immobilier"},
    {"symbol": "ATH", "name": "AUTO HALL", "sector": "Automobile"},
    {"symbol": "ATL", "name": "ATLANTASANAD", "sector": "Distribution"},
    {"symbol": "ATW", "name": "ATTIJARIWAFA BANK", "sector": "Banque"},
    {"symbol": "BAL", "name": "BALIMA", "sector": "Distribution"},
    {"symbol": "BCP", "name": "BANQUE CENTRALE POPULAIRE", "sector": "Banque"},
    {"symbol": "CRS", "name": "CARTIER SAADA", "sector": "Distribution"},
    {"symbol": "CIH", "name": "CREDIT IMMOBILIER ET HOTELIER", "sector": "Banque"},
    {"symbol": "CMT", "name": "CIMENTS DU MAROC", "sector": "Matériaux"},
    {"symbol": "COL", "name": "COLORADO", "sector": "Distribution"},
    {"symbol": "CTM", "name": "COMPAGNIE DE TRANSPORTS AU MAROC", "sector": "Transport"},
    {"symbol": "DIM", "name": "DELATTRE LEVIVIER MAROC", "sector": "Industrie"},
    {"symbol": "DRI", "name": "DARI COUSPATE", "sector": "Agroalimentaire"},
    {"symbol": "EQD", "name": "EQDOM", "sector": "Immobilier"},
    {"symbol": "FBR", "name": "FENIE BROSSETTE", "sector": "Distribution"},
    {"symbol": "IAM", "name": "MAROC TELECOM", "sector": "Télécom"},
    {"symbol": "INM", "name": "INDUSTRIE DU MAROC", "sector": "Industrie"},
    {"symbol": "JET", "name": "JET CONTRACTORS", "sector": "Construction"},
    {"symbol": "LES", "name": "LESIEUR CRISTAL", "sector": "Agroalimentaire"},
    {"symbol": "MOX", "name": "MAGHREB OXYGENE", "sector": "Industrie"},
    {"symbol": "MNG", "name": "MANAGEM", "sector": "Mines"},
    {"symbol": "MUT", "name": "MUTANDIS", "sector": "Agroalimentaire"},
    {"symbol": "RDS", "name": "RÉSIDENCES DAR SAADA", "sector": "Immobilier"},
    {"symbol": "SID", "name": "SONASID", "sector": "Agroalimentaire"},
    {"symbol": "SNP", "name": "SNEP", "sector": "Industrie"},
    {"symbol": "SOT", "name": "SOTHEMA", "sector": "Pharma"},
    {"symbol": "SRM", "name": "REALISATIONS MECANIQUES", "sector": "Industrie"},
    {"symbol": "STR", "name": "STROC INDUSTRIE", "sector": "Industrie"},
    {"symbol": "MDP", "name": "MED PAPER", "sector": "Industrie"},
    {"symbol": "VCN", "name": "VICENNE", "sector": "Santé"},
    {"symbol": "SMI", "name": "Société métallurgique d'imiter", "sector": "Finance"},
    {"symbol": "CDM", "name": "Crédit du Maroc", "sector": "Banque"},
    {"symbol": "REB", "name": "Rebab Company SA", "sector": "NA"},
    {"symbol": "IBM", "name": "IBMaroc", "sector": "NA"},
    
]


# --- Parsing Helper Functions ---
def _parse_price(text: str) -> float | None:
    cleaned = (
        text.replace("MAD", "").replace("\u202f", "").replace(" ", "").replace(",", "").strip()
    )
    try:
        return float(cleaned)
    except Exception:
        return None

def _parse_percent(text: str) -> float | None:
    t = (
        text.replace("%", "").replace("٪", "").replace("\u202f", "").replace("\xa0", "")
        .replace(" ", "").replace(",", ".").replace("−", "-").replace("–", "-")
        .replace("—", "-").replace("＋", "+").strip()
    )
    try:
        return float(t)
    except Exception:
        return None

def _parse_market_cap(text: str) -> float | None:
    text = text.upper().replace(",", "").replace("MAD", "").strip()
    multiplier = 1
    if text.endswith('B'):
        multiplier = 1_000_000_000
        text = text[:-1]
    elif text.endswith('M'):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith('K'):
        multiplier = 1_000
        text = text[:-1]
    try:
        return float(text) * multiplier
    except Exception:
        return None

def _parse_pe_ratio(text: str) -> float | None:
    text = text.replace(" ", "").replace(",", "").strip()
    if not text or text in ["N/A", "-", "--", "—", "N/A"]:
        return None
    try:
        value = float(text)
        return value if 0 < value < 1000 else None
    except Exception:
        return None

# --- Data Scraping ---
@st.cache_data(ttl=900)
def get_moroccan_stocks() -> pd.DataFrame | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        url = "https://www.tradingview.com/markets/stocks-morocco/market-movers-all-stocks/"
        response = requests.get(url, headers=headers, timeout=20)
        
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        
        if not table:
            return None

        stocks_data = []
        base_index = {s["symbol"]: s for s in BASE_STOCKS}

        for row in table.find_all("tr")[1:]:
            try:
                cells = row.find_all("td")
                if len(cells) < 8:
                    continue

                symbol = None
                symbol_link = cells[0].find("a")
                if symbol_link and symbol_link.text:
                    symbol = symbol_link.text.strip()
                else:
                    symbol = cells[0].get_text(strip=True)

                if not symbol or symbol not in base_index:
                    continue

                # Extract data from specific columns
                price = _parse_price(cells[1].get_text(" ", strip=True))
                change_percent = _parse_percent(cells[2].get_text(" ", strip=True))
                
                market_cap = None
                if len(cells) > 5:
                    market_cap_text = cells[5].get_text(" ", strip=True)
                    market_cap = _parse_market_cap(market_cap_text)
                
                pe_ratio = None
                if len(cells) > 6:
                    pe_text = cells[6].get_text(" ", strip=True)
                    pe_ratio = _parse_pe_ratio(pe_text)

                if price is None or change_percent is None:
                    continue

                stock_info = base_index[symbol]
                stocks_data.append({
                    "symbol": symbol,
                    "name": stock_info["name"],
                    "sector": stock_info["sector"],
                    "price": float(price),
                    "change_percent": float(change_percent),
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                })

            except Exception:
                continue

        if not stocks_data:
            return None

        df = pd.DataFrame(stocks_data)
        return df.sort_values(["sector", "symbol"]).reset_index(drop=True)
        
    except Exception:
        return None

# --- Enhanced Data Management ---
def load_stock_data():
    scraped_df = get_moroccan_stocks()
    st.session_state.is_live_data = (scraped_df is not None and not scraped_df.empty)
    
    if st.session_state.is_live_data:
        scraped_df['change'] = scraped_df['price'] * (scraped_df['change_percent'] / 100)
        scraped_df.rename(columns={'change_percent': 'percentChange'}, inplace=True)
        
        # Calculate additional metrics
        scraped_df['market_cap_formatted'] = scraped_df['market_cap'].apply(format_market_cap)
        scraped_df['performance'] = scraped_df['percentChange'].apply(
            lambda x: 'Haute Performance' if x > 5 else 'Performance Positive' if x > 0 else 'Performance Négative' if x < 0 else 'Neutre'
        )
        
        return scraped_df.to_dict('records')

    # Fallback data with enhanced simulation
    final_stocks = []
    base_stocks_map = {s["symbol"]: s for s in BASE_STOCKS}
    
    for stock_info in base_stocks_map.values():
        initial_price = round(random.uniform(50.0, 500.0), 2)
        change_mad = round(random.uniform(-5.0, 5.0), 2)
        percent_change = round((change_mad / initial_price) * 100, 2)
        market_cap = round(random.uniform(100_000_000, 50_000_000_000), 2)
        pe_ratio = round(random.uniform(5.0, 30.0), 2) if random.random() > 0.2 else None
        
        final_stocks.append({
            "symbol": stock_info["symbol"], 
            "name": stock_info["name"],
            "sector": stock_info["sector"],
            "price": initial_price,
            "change": change_mad,
            "percentChange": percent_change,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "market_cap_formatted": format_market_cap(market_cap),
            "performance": 'Haute Performance' if percent_change > 5 else 'Performance Positive' if percent_change > 0 else 'Performance Négative' if percent_change < 0 else 'Neutre'
        })
        
    return final_stocks

# --- Enhanced UI Helper Functions ---
def format_market_cap(market_cap):
    if market_cap is None:
        return "N/A"
    if market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.2f}M"
    elif market_cap >= 1_000:
        return f"{market_cap / 1_000:.2f}K"
    else:
        return f"{market_cap:.0f}"

def render_stock_card(stock):
    change = stock["change"]
    is_gainer = change > 0.005
    is_loser = change < -0.005
    delta_color = "inverse" if is_loser else "normal" if is_gainer else "off"
    
    with st.container(border=True):
        st.markdown(f"**{stock['symbol']}** <span style='font-size: 0.75rem; color: #6B7280;'>({stock['sector']})</span>", unsafe_allow_html=True)
        st.caption(f"{stock['name']}")
        
        col_price, col_change = st.columns(2)
        with col_price:
            st.metric(label="Prix", value=f"{stock['price']:.2f} MAD", label_visibility="collapsed")
        with col_change:
            st.metric(label="Changement", value=f"{abs(stock['percentChange']):.2f}%", 
                     delta=f"{change:.2f} MAD", delta_color=delta_color, label_visibility="collapsed")
        
        col_mcap, col_pe = st.columns(2)
        with col_mcap:
            st.metric(label="Capitalisation", value=stock.get('market_cap_formatted', 'N/A'), label_visibility="visible")
        with col_pe:
            pe_ratio = stock.get('pe_ratio')
            pe_display = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
            st.metric(label="Ratio P/E", value=pe_display, label_visibility="visible")

# --- Advanced Visualization Functions ---
def create_market_overview_charts(df):
    # Market sentiment gauge
    avg_change = df['percentChange'].mean()
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = avg_change,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment du Marché"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-10, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-10, -5], 'color': "lightcoral"},
                {'range': [-5, 0], 'color': "lightyellow"},
                {'range': [0, 5], 'color': "lightgreen"},
                {'range': [5, 10], 'color': "limegreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_change}}))
    
    fig_gauge.update_layout(height=300)
    
    return fig_gauge

def create_sector_performance_chart(df):
    sector_perf = df.groupby('sector').agg({
        'percentChange': 'mean',
        'symbol': 'count',
        'market_cap': 'sum'
    }).round(2).reset_index()
    
    sector_perf.columns = ['Secteur', 'Changement Moyen %', 'Nombre Actions', 'Capitalisation Totale']
    
    fig = px.bar(sector_perf, x='Secteur', y='Changement Moyen %',
                 title='Performance par Secteur (Changement Moyen %)',
                 color='Changement Moyen %',
                 color_continuous_scale='RdYlGn')
    
    fig.update_layout(xaxis_tickangle=-45, height=400)
    return fig

def create_market_cap_bubble_chart(df):
    fig = px.scatter(df, x='percentChange', y='price', size='market_cap',
                     color='sector', hover_name='symbol',
                     title='Capitalisation vs Performance',
                     labels={'percentChange': 'Changement Quotidien (%)', 'price': 'Prix (MAD)'})
    
    fig.update_layout(height=500, showlegend=True)
    return fig

def create_pe_ratio_analysis(df):
    pe_stocks = df[df['pe_ratio'].notna()]
    
    if len(pe_stocks) > 0:
        fig = px.histogram(pe_stocks, x='pe_ratio', nbins=20,
                          title='Distribution des Ratios P/E',
                          labels={'pe_ratio': 'Ratio P/E'})
        fig.update_layout(height=400)
        return fig
    return None

def create_top_performers_chart(df, top_n=10):
    top_gainers = df.nlargest(top_n, 'percentChange')
    top_losers = df.nsmallest(top_n, 'percentChange')
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Meilleurs Performers', 'Moins Bon Performers'))
    
    fig.add_trace(go.Bar(x=top_gainers['percentChange'], y=top_gainers['symbol'],
                         orientation='h', marker_color='green', name='Gagnants'),
                  row=1, col=1)
    
    fig.add_trace(go.Bar(x=top_losers['percentChange'], y=top_losers['symbol'],
                         orientation='h', marker_color='red', name='Perdants'),
                  row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, title_text=f"Top {top_n} Performers")
    return fig

# --- Main Advanced Dashboard ---
def app():
    # Initialize Session State
    if 'stocks' not in st.session_state:
        st.session_state.stocks = load_stock_data()
        st.session_state.last_updated = datetime.now()
        st.session_state.is_loading = False
        st.session_state.is_live_data = False

    # Refresh Logic
    def refresh_data():
        st.session_state.is_loading = True
        get_moroccan_stocks.clear()
        st.session_state.stocks = load_stock_data()
        time.sleep(0.5)
        st.session_state.last_updated = datetime.now()
        st.session_state.is_loading = False

  
    # Simple Clean CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
            html, body, [class*="stApp"] { 
                font-family: 'Inter', sans-serif;
            }
            
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.5rem;
            }
            
            .risk-brand {
                font-size: 1.2rem;
                font-weight: 600;
                color: #6B7280;
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown('<div class="main-header">RISK NETWORK DATA ANALYTICS</div>', unsafe_allow_html=True)
        st.markdown('<div class="risk-brand">RISK NETWORK</div>', unsafe_allow_html=True)
        status_text = "🟢 Données TradingView en Direct" if st.session_state.is_live_data else "🟡 Données Simulées"
        st.markdown(f"**{status_text}** | Dernière Mise à Jour: {st.session_state.last_updated.strftime('%H:%M:%S')}")
    
    with col3:
        button_label = "🔄 Actualisation..." if st.session_state.is_loading else "Actualiser Données"
        st.button(button_label, on_click=refresh_data, disabled=st.session_state.is_loading, type="primary")

    st.markdown("---")

    if not st.session_state.stocks:
        st.error("Échec du chargement des données boursières")
        return
        
    df = pd.DataFrame(st.session_state.stocks)
    
    # Key Metrics Overview
    st.markdown("## Aperçu du Marché")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_mcap = df['market_cap'].sum() if df['market_cap'].notna().any() else 0
        st.metric("Capitalisation Totale", format_market_cap(total_mcap))
    
    with col2:
        avg_change = df['percentChange'].mean()
        st.metric("Changement Moyen", f"{avg_change:.2f}%")
    
    with col3:
        gainers = len(df[df['percentChange'] > 0])
        st.metric("Actions en Hausse", gainers)
    
    with col4:
        losers = len(df[df['percentChange'] < 0])
        st.metric("Actions en Baisse", losers)
    
    with col5:
        stocks_with_pe = df['pe_ratio'].notna().sum()
        st.metric("Actions avec P/E", f"{stocks_with_pe}/{len(df)}")

    # Advanced Charts Section
    st.markdown("## Analytics Avancés")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Marché", "Analyse Sectorielle", "Métriques Valorisation", "Top Performers"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(create_market_overview_charts(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_top_performers_chart(df, 8), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_sector_performance_chart(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_market_cap_bubble_chart(df), use_container_width=True)
    
    with tab3:
        pe_chart = create_pe_ratio_analysis(df)
        if pe_chart:
            st.plotly_chart(pe_chart, use_container_width=True)
        else:
            st.info("Aucune donnée de ratio P/E disponible pour l'analyse")
        
        # P/E Ratio statistics
        if df['pe_ratio'].notna().any():
            pe_stats = df['pe_ratio'].describe()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("P/E Moyen", f"{pe_stats['mean']:.1f}")
            with col2:
                st.metric("P/E Médian", f"{pe_stats['50%']:.1f}")
            with col3:
                st.metric("P/E Minimum", f"{pe_stats['min']:.1f}")
            with col4:
                st.metric("P/E Maximum", f"{pe_stats['max']:.1f}")
    
    with tab4:
        st.plotly_chart(create_top_performers_chart(df, 15), use_container_width=True)

    # Stock Watchlist with Enhanced Filtering
    st.markdown("## 🔍 Watchlist des Actions")
    
    # Range sliders for P/E and Market Cap
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_query = st.text_input("🔎 Rechercher Actions", placeholder="Symbole ou Nom d'entreprise...")
    
    with col2:
        selected_sector = st.selectbox("Secteur", ["Tous les Secteurs"] + sorted(df['sector'].unique().tolist()))
    
    with col3:
        # P/E Ratio Range Filter
        pe_min = 0
        pe_max = 50
        if df['pe_ratio'].notna().any():
            pe_min = int(df['pe_ratio'].min())
            pe_max = int(df['pe_ratio'].max()) + 1
        pe_range = st.slider(
            "Ratio P/E",
            min_value=pe_min,
            max_value=pe_max,
            value=(pe_min, pe_max),
            help="Filtrer par fourchette de ratio Prix/Bénéfice"
        )
    
    with col4:
        # Market Cap Range Filter
        market_cap_min = 0
        market_cap_max = 50_000_000_000
        if df['market_cap'].notna().any():
            market_cap_min = int(df['market_cap'].min())
            market_cap_max = int(df['market_cap'].max())
        market_cap_range = st.slider(
            "Capitalisation (Milliards)",
            min_value=0,
            max_value=int(market_cap_max / 1_000_000_000) + 1,
            value=(0, int(market_cap_max / 1_000_000_000) + 1),
            help="Filtrer par fourchette de capitalisation boursière"
        )

    # Additional filters
    col5, col6 = st.columns(2)
    
    with col5:
        performance_filter = st.selectbox("Performance", ["Toutes", "Haute Performance (>5%)", "Performance Positive (0-5%)", "Performance Négative", "Neutre"])
    
    with col6:
        pe_exists_filter = st.selectbox("Disponibilité P/E", ["Toutes", "Avec P/E", "Sans P/E"])

    # Apply filters
    filtered_df = df.copy()
    
    if selected_sector != "Tous les Secteurs":
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
    
    if search_query:
        search_lower = search_query.lower()
        filtered_df = filtered_df[
            filtered_df['symbol'].str.lower().str.contains(search_lower) |
            filtered_df['name'].str.lower().str.contains(search_lower)
        ]
    
    # Apply P/E range filter
    filtered_df = filtered_df[
        (filtered_df['pe_ratio'].isna()) | 
        ((filtered_df['pe_ratio'] >= pe_range[0]) & (filtered_df['pe_ratio'] <= pe_range[1]))
    ]
    
    # Apply Market Cap range filter
    market_cap_min_val = market_cap_range[0] * 1_000_000_000
    market_cap_max_val = market_cap_range[1] * 1_000_000_000
    filtered_df = filtered_df[
        (filtered_df['market_cap'].isna()) | 
        ((filtered_df['market_cap'] >= market_cap_min_val) & (filtered_df['market_cap'] <= market_cap_max_val))
    ]
    
    if performance_filter != "Toutes":
        if performance_filter == "Haute Performance (>5%)":
            filtered_df = filtered_df[filtered_df['percentChange'] > 5]
        elif performance_filter == "Performance Positive (0-5%)":
            filtered_df = filtered_df[(filtered_df['percentChange'] > 0) & (filtered_df['percentChange'] <= 5)]
        elif performance_filter == "Performance Négative":
            filtered_df = filtered_df[filtered_df['percentChange'] < 0]
        elif performance_filter == "Neutre":
            filtered_df = filtered_df[filtered_df['percentChange'] == 0]
    
    if pe_exists_filter == "Avec P/E":
        filtered_df = filtered_df[filtered_df['pe_ratio'].notna()]
    elif pe_exists_filter == "Sans P/E":
        filtered_df = filtered_df[filtered_df['pe_ratio'].isna()]

    # Display filtered results
    if len(filtered_df) == 0:
        st.info("Aucune action ne correspond à vos critères de filtrage.")
    else:
        st.write(f"**Affichage de {len(filtered_df)} actions**")
        
        # Sort options
        sort_by = st.selectbox("Trier par", [
            "Symbole A-Z", "Symbole Z-A", 
            "Prix Haut-Bas", "Prix Bas-Haut",
            "Changement % Haut-Bas", "Changement % Bas-Haut",
            "Capitalisation Haut-Bas"
        ])
        
        if sort_by == "Symbole A-Z":
            filtered_df = filtered_df.sort_values('symbol')
        elif sort_by == "Symbole Z-A":
            filtered_df = filtered_df.sort_values('symbol', ascending=False)
        elif sort_by == "Prix Haut-Bas":
            filtered_df = filtered_df.sort_values('price', ascending=False)
        elif sort_by == "Prix Bas-Haut":
            filtered_df = filtered_df.sort_values('price')
        elif sort_by == "Changement % Haut-Bas":
            filtered_df = filtered_df.sort_values('percentChange', ascending=False)
        elif sort_by == "Changement % Bas-Haut":
            filtered_df = filtered_df.sort_values('percentChange')
        elif sort_by == "Capitalisation Haut-Bas":
            filtered_df = filtered_df.sort_values('market_cap', ascending=False)
        
        # Display cards in grid
        filtered_stocks = filtered_df.to_dict('records')
        cols = st.columns(4)
        
        for index, stock in enumerate(filtered_stocks):
            with cols[index % 4]:
                render_stock_card(stock)

    # Footer with Copyright
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;'>"
        "Tous droits réservés - <strong>www.risk.ma</strong><br>"
        "<span style='font-size: 0.8rem;'>RISK NETWORK</span>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    if 'is_live_data' not in st.session_state:
        st.session_state.is_live_data = False 
    app()
