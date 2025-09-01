import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Personal Financial Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data storage
def initialize_session_state():
    if 'income_data' not in st.session_state:
        st.session_state.income_data = pd.DataFrame(columns=['Date', 'Source', 'Amount'])
    
    if 'expense_data' not in st.session_state:
        st.session_state.expense_data = pd.DataFrame(columns=['Date', 'Category', 'Description', 'Amount'])
    
    if 'assets_data' not in st.session_state:
        st.session_state.assets_data = pd.DataFrame(columns=['Date', 'Type', 'Description', 'Amount'])
    
    if 'stocks_data' not in st.session_state:
        st.session_state.stocks_data = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase_Price'])
    
    if 'mf_nps_data' not in st.session_state:
        st.session_state.mf_nps_data = pd.DataFrame(columns=['Date', 'Type', 'Name', 'Amount', 'Expected_Growth'])

# Helper functions
def add_income(date, source, amount):
    new_row = pd.DataFrame({'Date': [date], 'Source': [source], 'Amount': [amount]})
    st.session_state.income_data = pd.concat([st.session_state.income_data, new_row], ignore_index=True)

def add_expense(date, category, description, amount):
    new_row = pd.DataFrame({'Date': [date], 'Category': [category], 'Description': [description], 'Amount': [amount]})
    st.session_state.expense_data = pd.concat([st.session_state.expense_data, new_row], ignore_index=True)
    
    # Auto-add to assets if category is jewels, land, mutual fund, or NPS
    if category.lower() in ['jewels', 'land', 'mutual fund', 'nps']:
        add_asset(date, category, description, amount)

def add_asset(date, asset_type, description, amount):
    new_row = pd.DataFrame({'Date': [date], 'Type': [asset_type], 'Description': [description], 'Amount': [amount]})
    st.session_state.assets_data = pd.concat([st.session_state.assets_data, new_row], ignore_index=True)

def add_stock(ticker, quantity, purchase_price):
    new_row = pd.DataFrame({'Ticker': [ticker], 'Quantity': [quantity], 'Purchase_Price': [purchase_price]})
    st.session_state.stocks_data = pd.concat([st.session_state.stocks_data, new_row], ignore_index=True)

def add_mf_nps(date, inv_type, name, amount, growth):
    new_row = pd.DataFrame({'Date': [date], 'Type': [inv_type], 'Name': [name], 'Amount': [amount], 'Expected_Growth': [growth]})
    st.session_state.mf_nps_data = pd.concat([st.session_state.mf_nps_data, new_row], ignore_index=True)

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Try multiple methods to get price
        hist = stock.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
        return float(price) if price else 0
    except Exception as e:
        st.warning(f"Could not fetch price for {ticker}: {str(e)}")
        return 0

def export_to_excel():
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.income_data.to_excel(writer, sheet_name='Income', index=False)
            st.session_state.expense_data.to_excel(writer, sheet_name='Expenses', index=False)
            st.session_state.assets_data.to_excel(writer, sheet_name='Assets', index=False)
            st.session_state.stocks_data.to_excel(writer, sheet_name='Stocks', index=False)
            st.session_state.mf_nps_data.to_excel(writer, sheet_name='MF_NPS', index=False)
    except ImportError:
        st.error("Excel functionality not available. Please use CSV export instead.")
        return None
    return output.getvalue()

def import_from_excel(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        
        if 'Income' in excel_file.sheet_names:
            income_df = pd.read_excel(uploaded_file, sheet_name='Income')
            st.session_state.income_data = income_df
        
        if 'Expenses' in excel_file.sheet_names:
            expense_df = pd.read_excel(uploaded_file, sheet_name='Expenses')
            st.session_state.expense_data = expense_df
            
        return True
    except Exception as e:
        st.error(f"Error importing file: {str(e)}")
        return False

# Initialize session state
initialize_session_state()

# Sidebar navigation
st.sidebar.title("💰 Financial Tracker")
st.sidebar.markdown("---")
tab_selection = st.sidebar.radio(
    "Navigate to:",
    ["📊 Summary", "💵 Income & Expenses", "🏠 Assets", "📈 Stocks", "🎯 Mutual Funds & NPS"]
)

# Main content based on selected tab
if tab_selection == "📊 Summary":
    st.title("📊 Financial Summary Dashboard")
    
    # Calculate totals
    total_income = st.session_state.income_data['Amount'].sum() if not st.session_state.income_data.empty else 0
    total_expenses = st.session_state.expense_data['Amount'].sum() if not st.session_state.expense_data.empty else 0
    net_balance = total_income - total_expenses
    total_assets = st.session_state.assets_data['Amount'].sum() if not st.session_state.assets_data.empty else 0
    
    # Stock portfolio value
    stock_value = 0
    if not st.session_state.stocks_data.empty:
        for _, row in st.session_state.stocks_data.iterrows():
            current_price = get_stock_price(row['Ticker'])
            stock_value += row['Quantity'] * current_price
    
    # MF & NPS value
    mf_nps_value = st.session_state.mf_nps_data['Amount'].sum() if not st.session_state.mf_nps_data.empty else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Income", f"₹{total_income:,.2f}")
        st.metric("Total Expenses", f"₹{total_expenses:,.2f}")
    
    with col2:
        st.metric("Net Balance", f"₹{net_balance:,.2f}")
        st.metric("Total Assets", f"₹{total_assets:,.2f}")
    
    with col3:
        st.metric("Stock Portfolio", f"₹{stock_value:,.2f}")
        st.metric("MF & NPS", f"₹{mf_nps_value:,.2f}")
    
    with col4:
        total_wealth = net_balance + total_assets + stock_value + mf_nps_value
        st.metric("Total Wealth", f"₹{total_wealth:,.2f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth distribution pie chart
        if total_wealth > 0:
            wealth_data = {
                'Category': ['Cash Balance', 'Assets', 'Stocks', 'MF & NPS'],
                'Value': [max(0, net_balance), total_assets, stock_value, mf_nps_value]
            }
            wealth_df = pd.DataFrame(wealth_data)
            wealth_df = wealth_df[wealth_df['Value'] > 0]
            
            if not wealth_df.empty:
                fig_pie = px.pie(wealth_df, values='Value', names='Category', 
                               title='Wealth Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No wealth data to display")
    
    with col2:
        # Income vs Expenses
        if total_income > 0 or total_expenses > 0:
            inc_exp_data = {
                'Type': ['Income', 'Expenses'],
                'Amount': [total_income, total_expenses]
            }
            inc_exp_df = pd.DataFrame(inc_exp_data)
            
            fig_bar = px.bar(inc_exp_df, x='Type', y='Amount', 
                           title='Income vs Expenses',
                           color='Type')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No income/expense data to display")

elif tab_selection == "💵 Income & Expenses":
    st.title("💵 Income & Expenses Management")
    
    # Two columns for input forms
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("➕ Add Income")
        with st.form("income_form"):
            income_date = st.date_input("Date", datetime.now().date())
            income_source = st.selectbox("Income Source", 
                                       ["My Salary", "Wife's Salary", "Dividend Returns", "Account Credits", "Other"])
            income_amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0)
            
            if st.form_submit_button("Add Income"):
                if income_amount > 0:
                    add_income(income_date, income_source, income_amount)
                    st.success("Income added successfully!")
                else:
                    st.error("Please enter a valid amount")
    
    with col2:
        st.subheader("➖ Add Expense")
        with st.form("expense_form"):
            expense_date = st.date_input("Date", datetime.now().date(), key="exp_date")
            expense_category = st.selectbox("Category", 
                                          ["General", "Jewels", "Land", "Mutual Fund", "NPS", "Other"])
            expense_description = st.text_input("Description")
            expense_amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0, key="exp_amount")
            
            if st.form_submit_button("Add Expense"):
                if expense_amount > 0 and expense_description:
                    add_expense(expense_date, expense_category, expense_description, expense_amount)
                    st.success("Expense added successfully!")
                    if expense_category.lower() in ['jewels', 'land', 'mutual fund', 'nps']:
                        st.info(f"Also added to {expense_category} assets!")
                else:
                    st.error("Please enter valid amount and description")
    
    st.markdown("---")
    
    # File operations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Data to Excel"):
            excel_data = export_to_excel()
            st.download_button(
                label="Download Excel File",
                data=excel_data,
                file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Upload Excel File", type=['xlsx'])
        if uploaded_file and st.button("Import Data"):
            if import_from_excel(uploaded_file):
                st.success("Data imported successfully!")
                st.rerun()
    
    # Display current data
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Income")
        if not st.session_state.income_data.empty:
            recent_income = st.session_state.income_data.tail(10)
            st.dataframe(recent_income, use_container_width=True)
        else:
            st.info("No income data available")
    
    with col2:
        st.subheader("Recent Expenses")
        if not st.session_state.expense_data.empty:
            recent_expenses = st.session_state.expense_data.tail(10)
            st.dataframe(recent_expenses, use_container_width=True)
        else:
            st.info("No expense data available")
    
    # Trends
    if not st.session_state.income_data.empty or not st.session_state.expense_data.empty:
        st.subheader("📈 Monthly Trends")
        
        # Prepare monthly data
        monthly_data = []
        
        if not st.session_state.income_data.empty:
            income_monthly = st.session_state.income_data.copy()
            income_monthly['Date'] = pd.to_datetime(income_monthly['Date'])
            income_monthly['Month'] = income_monthly['Date'].dt.to_period('M')
            income_summary = income_monthly.groupby('Month')['Amount'].sum().reset_index()
            income_summary['Type'] = 'Income'
            income_summary['Month'] = income_summary['Month'].astype(str)
            monthly_data.append(income_summary[['Month', 'Amount', 'Type']])
        
        if not st.session_state.expense_data.empty:
            expense_monthly = st.session_state.expense_data.copy()
            expense_monthly['Date'] = pd.to_datetime(expense_monthly['Date'])
            expense_monthly['Month'] = expense_monthly['Date'].dt.to_period('M')
            expense_summary = expense_monthly.groupby('Month')['Amount'].sum().reset_index()
            expense_summary['Type'] = 'Expenses'
            expense_summary['Month'] = expense_summary['Month'].astype(str)
            monthly_data.append(expense_summary[['Month', 'Amount', 'Type']])
        
        if monthly_data:
            combined_monthly = pd.concat(monthly_data, ignore_index=True)
            fig_trends = px.line(combined_monthly, x='Month', y='Amount', color='Type',
                               title='Monthly Income vs Expenses Trend')
            st.plotly_chart(fig_trends, use_container_width=True)

elif tab_selection == "🏠 Assets":
    st.title("🏠 Assets Portfolio")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ Add Manual Asset")
        with st.form("asset_form"):
            asset_date = st.date_input("Purchase Date", datetime.now().date())
            asset_type = st.selectbox("Asset Type", ["Gold", "Land", "Property", "Jewels", "Other"])
            asset_description = st.text_input("Description")
            asset_amount = st.number_input("Value (₹)", min_value=0.0, step=1000.0)
            
            if st.form_submit_button("Add Asset"):
                if asset_amount > 0 and asset_description:
                    add_asset(asset_date, asset_type, asset_description, asset_amount)
                    st.success("Asset added successfully!")
                else:
                    st.error("Please enter valid details")
    
    with col2:
        st.subheader("📋 Assets Summary")
        if not st.session_state.assets_data.empty:
            # Group by type
            assets_by_type = st.session_state.assets_data.groupby('Type')['Amount'].sum().reset_index()
            
            # Display pie chart
            fig_assets = px.pie(assets_by_type, values='Amount', names='Type',
                              title='Assets Distribution by Type')
            st.plotly_chart(fig_assets, use_container_width=True)
        else:
            st.info("No assets data available")
    
    # Display assets table
    st.subheader("📊 All Assets")
    if not st.session_state.assets_data.empty:
        # Add total value
        total_assets_value = st.session_state.assets_data['Amount'].sum()
        st.metric("Total Assets Value", f"₹{total_assets_value:,.2f}")
        
        # Display table with ability to sort
        assets_display = st.session_state.assets_data.copy()
        assets_display = assets_display.sort_values('Date', ascending=False)
        st.dataframe(assets_display, use_container_width=True)
        
        # Assets growth over time
        if len(assets_display) > 1:
            assets_cumulative = assets_display.copy()
            assets_cumulative['Date'] = pd.to_datetime(assets_cumulative['Date'])
            assets_cumulative = assets_cumulative.sort_values('Date')
            assets_cumulative['Cumulative_Value'] = assets_cumulative['Amount'].cumsum()
            
            fig_growth = px.line(assets_cumulative, x='Date', y='Cumulative_Value',
                               title='Assets Growth Over Time')
            st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.info("No assets recorded yet. Assets will automatically appear here when you add expenses under 'Jewels', 'Land', 'Mutual Fund', or 'NPS' categories in the Income & Expenses tab.")

elif tab_selection == "📈 Stocks":
    st.title("📈 Stock Portfolio")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ Add Stock")
        with st.form("stock_form"):
            stock_ticker = st.text_input("Stock Ticker (e.g., AAPL, TSLA)", placeholder="AAPL").upper()
            stock_quantity = st.number_input("Quantity", min_value=1, step=1)
            stock_price = st.number_input("Purchase Price per Share (₹)", min_value=0.0, step=0.01)
            
            if st.form_submit_button("Add Stock"):
                if stock_ticker and stock_quantity > 0 and stock_price > 0:
                    add_stock(stock_ticker, stock_quantity, stock_price)
                    st.success("Stock added successfully!")
                else:
                    st.error("Please enter valid stock details")
    
    with col2:
        if not st.session_state.stocks_data.empty:
            st.subheader("💰 Portfolio Performance")
            
            portfolio_data = []
            total_invested = 0
            total_current_value = 0
            
            for _, row in st.session_state.stocks_data.iterrows():
                current_price = get_stock_price(row['Ticker'])
                invested_value = row['Quantity'] * row['Purchase_Price']
                current_value = row['Quantity'] * current_price
                gain_loss = current_value - invested_value
                gain_loss_pct = (gain_loss / invested_value * 100) if invested_value > 0 else 0
                
                portfolio_data.append({
                    'Ticker': row['Ticker'],
                    'Quantity': row['Quantity'],
                    'Purchase_Price': row['Purchase_Price'],
                    'Current_Price': current_price,
                    'Invested': invested_value,
                    'Current_Value': current_value,
                    'Gain/Loss': gain_loss,
                    'Gain/Loss %': gain_loss_pct
                })
                
                total_invested += invested_value
                total_current_value += current_value
            
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # Display metrics
            total_gain_loss = total_current_value - total_invested
            total_gain_loss_pct = (total_gain_loss / total_invested * 100) if total_invested > 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Invested", f"₹{total_invested:,.2f}")
            with col_b:
                st.metric("Current Value", f"₹{total_current_value:,.2f}")
            with col_c:
                st.metric("Total Gain/Loss", f"₹{total_gain_loss:,.2f}", f"{total_gain_loss_pct:.2f}%")
        else:
            st.info("No stocks in portfolio yet")
    
    # Display portfolio table
    if not st.session_state.stocks_data.empty and 'portfolio_df' in locals():
        st.subheader("📊 Portfolio Details")
        
        # Format the dataframe for display
        display_df = portfolio_df.copy()
        for col in ['Purchase_Price', 'Current_Price', 'Invested', 'Current_Value', 'Gain/Loss']:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}")
        display_df['Gain/Loss %'] = display_df['Gain/Loss %'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Portfolio composition pie chart
        if len(portfolio_df) > 0:
            fig_composition = px.pie(portfolio_df, values='Current_Value', names='Ticker',
                                   title='Portfolio Composition by Current Value')
            st.plotly_chart(fig_composition, use_container_width=True)
    
    # Refresh stock prices button
    if not st.session_state.stocks_data.empty:
        if st.button("🔄 Refresh Stock Prices"):
            st.rerun()

elif tab_selection == "🎯 Mutual Funds & NPS":
    st.title("🎯 Mutual Funds & NPS Investments")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ Add Investment")
        with st.form("mf_nps_form"):
            inv_date = st.date_input("Investment Date", datetime.now().date())
            inv_type = st.selectbox("Investment Type", ["Mutual Fund", "NPS", "SIP", "Other"])
            inv_name = st.text_input("Investment Name/Scheme")
            inv_amount = st.number_input("Amount (₹)", min_value=0.0, step=1000.0)
            expected_growth = st.number_input("Expected Annual Growth (%)", min_value=0.0, max_value=50.0, step=0.5, value=12.0)
            
            if st.form_submit_button("Add Investment"):
                if inv_amount > 0 and inv_name:
                    add_mf_nps(inv_date, inv_type, inv_name, inv_amount, expected_growth)
                    st.success("Investment added successfully!")
                else:
                    st.error("Please enter valid investment details")
    
    with col2:
        if not st.session_state.mf_nps_data.empty:
            st.subheader("💰 Investment Summary")
            
            # Summary by type
            inv_summary = st.session_state.mf_nps_data.groupby('Type')['Amount'].sum().reset_index()
            
            fig_inv_pie = px.pie(inv_summary, values='Amount', names='Type',
                               title='Investment Distribution by Type')
            st.plotly_chart(fig_inv_pie, use_container_width=True)
        else:
            st.info("No investments recorded yet")
    
    # Display investments table and projections
    if not st.session_state.mf_nps_data.empty:
        st.subheader("📊 Investment Details")
        
        # Calculate current totals
        total_invested = st.session_state.mf_nps_data['Amount'].sum()
        st.metric("Total Invested", f"₹{total_invested:,.2f}")
        
        # Display table
        display_inv = st.session_state.mf_nps_data.copy()
        display_inv = display_inv.sort_values('Date', ascending=False)
        st.dataframe(display_inv, use_container_width=True)
        
        st.subheader("📈 Future Value Projections")
        
        # Create projection chart
        years = list(range(1, 31))  # 30 years projection
        projection_data = []
        
        for _, inv in st.session_state.mf_nps_data.iterrows():
            for year in years:
                # Calculate compound growth
                future_value = inv['Amount'] * ((1 + inv['Expected_Growth']/100) ** year)
                projection_data.append({
                    'Year': year,
                    'Investment': inv['Name'],
                    'Future_Value': future_value,
                    'Type': inv['Type']
                })
        
        if projection_data:
            projection_df = pd.DataFrame(projection_data)
            
            # Total projection by year
            total_projection = projection_df.groupby('Year')['Future_Value'].sum().reset_index()
            
            fig_projection = px.line(total_projection, x='Year', y='Future_Value',
                                   title='Total Investment Growth Projection (30 Years)')
            fig_projection.update_layout(yaxis_title="Future Value (₹)")
            st.plotly_chart(fig_projection, use_container_width=True)
            
            # Individual investment projections
            fig_individual = px.line(projection_df, x='Year', y='Future_Value', color='Investment',
                                   title='Individual Investment Projections')
            fig_individual.update_layout(yaxis_title="Future Value (₹)")
            st.plotly_chart(fig_individual, use_container_width=True)
            
            # Show specific year projections
            st.subheader("🎯 Specific Year Projections")
            selected_years = st.multiselect("Select Years to View", [5, 10, 15, 20, 25, 30], default=[10, 20, 30])
            
            if selected_years:
                year_projections = []
                for year in selected_years:
                    year_data = total_projection[total_projection['Year'] == year]
                    if not year_data.empty:
                        year_projections.append({
                            'Year': year,
                            'Projected_Value': year_data['Future_Value'].iloc[0],
                            'Growth_Multiple': year_data['Future_Value'].iloc[0] / total_invested
                        })
                
                if year_projections:
                    proj_df = pd.DataFrame(year_projections)
                    proj_df['Projected_Value'] = proj_df['Projected_Value'].apply(lambda x: f"₹{x:,.2f}")
                    proj_df['Growth_Multiple'] = proj_df['Growth_Multiple'].apply(lambda x: f"{x:.2f}x")
                    st.table(proj_df)
    else:
        st.info("Add some investments to see projections. Investments will also automatically appear here when you add expenses under 'Mutual Fund' or 'NPS' categories in the Income & Expenses tab.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tips:**")
st.sidebar.markdown("• Add regular income and expenses for better tracking")
st.sidebar.markdown("• Use Excel import/export for bulk data management")
st.sidebar.markdown("• Check stock prices regularly for updated portfolio value")
st.sidebar.markdown("• Set realistic growth expectations for investments")

# Display app info
st.sidebar.markdown("---")
st.sidebar.info("Personal Financial Tracker v1.0")
st.sidebar.markdown("Built with Streamlit 🚀")
