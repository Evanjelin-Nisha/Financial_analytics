import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data into a Pandas DataFrame
df = pd.read_csv('C:\project\Financial Analytics data.csv')

# Select a company
company = df[df['Name of Company'] == 'TCS']

# Calculate the market capitalization to quarterly sales ratio
company['Market Cap to Sales Ratio'] = company['Mar Cap – Crore'] / company['Sales Qtr – Crore']

# Calculate the return on equity (ROE)
company['Earnings Per Share'] = (company['Profit After Tax'] / company['Number of Shares']) * 10
company['ROE'] = company['Earnings Per Share'] / company['Net Worth']

# Calculate the price-to-earnings (P/E) ratio
company['P/E Ratio'] = company['Mar Cap – Crore'] / company['Earnings Per Share']

# Calculate the price-to-book (P/B) ratio
company['P/B Ratio'] = company['Mar Cap – Crore'] / company['Net Worth']

# Calculate the debt-to-equity (D/E) ratio
company['D/E Ratio'] = company['Total Debt'] / company['Net Worth']

# Calculate the liquidity ratios
company['Current Ratio'] = company['Current Assets'] / company['Current Liabilities']
company['Quick Ratio'] = (company['Current Assets'] - company['Inventory']) / company['Current Liabilities']

# Calculate the profitability ratios
company['Gross Profit Margin'] = company['Gross Profit'] / company['Sales Qtr – Crore']
company['Net Profit Margin'] = company['Profit After Tax'] / company['Sales Qtr – Crore']

# Calculate the growth ratios
company['Sales Growth Rate'] = (company['Sales Qtr – Crore'].diff() / company['Sales Qtr – Crore'].shift()) * 100
company['Profit Growth Rate'] = (company['Profit After Tax'].diff() / company['Profit After Tax'].shift()) * 100

# Calculate the efficiency ratios
company['Asset Turnover Ratio'] = company['Sales Qtr – Crore'] / company['Total Assets']
company['Inventory Turnover Ratio'] = company['Sales Qtr – Crore'] / company['Inventory']

# Calculate the dividend payout ratio
company['Dividend Payout Ratio'] = company['Dividend Paid'] / company['Profit After Tax']

# Calculate the beta coefficient
company['Beta Coefficient'] = company['Market Capitalization'].rolling(window=5).mean() / company['Market Capitalization'].rolling(window=5).std()

# Visualize the relationships between attributes
plt.scatter(company['Market Cap to Sales Ratio'], company['ROE'])
plt.xlabel('Market Capitalization to Quarterly Sales Ratio')
plt.ylabel('Return on Equity (ROE)')
plt.title('Relationship between Market Capitalization to Quarterly Sales Ratio and Return on Equity (ROE)')
plt.show()

plt.scatter(company['P/E Ratio'], company['Market Cap to Sales Ratio'])
plt.xlabel('Price-to-Earnings (P/E) Ratio')
plt.ylabel('Market Capitalization to Quarterly Sales Ratio')
plt.title('Relationship between Price-to-Earnings (P/E) Ratio and Market Capitalization to Quarterly Sales Ratio')
plt.show()

plt.scatter(company['D/E Ratio'], company['Market Cap to Sales Ratio'])
plt.xlabel('Debt-to-Equity (D/E) Ratio')
plt.ylabel('Market Capitalization to Quarterly Sales Ratio')
plt.title('Relationship between Debt-to-Equity (D/E) Ratio and Market Capitalization to Quarterly Sales Ratio')
plt.show()