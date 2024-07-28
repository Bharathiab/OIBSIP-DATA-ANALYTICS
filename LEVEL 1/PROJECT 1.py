import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data Loading and Cleaning
file_path = "C:\\Users\\S.Bharathi\\Downloads\\retail_sales_dataset.csv"
data = pd.read_csv(file_path)

# Inspect the dataset
print(data.info())
print(data.head())

# Handle missing values
data = data.dropna()  # or use other imputation techniques

# Remove duplicates
data = data.drop_duplicates()

# Correct inconsistencies if needed
# For example: data['column'] = data['column'].apply(correct_function)

# Descriptive Statistics
statistics = data.describe()
print(statistics)

# Explore categorical variables
categorical_stats = data['Product Category'].value_counts()
print(categorical_stats)

# Time Series Analysis
data['Date'] = pd.to_datetime(data['Date'])

# Aggregate sales data
daily_sales = data.resample('D', on='Date').sum()
weekly_sales = data.resample('W', on='Date').sum()
monthly_sales = data.resample('M', on='Date').sum()

# Identify trends and seasonality
plt.plot(daily_sales.index, daily_sales['Total Amount'])
plt.title('Daily Sales Trends')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.show()

plt.plot(weekly_sales.index, weekly_sales['Total Amount'])
plt.title('Weekly Sales Trends')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.show()

plt.plot(monthly_sales.index, monthly_sales['Total Amount'])
plt.title('Monthly Sales Trends')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.show()

# Customer and Product Analysis
customer_demographics = data.groupby(['Customer ID', 'Gender', 'Age']).agg({
    'Total Amount': 'sum',
    'Quantity': 'sum'
}).reset_index()
print(customer_demographics)

product_analysis = data.groupby('Product Category').agg({
    'Total Amount': 'sum',
    'Quantity': 'sum'
}).reset_index()
print(product_analysis)

# Calculate metrics
average_transaction_value = data['Total Amount'].mean()
print(f"Average Transaction Value: {average_transaction_value}")

# Visualization
# Bar chart for 'Product Category'
sns.barplot(x=data['Product Category'].value_counts().index, y=data['Product Category'].value_counts().values)
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Line plot for sales trends over time
sales_trend = data.groupby('Date')['Total Amount'].sum()
plt.plot(sales_trend.index, sales_trend.values)
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.show()

# Heatmap for correlations
numeric_data = data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Pie chart for gender distribution
data['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()

# Recommendations
# Based on the insights gained from the analysis, summarize key findings and provide actionable recommendations
print("Recommendations:")
print("- Marketing Strategies: Target marketing campaigns based on customer demographics.")
print("- Inventory Management: Stock more of the best-selling products.")
print("- Sales Promotions: Offer discounts on products with lower sales to boost their sales.")
