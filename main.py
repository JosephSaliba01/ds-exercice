__author__ = "Joseph Saliba"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

# load dataset
df = pd.read_csv('data/sales_data_sample.csv', encoding= 'unicode_escape')

#! Part 0: Preprocessing of the data
# drop columns we don't need
df = df.drop([
    'PHONE',
    'ADDRESSLINE1',
    'MSRP',
    'STATE',
    'ADDRESSLINE1',
    'ADDRESSLINE2',
    'TERRITORY',
    'CONTACTLASTNAME',
    'CONTACTFIRSTNAME',
    'DEALSIZE'
], axis=1)

# drop rows with cancelled orders
df = df[df.STATUS != 'Cancelled']

# convert date to datetime datatype
df.ORDERDATE = pd.to_datetime(df.ORDERDATE)

#! Part 1: Visualizing top 5 customers with the highest sales growth
N_CUSTOMERS = 5

# get the customer sales ordered
top_N_customer_sales = df.groupby('CUSTOMERNAME').SALES.sum().to_frame().sort_values(by='SALES', ascending=False).reset_index()

# plot growth
for index, row in top_N_customer_sales.iterrows():
    customer_df = df[df.CUSTOMERNAME == row['CUSTOMERNAME']].sort_values(by='ORDERDATE')
    if index < N_CUSTOMERS:
        plt.plot(customer_df.ORDERDATE.unique(), np.cumsum(list(customer_df.groupby('ORDERDATE').SALES.sum())), label=row['CUSTOMERNAME'], marker='.')
    else:
        plt.plot(customer_df.ORDERDATE.unique(), np.cumsum(list(customer_df.groupby('ORDERDATE').SALES.sum())), color='gray', alpha=0.1)

print(f'Top {N_CUSTOMERS} customer sales:\n{top_N_customer_sales.head(N_CUSTOMERS)}\n')
plt.title(f"Visualization of the top {N_CUSTOMERS} customer sales")
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

#! Part 2: Visualizing the least popular products
N_PRODUCTS = 5

# get the top N products
least_N_popular_products = df.groupby('PRODUCTCODE').QUANTITYORDERED.sum().to_frame().sort_values(by='QUANTITYORDERED', ascending=True).reset_index()

# plot quantity ordered over time
for index, row in least_N_popular_products.iterrows():
    product_df = df[df.PRODUCTCODE == row['PRODUCTCODE']].sort_values(by='ORDERDATE')
    
    if index < N_PRODUCTS:
        plt.plot(product_df.ORDERDATE.unique(), np.cumsum(list(product_df.groupby('ORDERDATE').QUANTITYORDERED.sum())), label=row['PRODUCTCODE'], marker='.')
    else:
        plt.plot(product_df.ORDERDATE.unique(), np.cumsum(list(product_df.groupby('ORDERDATE').QUANTITYORDERED.sum())), color='gray', alpha=0.1)

print(f'Least {N_PRODUCTS} popular products:\n{least_N_popular_products}\n')
plt.title(f"Visualization of the {N_PRODUCTS} least popular products")
plt.xlabel('Date')
plt.ylabel('Total Sales') 
plt.legend()
plt.show()

#! Part 3: Seasonality in sales
N_PRODUCT_LINES = 6

# get the top N product lines
top_N_popular_products = df.groupby('PRODUCTLINE').SALES.sum().to_frame().sort_values(by='SALES', ascending=False).reset_index()

fig, ax = plt.subplots(2, 3)

for n in range(N_PRODUCT_LINES):

    years = [2003, 2004]
    product_line_str = top_N_popular_products.iloc[n].PRODUCTLINE

    product_df = df[df.PRODUCTLINE == product_line_str].sort_values(by='ORDERDATE')

    for year in years:
        p_y = product_df[product_df.YEAR_ID == year]
        ax[n//3, n%3].plot(p_y.MONTH_ID.unique(), list(p_y.groupby('MONTH_ID').SALES.sum()), label=year, marker='.')
        ax[n//3, n%3].set_title(f'{product_line_str}')
        ax[n//3, n%3].legend()

plt.show()

def triple_exponential_moving_average(iterable, factor = 0.1):
    weight, average1, average2, average3 = 0.0, 0.0, 0.0, 0.0
    for x in iterable:
        weight += factor * (1 - weight)
        average1 += factor * (x - average1)
        average2 += factor * (average1 / weight - average2)
        average3 += factor * (average2 / weight - average3)
        yield (3 * (average1 - average2) + average3) / weight

df = df[df.YEAR_ID != 2005]

product_line_df = df[df.PRODUCTLINE == top_N_popular_products.iloc[0].PRODUCTLINE][['ORDERDATE', 'SALES']]

product_line_df = product_line_df.groupby('ORDERDATE').SALES.sum().to_frame().sort_index().asfreq('D').interpolate()

plt.plot(product_line_df, label="Interpolated data", color='gray', alpha=0.7)

product_line_df = pd.DataFrame({"SALES": list(triple_exponential_moving_average(product_line_df.SALES))}, product_line_df.index)

plt.plot(product_line_df, label="Moving Average", color='blue')
plt.title(f"Application of the TRIPLE EXPONENTIAL MOVING AVERAGE on the sales data of Classic Cars")
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# decompose using additive model to check for trend and seasonality
# y(t) = Trend + Seasonality + Residual
decompose_result_mult = seasonal_decompose(x=product_line_df, model="additive", period=687//2)

decompose_result_mult.plot()
plt.show()