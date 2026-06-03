import pandas as pd

CSV_PATH = r"C:\Users\Projects\Ecommerce-Ai-System\Dataset\data.csv"

def load_and_prepare(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, encoding_errors='replace')

    df.rename(columns={
        'InvoiceNo'   : 'order_id',
        'StockCode'   : 'product_code',
        'Description' : 'product_name',
        'Quantity'    : 'order_quantity',
        'InvoiceDate' : 'order_date',
        'UnitPrice'   : 'price_per_unit',
        'CustomerID'  : 'retailer_id',
        'Country'     : 'buyer_region'
    }, inplace=True)

    region_mapping = {
        'United Kingdom' : 'Greater Accra',
        'USA'            : 'Central Region',
        'France'         : 'Ashanti Region',
        'Germany'        : 'Western Region',
        'Australia'      : 'Northern Region',
        'EIRE'           : 'Oti Region',
        'Poland'         : 'Savannah Region',
        'Belgium'        : 'Upper East Region',
        'Iceland'        : 'Upper West Region',
        'Cyprus'         : 'Western North Region',
        'Israel'         : 'North East Region',
        'Brazil'         : 'Eastern Region',
        'Greece'         : 'Volta Region',
        'Lebanon'        : 'Ahafo Region',
        'Malta'          : 'Bono Region',
        'Spain'          : 'Bono East Region'
    }

    df['buyer_region_original'] = df['buyer_region']
    df['buyer_region']          = df['buyer_region'].replace(region_mapping)
    df['order_date']            = pd.to_datetime(df['order_date'])
    df['retailer_id']           = df['retailer_id'].astype(str)
    df['order_value_ghs']       = df['order_quantity'] * df['price_per_unit']
    df['order_month']           = df['order_date'].dt.month
    df['order_day']             = df['order_date'].dt.day_name()
    df['order_year']            = df['order_date'].dt.year

    return df