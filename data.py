"""Training data managemenet"""

import pandas as pd

def build_train(dd):
    sales = pd.read_csv(dd + 'sales_train.csv', parse_dates=['date'])
    items = pd.read_csv(dd + 'items.csv')
    item_categories = pd.read_csv(dd + 'item_categories.csv')
    items = items.merge(item_categories, how='left', on='item_category_id')
    sales = sales.merge(items, how='left', on='item_id')

    sales = sales.sort_values(by=['date'])
    # sales['item_price_diff'] = sales.groupby(['shop_id', 'item_id'])['item_price'].apply(lambda x: x.diff())

    month_sales = sales.groupby(['shop_id', 'item_id', 'date_block_num'])[['date', 'item_price', 'item_cnt_day']].agg({
        'date': [min, max],
        'item_price': [min, max],
        'item_cnt_day': sum
    })

    return month_sales

def build_test(dd):
    test = pd.read_csv(dd + 'test.csv')

    return test