"""Training data managemenet"""

import pandas as pd

import numpy as np
import jax.numpy as jnp

from .lgt_model import Item

def build_data(dd):
    sales = pd.read_csv(dd + 'sales_train.csv', parse_dates=['date'])
    items = pd.read_csv(dd + 'items.csv')
    item_categories = pd.read_csv(dd + 'item_categories.csv')
    shops = pd.read_csv(dd + 'shops.csv')
    items = items.merge(item_categories, how='left', on='item_category_id')
    sales = sales.merge(items, how='left', on='item_id')

    sales = sales.sort_values(by=['date'])
    # sales['item_price_diff'] = sales.groupby(['shop_id', 'item_id'])['item_price'].apply(lambda x: x.diff())

    month_sales = sales.groupby(['shop_id', 'item_id', 'date_block_num'])[['date', 'item_price', 'item_cnt_day']].agg({
        'date': [min, max],
        'item_price': [min, max],
        'item_cnt_day': sum
    })

    test = pd.read_csv(dd + 'test.csv')

    agg_sales = month_sales.reset_index().groupby([('shop_id', ''), ('item_id', '')])[[('item_cnt_day', 'sum'), ('date_block_num', '')]].agg({
        ('item_cnt_day', 'sum'): lambda x: np.array(x),
        ('date_block_num', ''): lambda x: np.array(x)
    }).reset_index()
    agg_sales.columns = ['shop_id', 'item_id', 'item_cnt_day', 'date_block_num']
    all_data = agg_sales.merge(test, how='outer', on=['shop_id', 'item_id'])
    shop_item_cnt = month_sales.reset_index().groupby([('shop_id', ''), ('item_id', '')])[[('date_block_num', '')]].agg({
        ('date_block_num', ''): [len, min, max]
    }).reset_index()
    shop_item_cnt.columns = ['shop_id', 'item_id', 'shop_item_blocks', 'shop_item_minblock', 'shop_item_maxblock']
    all_data = all_data.merge(shop_item_cnt, how='left', on=['shop_id', 'item_id'])
    item_cnt = month_sales.reset_index().groupby([('item_id', '')])[[('date_block_num', '')]].agg({
        ('date_block_num', ''): [len, min, max]
    }).reset_index()
    item_cnt.columns = ['item_id', 'item_blocks', 'item_minblock', 'item_maxblock']
    all_data = all_data.merge(item_cnt, how='left', on='item_id')
    shop_cnt = month_sales.reset_index().groupby([('shop_id', '')])[[('date_block_num', ''), ('item_id', '')]].agg({
        ('date_block_num', ''): [len, min, max],
        ('item_id', ''): lambda x: len(set(x))
    }).reset_index()
    shop_cnt.columns = ['shop_id', 'shop_blocks', 'shop_minblock', 'shop_maxblock', 'shop_items']
    all_data = all_data.merge(shop_cnt, how='left', on='shop_id')
    # all_data = all_data.merge(items, how='left', on='item_id')
    # all_data = all_data.merge(shops, how='left', on='shop_id')

    def _make_array(cnt, dates):
        array = np.empty(34)
        array.fill(np.nan)
        if not np.isnan(dates).all():
            array[dates] = cnt
        return array
    all_data['array'] = all_data.apply(
        lambda row: _make_array(row['item_cnt_day'], row['date_block_num']),
        axis=1
    )
    all_data = all_data.sort_values(by=['shop_id', 'item_id'])
    item_means = jnp.array(all_data.groupby('item_id')['array'].agg(lambda x: np.nanmean(np.vstack(x), axis=0)).to_list())
    item_stds = jnp.array(all_data.groupby('item_id')['array'].agg(lambda x: np.nanstd(np.vstack(x), axis=0)).to_list())
    shop_means = jnp.array(all_data.groupby('shop_id')['array'].agg(lambda x: np.nanmean(np.vstack(x), axis=0)).to_list())
    shop_stds = jnp.array(all_data.groupby('shop_id')['array'].agg(lambda x: np.nanstd(np.vstack(x), axis=0)).to_list())

    def _make_data(cnt, dates, start, start_shop, stop):
        array = np.empty(34)
        array.fill(np.nan)
        if np.isnan(start):
            return jnp.array(array[int(start_shop):int(stop)+1])
        if not np.isnan(dates).all():
            array[dates] = cnt
        return jnp.array(array[int(start):int(stop)+1])
    def _make_item(row):
        data = _make_data(row['item_cnt_day'], row['date_block_num'], row['item_minblock'], row['shop_minblock'], row['shop_maxblock'])
        return Item(
            shop_id=row['shop_id'],
            item_id=row['item_id'],
            data=data,
            start=row['shop_minblock'] if np.isnan(row['item_minblock']) else row['item_minblock'],
            stop=row['shop_maxblock'],
            level=np.nanmean(data),
            trend=np.nanmean(data)
        )
    y = all_data.apply(_make_item, axis=1).to_list()

    return y, test, item_means, item_stds, shop_means, shop_stds
