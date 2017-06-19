#!/usr/bin/python
# -*- encoding: utf-8

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def enum_row(row):
    row['state']


def find_state_code(row):
    if row['state'] != 0:
        process.extractOne(row['state'], states, score_cutoff=80)


def fill_state_code(row):
    if row['state'] != 0:
        state = process.extractOne(row['state'], states, score_cutoff=80)
        if state:
            state_name = state[0]
            return state_to_code[state_name]
    return ''


def capital(str):
    return str.capitalize()


def correct_stat(row):
    if row['state'] != 0:
        state = process.extractOne(row['state'], states, score_cutoff=80)
        if state:
            state_name = state[0]
            return ' '.join(map(capital, state_name.split(' ')))
    return row['state']


if __name__ == "__main__":
    # data.head(), data.tail(), data.dtypes, data.columns
    data = pd.read_excel('sales.xlsx', sheetname='sheet1', header=0)

    data['total'] = data['Jan'] + data['Feb'] + data['Mar']

    # Insert a row
    row_sum = data[['Jan', 'Feb', 'Mar', 'total']].sum()
    row = pd.DataFrame(data=row_sum)

    # Insert a "total" column
    row = pd.DataFrame(data=data[['Jan', 'Feb', 'Mar', 'total']].sum()).T
    row_s = row.reindex(columns=data.columns, fill_value=0)

    # append the new "total" column
    data = data.append(row_s, ignore_index=True)

    # Insert a new total row at the 15 row to sum each columns in total
    data = data.rename(index={15: 'Total'})

    data.apply(enum_row, axis=1)

    state_to_code = {"VERMONT": "VT", "GEORGIA": "GA", "IOWA": "IA", "Armed Forces Pacific": "AP", "GUAM": "GU",
                     "KANSAS": "KS", "FLORIDA": "FL", "AMERICAN SAMOA": "AS", "NORTH CAROLINA": "NC", "HAWAII": "HI",
                     "NEW YORK": "NY", "CALIFORNIA": "CA", "ALABAMA": "AL", "IDAHO": "ID",
                     "FEDERATED STATES OF MICRONESIA": "FM",
                     "Armed Forces Americas": "AA", "DELAWARE": "DE", "ALASKA": "AK", "ILLINOIS": "IL",
                     "Armed Forces Africa": "AE", "SOUTH DAKOTA": "SD", "CONNECTICUT": "CT", "MONTANA": "MT",
                     "MASSACHUSETTS": "MA",
                     "PUERTO RICO": "PR", "Armed Forces Canada": "AE", "NEW HAMPSHIRE": "NH", "MARYLAND": "MD",
                     "NEW MEXICO": "NM",
                     "MISSISSIPPI": "MS", "TENNESSEE": "TN", "PALAU": "PW", "COLORADO": "CO",
                     "Armed Forces Middle East": "AE",
                     "NEW JERSEY": "NJ", "UTAH": "UT", "MICHIGAN": "MI", "WEST VIRGINIA": "WV", "WASHINGTON": "WA",
                     "MINNESOTA": "MN", "OREGON": "OR", "VIRGINIA": "VA", "VIRGIN ISLANDS": "VI",
                     "MARSHALL ISLANDS": "MH",
                     "WYOMING": "WY", "OHIO": "OH", "SOUTH CAROLINA": "SC", "INDIANA": "IN", "NEVADA": "NV",
                     "LOUISIANA": "LA",
                     "NORTHERN MARIANA ISLANDS": "MP", "NEBRASKA": "NE", "ARIZONA": "AZ", "WISCONSIN": "WI",
                     "NORTH DAKOTA": "ND",
                     "Armed Forces Europe": "AE", "PENNSYLVANIA": "PA", "OKLAHOMA": "OK", "KENTUCKY": "KY",
                     "RHODE ISLAND": "RI",
                     "DISTRICT OF COLUMBIA": "DC", "ARKANSAS": "AR", "MISSOURI": "MO", "TEXAS": "TX", "MAINE": "ME"}

    states = state_to_code.keys()

    fuzz.ratio('Python Package', 'PythonPackage')
    process.extract('Mississippi', states)
    process.extract('Mississipi', states, limit=1)
    process.extractOne('Mississipi', states)

    data.apply(find_state_code, axis=1)

    data['state'] = data.apply(correct_stat, axis=1)
    data.insert(5, 'State Code', np.nan)
    data['State Code'] = data.apply(fill_state_code, axis=1)

    # group by
    data.groupby('State Code')
    data.groupby('State Code').sum()
    data[['State Code', 'Jan', 'Feb', 'Mar', 'total']].groupby('State Code').sum()

    # Output the result
    data.to_excel('sales_results.xls', sheet_name='Sheet1', index=False)

