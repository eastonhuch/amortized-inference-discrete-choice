import pandas as pd
from datetime import timedelta
import os


CATEGORY = "LAUNDRY DETERGENT"
SUB_CATEGORIES = ["LIQUID", "POWDER", "UNIT DOSE"]
BRAND_REGEX = r" \(.+"
BASE_DIR = "./nielsen-data"

def get_panel_dir(year):
    year_str = str(year)
    return f"{BASE_DIR}/panel/{year_str}/nielsen_extracts/HMS/{year_str}/Annual_Files"


def get_scanner_dir(year):
    year_str = str(year)
    return f"{BASE_DIR}/scanner/{year_str}"


def get_brand_form(filter_columns=True):
    df_brand_form = pd.read_csv(os.path.join(BASE_DIR, "brand_sub_category.csv"), index_col=0)
    df_brand_form["brand_form"] = df_brand_form["BRAND"] + "-" + df_brand_form["sub_category"]
    df_brand_form_subset = df_brand_form[["brand_form", "total_price_paid"]]
    total = df_brand_form_subset["total_price_paid"].sum()
    new_row = pd.DataFrame({"brand_form": ["OTHER"], "total_price_paid": [1.-total]})
    df_brand_form_expanded = pd.concat([df_brand_form_subset, new_row], axis=0).sort_values(by="total_price_paid", ascending=False).reset_index(drop=False)
    df_brand_form_expanded["brand"] = df_brand_form_expanded["brand_form"].str.split("-").str[0]
    df_brand_form_expanded["form"] = df_brand_form_expanded["brand_form"].str.split("-").str[1]
    if filter_columns:
        df_brand_form_expanded = df_brand_form_expanded[["brand_form", "brand", "form"]]
    return df_brand_form_expanded


def get_hierarchy_upcs(year):
    base_dir = get_scanner_dir(year)
    df_hierarchy = pd.read_csv(os.path.join(base_dir, "producthierarchy.tsv"), sep="\t")
    df_hierarchy_laundry_full = df_hierarchy[df_hierarchy["category"] == CATEGORY].copy()
    del df_hierarchy
    df_hierarchy_laundry = df_hierarchy_laundry_full[df_hierarchy_laundry_full["sub_category"].isin(SUB_CATEGORIES)]
    upcs = df_hierarchy_laundry["upc"].unique().tolist()
    return df_hierarchy_laundry, upcs


def get_table(table_name, year, upcs=None, panel=True):
    if panel:
        base_dir = get_panel_dir(year)
    else:
        base_dir = get_scanner_dir(year)
    df = pd.read_csv(os.path.join(base_dir, table_name), sep="\t", low_memory=False)
    if upcs is None:
        return df
    else:
        df_laundry = df[df["upc"].isin(upcs)].copy()
        del df
        return df_laundry


def transform_brand(s):
   return s.str.replace(BRAND_REGEX, "", regex=True).str.strip()


def get_week_end(d):
    """Gets Saturday of the given week"""
    d_dt = pd.to_datetime(d)
    purchase_week = d_dt.apply(lambda x: (x + timedelta(days=6-x.isoweekday())).date())
    return purchase_week

