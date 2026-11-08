r"""Table V (PDF page 12): sinkings credited to a "probable" agent (a subset of Tables I-IV).

Run:  python summary_tables\page_12_table_V.py
Saves data/summary/table_V.csv
"""

from summary_page import build_table

if __name__ == "__main__":
    df = build_table(12)
