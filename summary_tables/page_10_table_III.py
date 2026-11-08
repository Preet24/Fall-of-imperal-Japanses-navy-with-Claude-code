r"""Table III (PDF page 10): vessels sunk by Allied forces other than the United States.

Run:  python summary_tables\page_10_table_III.py
Saves data/summary/table_III.csv
"""

from summary_page import build_table

if __name__ == "__main__":
    df = build_table(10)
