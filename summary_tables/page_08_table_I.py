r"""Table I (PDF page 8): total Japanese naval and merchant vessels sunk, by sinking agent.

Run:  python summary_tables\page_08_table_I.py
Saves data/summary/table_I.csv
"""

from summary_page import build_table

if __name__ == "__main__":
    df = build_table(8)
