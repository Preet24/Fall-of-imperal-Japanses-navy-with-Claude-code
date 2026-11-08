r"""Table IV (PDF page 11): vessels sunk by combined US/Allied forces, marine casualties, mines and unidentified forces.

Run:  python summary_tables\page_11_table_IV.py
Saves data/summary/table_IV.csv
"""

from summary_page import build_table

if __name__ == "__main__":
    df = build_table(11)
