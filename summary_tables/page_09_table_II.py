r"""Table II (PDF page 9): vessels sunk by United States forces, by type of US force.

Run:  python summary_tables\page_09_table_II.py
Saves data/summary/table_II.csv
"""

from summary_page import build_table

if __name__ == "__main__":
    df = build_table(9)
