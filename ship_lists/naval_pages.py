r"""Extract the chronological list of Japanese naval vessels sunk (PDF pages 15-44).

One row per sunk vessel: date, name, type, standard tonnage, position, flag and type
of the sinking agent, and JANAC's assessment. Shared logic is in ship_list.py.

Run:
    python ship_lists\naval_pages.py --pages 15-20
    python ship_lists\naval_pages.py            # all naval pages

Writes data/naval/naval_losses_<pages>.csv and naval_subtotals_<pages>.csv.
"""

from ship_list import run_list

# Cells OCR could not read, transcribed from the page image: (page, vessel) -> {column: value}
MANUAL_FIXES = {
    (21, "I 182"): {"location": "10-33N, 125-31E"},
    (25, "RO 116"): {"location": "00-43N, 149-14E"},
}

if __name__ == "__main__":
    run_list("naval", "15-44", MANUAL_FIXES, __doc__)
