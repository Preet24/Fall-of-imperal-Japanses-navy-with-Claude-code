r"""Extract the chronological list of Japanese merchant vessels sunk (PDF pages 45-115).

One row per sunk vessel: date, name, type, gross tonnage, position, flag and type of
the sinking agent, and JANAC's assessment. Shared logic is in ship_list.py.

Run:
    python ship_lists\merchant_pages.py --pages 45-54
    python ship_lists\merchant_pages.py         # all merchant pages

Writes data/merchant/merchant_losses_<pages>.csv and merchant_subtotals_<pages>.csv.
"""

from ship_list import run_list

# Cells OCR could not read, transcribed from the page image: (page, vessel) -> {column: value}
MANUAL_FIXES = {
    (60, "KOZAN MARU"): {"tonnage": 4180},              # OCR "4,1.0"
    (60, "KANSAI MARU"): {"tonnage": 8614},             # OCR "3,614"
    (60, "KOSHIN MARU"): {"tonnage": 1358},             # OCR "1,353"
    (60, "YANATO E MARU"): {"vessel_name": "YAMATO MARU"},
    (60, "YAPASHIRO MARU"): {"vessel_name": "YAMASHIRO MARU"},
    (108, "KOCHO MARU"): {"tonnage": 6888},             # OCR "6,80g"
    # days the OCR dropped or misread
    (46, "HORAI MARU"): {"day": 1},
    (49, "NIKKEI MARU"): {"day": 8},
    (50, "TAKUSEI MARU"): {"day": 14},
    (78, "RYUKO MARU"): {"day": 4},
    (79, "ASUKAZAN MARU"): {"day": 4},
    (80, "ENJU MARU"): {"day": 4},
    (80, "UNKAI MARU NO 7"): {"day": 4},
    (95, "KANKO MARU"): {"day": 5},
    (95, "NITTEI MARU"): {"day": 6},
    # positions the OCR garbled
    (51, "YASUKAWA MARU"): {"location": "8-41S, 148-27E"},
    (54, "TATEYAMA MARU"): {"location": "39-18N, 142-08E"},
    (56, "KOSEI MARU"): {"location": "8-50N, 147-06E", "_where": {"raw_location": "8-50B, 147-06E"}},
    (59, "TAIKO MARU"): {"location": "12-53N, 131-49E"},
    (59, "SAIPAN MARU"): {"location": "16-12N, 134-09E"},
    (59, "RYOKAI MARU"): {"location": "42-40N, 139-48E"},
    (60, "TAMATSU MARU"): {"location": "5-23N, 100-21E"},
    (60, "ARGUN MARU"): {"location": "26-27N, 122-40E"},
    (60, "DAISHIN MARU"): {"location": "34-15N, 137-00E"},
    (81, "MIIKESAN MARU"): {"location": "22-00N, 120-00E"},
    (113, "HOSHI MARU"): {"location": "34-35N, 135-21E"},   # minutes digit unclear; 34-35N = Osaka Bay
    # assessment lost to OCR
    (50, "KINKAI MARU"): {"assessment": "Sunk"},
    (93, "SHINTO MARU"): {"assessment": "Sunk", "agent_type": "Unknown Aircraft",
                          "agent_probable": True},          # OCR "Unknown Aircraft*Sunk"
}
# Left flagged on purpose - the report itself is incomplete here:
#   p54 KENKOKU: only name and tonnage are printed (no type, position, agent or assessment)
#   p55 NITTSU MARU: position printed as "38-05N, 124-93E" (93 minutes is impossible)
# Printed monthly subtotals that disagree with the report's own rows (kept as printed).
SOURCE_ERRORS = {
    "1945-08": "page 115 lists 49 ships (99,969 t) but the subtotal says 48 / 99,094 t; "
               "Table I's merchant total (2,346 ships, 8,618,109 t) agrees with the 49 rows",
}

if __name__ == "__main__":
    run_list("merchant", "45-115", MANUAL_FIXES, __doc__, SOURCE_ERRORS)
