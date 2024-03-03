"""All the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

GROUPS = ["marital_status", "qualification"]

DATA = {
    "household": ["hbrutto", "hl", "hgen"],
    "individual": ["pgen", "pl", "ppathl", "pequiv", "kidlong"],
}

COLS = {
    "hbrutto": ["hid", "syear", "hhgr"],
    "hl": ["hid", "syear", "hlf0001_v3"],
    "hgen": ["hid", "syear", "hghinc"],
    "kidlong": ["pid", "hid", "syear", "k_nrkid"],
    "pgen": ["pid", "hid", "syear", "pgbilzeit", "pglfs", "pgfamstd"],
    "pl": [
        "pid",
        "hid",
        "syear",
        "ple0006",
        "ple0007",
        "pmonin",
        "plh0258_h",
        "plb0183",
        "plh0182",
    ],
    "ppathl": [
        "pid",
        "hid",
        "syear",
        "gebjahr",
        "birthregion",
        "migback",
        "sex",
    ],
    "pequiv": ["pid", "hid", "syear", "m11124", "m11127"],
    "health": ["pid", "syear", "bmi"],
}

__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]
