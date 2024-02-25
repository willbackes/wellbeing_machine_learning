"""All the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

GROUPS = ["marital_status", "qualification"]

DATA = {"household": ["hbrutto", "hl", "hgen"], "individual": ["pgen", "pl", "ppathl"]}

COLS = {
    "hbrutto": ["cid", "syear", "hhgr"],
    "hl": ["cid", "syear", "hlf0001_v3"],
    "hgen": ["cid", "syear", "hghinc"],
    "pgen": ["pid", "cid", "syear", "pgbilzeit", "pglfs", "pgfamstd"],
    "pl": [
        "pid",
        "cid",
        "syear",
        "ple0006",
        "ple0007",
        "plc0446",
        "ple0059",
        "pmonin",
        "plh0258_h",
        "plb0178",
        "plh0182",
    ],
    "ppathl": ["pid", "cid", "syear", "gebjahr", "birthregion", "migback", "sex"],
}

__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]
