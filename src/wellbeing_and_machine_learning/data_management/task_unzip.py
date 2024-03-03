import zipfile

from wellbeing_and_machine_learning.config import BLD, SRC

DATA_NAMES = {
    "hbrutto.dta",
    "hgen.dta",
    "hl.dta",
    "pgen.dta",
    "pl.dta",
    "ppathl.dta",
    "kidlong.dta",
    "pequiv.dta",
    "health.dta",
}

PRODUCES = {name: BLD / "data" / "unzip" / name for name in DATA_NAMES}


def task_unzip_data(
    depends_on=SRC / "data" / "SOEP_data.zip",
    produces=PRODUCES,
):
    with zipfile.ZipFile(depends_on, "r") as zip_ref:
        zip_ref.extractall(BLD / "data" / "unzip")
