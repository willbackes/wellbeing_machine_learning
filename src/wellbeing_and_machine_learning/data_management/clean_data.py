import numpy as np
import pandas as pd
from pandas import NA


def clean_data(merged_data):
    valid_data = pd.DataFrame()
    valid_data = _clean_invalid_data(merged_data)
    valid_data["hghinc"] = valid_data["hghinc"].astype(pd.Float64Dtype())

    df = pd.DataFrame()
    df["pid"] = valid_data["pid"]
    df["hid"] = valid_data["hid"]
    df["cid"] = valid_data["cid_x_x"].astype(pd.UInt64Dtype())
    df["syear"] = valid_data["syear"].astype(pd.UInt16Dtype())
    df["smonth"] = _extract_number_from_brackets(valid_data["pmonin"])
    df["age"] = valid_data["syear"] - valid_data["gebjahr"]
    df["agesquared"] = df["age"] ** 2
    df["birthregion"] = _extract_number_from_brackets(valid_data["birthregion"])
    df["weight"] = valid_data["ple0007"].astype(pd.UInt16Dtype())
    df["height"] = valid_data["ple0006"].astype(pd.UInt16Dtype()) / 100
    df["bmi"] = df["weight"] / (df["height"] ** 2)
    df["disability"] = _clean_binary_data(valid_data["plc0446"], is_one="[1] Ja")
    df["education"] = valid_data["pgbilzeit"].astype(pd.Float32Dtype())
    df["labourstatus"] = _clean_binary_data(
        valid_data["pglfs"],
        is_one="[11] Erwerbstätig",
    )
    df["logincome"] = np.log(valid_data["hghinc"])
    df["migback"] = _extract_number_from_brackets(valid_data["migback"])
    df["health"] = valid_data["ple0059"].astype(pd.UInt8Dtype())
    df["housingstatus"] = _extract_number_from_brackets(valid_data["hlf0001_v3"])
    df["maritalstatus"] = _clean_marital_status(valid_data["pgfamstd"])
    df["religion"] = _extract_number_from_brackets(valid_data["plh0258_h"])
    df["sex"] = _extract_number_from_brackets(valid_data["sex"]) - 1
    df["workinghours"] = valid_data["plb0183"].astype(pd.Float64Dtype())
    df["numberofpeople"] = (
        valid_data["hhgr"]
        .replace("[0] Aufgeloeste/n.bearbeitete Haushalte", NA)
        .astype(pd.UInt8Dtype())
    )

    return df


def _clean_invalid_data(data):
    invalid_data_mapping = {
        "[-1] keine Angabe": NA,
        "[-2] trifft nicht zu": NA,
        "[-3] nicht valide": NA,  # codespell: 65
        "[-4] Unzulaessige Mehrfachantwort": NA,
        "[-5] in Fragebogenversion nicht enthalten": NA,
        "[-6] Fragebogenversion mit geaenderter Filterfuehrung": NA,
        "[-7] Nur in weniger eingeschränkter Edition verfügbar": NA,
        "[-8] Frage in diesem Jahr nicht Teil des Frageprogramms": NA,
    }
    return data.replace(invalid_data_mapping)


def _extract_number_from_brackets(data):
    df = data.str.extract(r"\[(\d+)\]")
    return df.astype(pd.UInt16Dtype())


def _clean_binary_data(data, is_one):
    df = data.apply(lambda x: 1 if x == is_one else 0)
    return df.astype(pd.UInt8Dtype())


def _clean_marital_status(data):
    invalid_marital_data = {
        "[6] Ehepartner im Ausland": NA,
        "[7] Eingetragene gleichgeschlechtliche Partnerschaft zusammenlebend": NA,
        "[8] Eingetragene gleichgeschlechtliche Partnerschaft getrennt lebend": NA,
    }
    df = data.replace(invalid_marital_data)
    return _extract_number_from_brackets(df)
