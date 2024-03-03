import numpy as np
import pandas as pd
from pandas import NA
from sklearn.preprocessing import OneHotEncoder


def clean_data(merged_data):
    valid_data = pd.DataFrame()
    valid_data = merged_data[
        (merged_data["syear"] >= 2010) & (merged_data["syear"] <= 2018)
    ]
    valid_data = _clean_invalid_data(valid_data)
    valid_data["hghinc"] = _positive_number_only(valid_data["hghinc"]).astype(
        pd.Float64Dtype(),
    )

    df = pd.DataFrame()
    df["pid"] = valid_data["pid"]
    df["hid"] = valid_data["hid"]
    df["syear"] = valid_data["syear"].astype(pd.UInt16Dtype())

    df["birthyear"] = _positive_number_only(valid_data["gebjahr"]).astype(
        pd.UInt16Dtype(),
    )
    df["age"] = np.where(
        (valid_data["syear"] - valid_data["gebjahr"]) < 130,
        valid_data["syear"] - valid_data["gebjahr"],
        NA,
    )
    df["agesquared"] = df["age"] ** 2
    df["bmi"] = _positive_number_only(valid_data["bmi"])
    df["education"] = valid_data["pgbilzeit"].astype(pd.Float32Dtype())
    df["logincome"] = np.log(valid_data["hghinc"])
    df["health"] = valid_data["m11127"].astype(pd.UInt16Dtype())
    df["workinghours"] = valid_data["plb0183"].astype(pd.Float64Dtype())
    df["numberofpeople"] = (
        valid_data["hhgr"]
        .replace("[0] Aufgeloeste/n.bearbeitete Haushalte", NA)
        .astype(pd.UInt8Dtype())
    )
    df["numberofchildren"] = valid_data["k_nrkid"].astype(pd.UInt8Dtype())

    df["smonth"] = valid_data["pmonin"]
    df["birthregion"] = valid_data["birthregion"]
    df["migback"] = valid_data["migback"]
    df["housingstatus"] = valid_data["hlf0001_v3"]
    df["maritalstatus"] = _clean_marital_status(valid_data["pgfamstd"])
    df["religion"] = valid_data["plh0258_h"]

    df["disability"] = _extract_number_from_brackets(valid_data["m11124"])
    df["labourstatus"] = _clean_binary_data(
        valid_data["pglfs"],
        is_one="[11] Erwerbstätig",
    )
    df["sex"] = _extract_number_from_brackets(valid_data["sex"]) - 1

    df["lifesatisfaction"] = _extract_number_from_brackets(valid_data["plh0182"])

    return df


def convert_categorical_to_dummy(data, columns):
    df = data.drop(columns, axis=1)
    encoder = OneHotEncoder(sparse_output=False)
    df_dummy = pd.DataFrame(
        encoder.fit_transform(data[columns]),
        columns=encoder.get_feature_names_out(columns),
        index=data.index,
    )
    return pd.concat([df, df_dummy], axis=1)


def _clean_invalid_data(data):
    invalid_data_mapping = {
        "[-1] keine Angabe": NA,
        "[-2] trifft nicht zu": NA,
        "[-3] nicht valide": NA,
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
    return data.replace(invalid_marital_data)


def _positive_number_only(data):
    return data.where(data > 0, NA)
