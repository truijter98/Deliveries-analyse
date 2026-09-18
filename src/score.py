"""Scoort nieuwe orders op het risico van te late levering.

Draaien vanuit de projectmap:
    python src/score.py
"""

from pathlib import Path
import pandas as pd
import joblib

BASIS = Path(__file__).resolve().parent.parent
MODEL = BASIS / "modellen" / "levering_model.joblib"
INVOER = BASIS / "Data" / "nieuwe_orders.csv"
UITVOER = BASIS / "Data" / "risicolijst.csv"
CONTROLE = BASIS / "Data" / "handmatig_controleren.csv"

VERPLICHT = ["aantal", "stuksprijs", "korting_pct", "beloofde_levertijd"]


def main():
    bundel = joblib.load(MODEL)
    pijplijn = bundel["pijplijn"]
    kolommen = bundel["kolommen"]
    drempel = bundel["drempel"]

    orders = pd.read_csv(INVOER)
    orders["datum_besteld"] = pd.to_datetime(orders["datum_besteld"])
    orders["maand"] = orders["datum_besteld"].dt.month
    orders["weekdag"] = orders["datum_besteld"].dt.dayofweek

    ontbreekt = [k for k in kolommen if k not in orders.columns]
    if ontbreekt:
        raise SystemExit(f"FOUT: deze kolommen ontbreken in de invoer: {ontbreekt}")

    onvolledig = orders[VERPLICHT].isna().any(axis=1)
    if onvolledig.any():
        orders.loc[onvolledig].to_csv(CONTROLE, index=False)
        print(f"LET OP: {onvolledig.sum()} orders onvolledig, apart gezet")
        orders = orders[~onvolledig].copy()

    orders["risico"] = pijplijn.predict_proba(orders[kolommen])[:, 1]
    orders["waarschuwing"] = (orders["risico"] >= drempel).astype(int)

    orders = orders.sort_values("risico", ascending=False)
    orders.to_csv(UITVOER, index=False)

    print(f"model versie {bundel['versie']}, drempel {drempel}")
    print(f"{len(orders)} orders gescoord")
    print(f"{orders['waarschuwing'].sum()} waarschuwingen")
    print(f"weggeschreven naar {UITVOER.name}")


if __name__ == "__main__":
    main()