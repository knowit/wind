import polars as pl

windpark_lookup = pl.read_csv("data/windparks_lookup.csv")

windparks_nve = pl.read_csv(
    "data/windparks_nve.csv", separator=";", decimal_comma=True
).with_columns(
    pl.col("Middelproduksjon [GWh]").str.replace_all(" ", "").cast(pl.Int64),
    windpark_nve=pl.col("Kraftverknavn"),
    windpark_nve_id=pl.col("KraftverkID"),
)
windparks = (
    pl.read_csv("data/windparks_bidzone.csv", try_parse_dates=True)
    .join(windpark_lookup, on="eic_code", how="inner")
    .with_columns(windpark_name=pl.col("name"))
)
windparks_match = windparks.join(
    windparks_nve, left_on="windpark_name", right_on="windpark_nve", how="left"
)

mw_per_turbine = windparks_match.select(
    (pl.col("Installert effekt [MW]") / pl.col("Antall turbiner")).mean()
).item()
GWh_per_MW = windparks_match.select(
    (pl.col("Middelproduksjon [GWh]") / pl.col("Installert effekt [MW]")).mean()
).item()

windparks_enriched = (
    windparks_match.with_columns(
        operating_power_max=pl.coalesce("Installert effekt [MW]", "operating_power_max")
    )
    .with_columns(
        num_turbines=pl.coalesce(
            "Antall turbiner", pl.col("operating_power_max") / mw_per_turbine
        ),
        mean_production=pl.col("operating_power_max") * GWh_per_MW,
    )
    .select(
        "bidding_area",
        "substation_name",
        "windpark_name",
        "windpark_nve_id",
        "prod_start_new",
        "Fylke",
        "Kommune",
        "operating_power_max",
        "mean_production",
        "num_turbines",
    )
)

windparks_enriched.write_csv("data/windparks_enriched.csv")
