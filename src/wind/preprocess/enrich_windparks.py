import geopandas as gpd
import polars as pl
import requests
import xarray as xr

from wind.preprocess.prepare_local_data import get_local_windpower


def get_windpark_metadata():
    data = requests.get(
        "https://api.nve.no/web/WindPowerplant/GetWindPowerPlants"
    ).json()
    df = pl.json_normalize(data).select(
        windpark_nve_id="VindkraftAnleggId",
        production_start_date=pl.col("IdriftsettelseForsteByggetrinn").cast(
            pl.Datetime
        ),
        operating_power_max="InstallertEffekt_MW",
        province="Fylke",
        county="Kommune",
        mean_hub_height=pl.col("GjsnittNavhoeyde").fill_null(strategy="mean"),
        mean_production="GjsnittGeneratorytelse",
        mean_rotor_diameter="GjsnittRotordiameter",
        num_turbines="AntallOperativeTurbiner",
    )
    return df


def get_geo_info() -> pl.DataFrame:
    windparks_geo = gpd.read_file("data/NVEData.gdb/")
    windparks_geo["has_start_date"] = ~windparks_geo["idriftDato"].isnull()

    windpark_lookup = (
        pl.read_csv("data/windparks_lookup.csv")
        .select("windpark_nve_id", "windpark_nve")
        .filter(pl.row_index().over("windpark_nve_id") == 0)
        .with_columns(
            pl.col("windpark_nve").replace(
                {
                    "Nye Sandøy": "Sandøy",
                    "Valsneset vindkraftverk": "Valsneset",
                    "Raggovidda 2": "Raggovidda",
                }
            )
        )
        .to_pandas()
    )
    windparks_geo = windparks_geo.merge(
        windpark_lookup, how="inner", left_on="saksTittel", right_on="windpark_nve"
    )
    windparks_geo = windparks_geo.sort_values(
        ["windpark_nve_id", "has_start_date"]
    ).drop_duplicates("windpark_nve_id", keep="first")

    topo = xr.open_dataarray("data/topography.tif").squeeze()
    coords = windparks_geo.geometry.get_coordinates()
    xi = xr.DataArray(
        coords.x.values, dims="point", coords={"point": windparks_geo.index}, name="x"
    )
    yi = xr.DataArray(
        coords.y.values, dims="point", coords={"point": windparks_geo.index}, name="y"
    )
    elevation = topo.interp(
        x=xi,
        y=yi,
    )

    rix = xr.open_dataarray(
        "data/Vindressurs/Vindressurs_Terrengkompleksitet.tif"
    ).squeeze()
    coords = windparks_geo.geometry.to_crs(
        rix.spatial_ref.attrs["crs_wkt"]
    ).get_coordinates()
    xi = xr.DataArray(
        coords.x.values, dims="point", coords={"point": windparks_geo.index}, name="x"
    )
    yi = xr.DataArray(
        coords.y.values, dims="point", coords={"point": windparks_geo.index}, name="y"
    )
    ruggedness = rix.interp(
        x=xi,
        y=yi,
    ).fillna(0)  # METCentre Karmøy is missing ruggedess since it is offshore.
    # Ruggedness at sea is 0

    geo_info = pl.DataFrame(
        {
            "windpark_nve_id": windparks_geo["windpark_nve_id"].values,
            "elevation": elevation.values,
            "ruggedness": ruggedness.values,
        }
    )
    return geo_info


def get_actual_max_output():
    return (
        get_local_windpower("data/windpower2002-2024_utcplus1.csv")
        .group_by("windpark_nve_id")
        .agg(actual_max_output=pl.col("local_power").max())
    )


windparks = get_windpark_metadata()
windpark_lookup = (
    pl.read_csv("data/windparks_lookup.csv")
    .select("windpark_nve_id", "windpark_nve", "windpark_statnet", "bidding_area")
    .filter(pl.row_index().over("windpark_nve_id") == 0)
)
geo_info = get_geo_info()

actual_max_output = get_actual_max_output().collect()

windparks_enriched = (
    windpark_lookup.join(windparks, on="windpark_nve_id", how="left")
    .join(geo_info, on="windpark_nve_id", how="left")
    .join(actual_max_output, on="windpark_nve_id", how="left")
    .select(
        "windpark_nve_id",
        "windpark_nve",
        "windpark_statnet",
        "bidding_area",
        "province",
        "county",
        "production_start_date",
        pl.col("operating_power_max").alias("reported_operating_power_max"),
        pl.max_horizontal(
            "operating_power_max",
            "actual_max_output",
        ).alias("operating_power_max"),
        "mean_hub_height",
        "mean_production",
        "mean_rotor_diameter",
        "num_turbines",
        "elevation",
        "ruggedness",
    )
)

windparks_enriched.write_csv("data/windparks_enriched.csv")
