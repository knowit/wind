import cf_xarray as cfxr
import numpy as np
import polars as pl
import torch
import xarray as xr

from model import SharedPerLocationSum

ckpt_path = "checkpoints/wind_no_workers_last.pth"
data_path = "data/dataset_all_zones.zarr"

ckpt = torch.load(ckpt_path)
model_kwargs = ckpt.get("model_kwargs")
model = SharedPerLocationSum(**model_kwargs)
model.load_state_dict(ckpt["state_dict"])
model.eval()

encoded = xr.open_dataset(data_path)
ds = cfxr.decode_compress_to_multi_index(encoded, "forecast_index")
ds_newest_time_ref = ds.time_ref.max()
ds_newest = ds.sel(time_ref=ds_newest_time_ref)
X_newest = torch.from_numpy(ds_newest["X"].values)
time = ds_newest.time.values
bidding_area = ds_newest.bidding_area.values

x_mean = ckpt.get("x_mean")
x_std = ckpt.get("x_std")
X_norm = (X_newest - x_mean) / x_std

with torch.no_grad():
    preds = model(X_norm)

df_pred = (
    pl.DataFrame(
        {
            "y_pred": preds,
            "time": time,
            "bidding_area": bidding_area,
        }
    )
    .pivot("bidding_area", index="time", values="y_pred")
    .select("time", "ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4")
)
forecast_date = str(ds_newest_time_ref.values.astype("datetime64[D]"))
output_path = f"output/knowit_{forecast_date}.csv"
df_pred.write_csv(output_path)
