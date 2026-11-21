# Weather data and model sources

The forecast starts at 00:00 UTC on 1 September 2022. FourCastNet takes the initial ERA5 state and predicts eight six-hour steps. The ERA5 fields for later times serve as the reference for each forecast.

## FourCastNet and Earth2Studio

The workshop uses NVIDIA's 26-channel FourCastNet1 checkpoint at revision `c67a63995f6c8e0e557eb3d791f32f437e9b02d5`. The model card lists the license as Apache-2.0. The setup script downloads the checkpoint and normalization files and verifies their SHA-256 hashes. [NVIDIA FourCastNet1](https://huggingface.co/nvidia/fourcastnet1)

Normalization, channel order and recurrent inference follow NVIDIA's FCN wrapper. Each six-hour step predicts the full normalized state. [Earth2Studio FCN](https://github.com/NVIDIA/earth2studio/blob/486c5daa98841b0ce93cb78cb70d0dee7a8a56e6/earth2studio/models/px/fcn.py)

The relative humidity calculation adapts `DerivedRH` from Earth2Studio, copyright © 2024–2026 NVIDIA CORPORATION & AFFILIATES, under Apache-2.0. The NumPy implementation uses ERA5 temperature (K), specific humidity (kg/kg) and pressure (hPa) in the IFS mixed-phase formula. It applies the original 0–100% bounds and stores relative humidity in single precision. [DerivedRH source](https://github.com/NVIDIA/earth2studio/blob/486c5daa98841b0ce93cb78cb70d0dee7a8a56e6/earth2studio/models/dx/derived.py) · [License](../../../LICENSE)

## ERA5 reanalysis

The Copernicus Climate Change Service at ECMWF produces ERA5. The workshop reads the public Google ARCO-ERA5 archive. [ERA5 data license](https://cds.climate.copernicus.eu/licences/cc-by) · [ARCO-ERA5 citation guidance](https://github.com/google-research/arco-era5#how-to-cite-this-work)

`era5-source-plan.json` identifies the source objects by generation, byte count and checksum. `era5-source-metadata.json` records the dimensions and units. `prepare.py` checks these manifests and the downloaded source files. It selects the model's channels and pressure levels, removes the south-pole row to fit the 720 × 1440 grid, and derives relative humidity. The cache contains the resulting arrays and a record of these transformations in `data-20220901/provenance.json`.

## Forecast evaluation

At each of the eight lead times, the lab compares the forecast and the initial-state persistence baseline with ERA5. The global and East Asia error scores use cosine-latitude weights. The 24- and 48-hour thresholds for this case are in [weather-acceptance.json](weather-acceptance.json).

## Coastline

`coastline.geojson` contains the Natural Earth 1:110m public-domain coastline. Its SHA-256 hash is `851f581ff5ffb844deed8ae1a9ce22e3c4bb3d74fa342cadb5d8e39b41ae7c3c`. [Source file](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_coastline.geojson) · [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/)
