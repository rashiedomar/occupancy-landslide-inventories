# Data

The MIT licence in the repository root covers the code only. The grids here are derived from third-party data and stay under the terms of their sources.

## Included: `processed/`

| File | Contents | Derived from |
|---|---|---|
| `iburi_grid.npz` | 120 m grid of terrain, road-distance and built-up layers, plus the reference landslide raster | Zhang et al. (2019) inventory, Copernicus GLO-30, ESA WorldCover 2021, OpenStreetMap |
| `gorkha_grid.npz` | 210 m grid, same layers | Roback et al. (2018) inventory, USGS data release by Zekkos et al. (2017), Copernicus GLO-30, ESA WorldCover 2021, OpenStreetMap |
| `haiti_grid.npz` | 210 m grid, same layers | Harp et al. (2016) inventory (USGS), Copernicus GLO-30, ESA WorldCover 2021, OpenStreetMap |
| `gorkha_real_obs.npz` | Rasters of the Gnyawali, Valagussa and Zhang inventories, with coverage masks | USGS Ground Failure Database v4 (Schmitt et al., 2017) |

Attribution required by the sources:

- **Iburi inventory:** Zenodo record [2577300](https://zenodo.org/records/2577300), CC BY 4.0.
- **Road-distance layers:** © OpenStreetMap contributors, Open Database License ([ODbL](https://www.openstreetmap.org/copyright)).
- **Terrain layers:** produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.
- **Built-up density:** © ESA WorldCover project 2021, contains modified Copernicus Sentinel data, CC BY 4.0.

Full references are in the paper's bibliography.

## Not included: raw downloads

The raw data are about 2 GB. `manifests/` lists the source URLs, sizes and MD5 checksums of the Iburi, Haiti and Northridge downloads. Northridge is not used in the paper.

To rebuild a grid, install `requirements-prep.txt` and run, from `real_terrain/`:

```bash
python prep_iburi.py ../data/raw ../data/processed

python prep_generic.py --inventory .../Source20170209.shp --extent .../MappingExtent20170209.shp \
    --obscured .../ObscuredAreas20170209.shp --window 27.65,85.40,28.05,86.10 --dem-dir ../data/raw_gorkha/dem \
    --roads ../data/raw_gorkha/osm/roads_q1.json ../data/raw_gorkha/osm/roads_q2.json \
    --landcover ../data/raw_gorkha/landcover/*.tif --crs EPSG:32645 --cell 210 \
    --epicenters "28.231,84.731;27.809,86.066" --out ../data/processed/gorkha_grid.npz
```

OpenStreetMap changes over time, so rebuilt road layers will differ slightly from the ones used here. The retrieval date is in the paper's references.
