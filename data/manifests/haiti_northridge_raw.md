# Raw extra geodata manifest

Download and validation date: 2026-11-02 UTC.

## Files

MANIFEST.md is excluded from its own table because it cannot contain a stable MD5 of itself.

| Relative path | Source URL | Date (UTC) | Size (bytes) | MD5 | Verification |
|---|---|---:|---:|---|---|
| haiti/dem/Copernicus_DSM_COG_10_N18_00_W072_00_DEM.tif | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N18_00_W072_00_DEM/Copernicus_DSM_COG_10_N18_00_W072_00_DEM.tif | 2026-11-02 | 41781845 | d06cdf184a658922698157554a35b7ac | GTiff; EPSG:4326; 3600x3600; bounds -72.0001388889,18.0001388889,-71.0001388889,19.0001388889 |
| haiti/dem/Copernicus_DSM_COG_10_N18_00_W073_00_DEM.tif | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N18_00_W073_00_DEM/Copernicus_DSM_COG_10_N18_00_W073_00_DEM.tif | 2026-11-02 | 27136394 | e40f4c96735e66aa5b9bef0fa04493a2 | GTiff; EPSG:4326; 3600x3600; bounds -73.0001388889,18.0001388889,-72.0001388889,19.0001388889 |
| haiti/dem/Copernicus_DSM_COG_10_N18_00_W074_00_DEM.tif | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N18_00_W074_00_DEM/Copernicus_DSM_COG_10_N18_00_W074_00_DEM.tif | 2026-11-02 | 18882561 | 3daf9537f958fc5bfb6b07cb5d457f92 | GTiff; EPSG:4326; 3600x3600; bounds -74.0001388889,18.0001388889,-73.0001388889,19.0001388889 |
| haiti/landcover/ESA_WorldCover_10m_2021_v200_N18W072_Map.tif | https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N18W072_Map.tif | 2026-11-02 | 51090521 | c1704cc4844765381c601adb5d0ce2f5 | GTiff; EPSG:4326; 36000x36000; bounds -72,18,-69,21 |
| haiti/landcover/ESA_WorldCover_10m_2021_v200_N18W075_Map.tif | https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N18W075_Map.tif | 2026-11-02 | 35189570 | 3466045cd8c9e1b29e82247b855a6aed | GTiff; EPSG:4326; 36000x36000; bounds -75,18,-72,21 |
| haiti/osm/roads_q1.json | https://overpass-api.de/api/interpreter | 2026-11-02 | 17714831 | dfc10392232af3fb5910fa74eb6c590c | JSON parsed; 16,863 ways; bbox 18.10,-73.10,18.60,-72.50 |
| haiti/osm/roads_q2.json | https://overpass-api.de/api/interpreter | 2026-11-02 | 24178899 | 0e965b940a09be9c62a3fae25b8a2138 | JSON parsed; 30,448 ways; bbox 18.10,-72.50,18.60,-71.95 |
| northridge/dem/Copernicus_DSM_COG_10_N34_00_W119_00_DEM.tif | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N34_00_W119_00_DEM/Copernicus_DSM_COG_10_N34_00_W119_00_DEM.tif | 2026-11-02 | 42718799 | 7159b3ec2537eebe6eb1c8430bb07403 | GTiff; EPSG:4326; 3600x3600; bounds -119.0001388889,34.0001388889,-118.0001388889,35.0001388889 |
| northridge/dem/Copernicus_DSM_COG_10_N34_00_W120_00_DEM.tif | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N34_00_W120_00_DEM/Copernicus_DSM_COG_10_N34_00_W120_00_DEM.tif | 2026-11-02 | 31815801 | 7bafcad85909e5e9208270d66b0d7257 | GTiff; EPSG:4326; 3600x3600; bounds -120.0001388889,34.0001388889,-119.0001388889,35.0001388889 |
| northridge/landcover/ESA_WorldCover_10m_2021_v200_N33W120_Map.tif | https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N33W120_Map.tif | 2026-11-02 | 101202142 | cfa9304106d1c2b28c50590953ecf5de | GTiff; EPSG:4326; 36000x36000; bounds -120,33,-117,36 |
| northridge/osm/roads_q1.json | https://overpass-api.de/api/interpreter | 2026-11-02 | 56340578 | 48ddcae33e0258714dce7bb2f15e0798 | JSON parsed; 64,508 ways; southwest bbox 34.00,-119.10,34.35,-118.62 |
| northridge/osm/roads_q2.json | https://overpass-api.de/api/interpreter | 2026-11-02 | 182134345 | 1ce52d9f43b946e10792de451d5536e1 | JSON parsed; 291,804 ways; southeast bbox 34.00,-118.62,34.35,-118.15 |


## NASA landslide report catalogs

Requested paging parameters were tested against both supplied ArcGIS REST query endpoints, using GET and form-encoded POST. Both services returned HTTP 404 before any GeoJSON page was available.

| Area | Requested output | Service A | Service B fallback | Saved file | Feature count | landslide_trigger counts | event_date range |
|---|---|---|---|---|---|---|---|
| Gorkha | reports/gorkha_reports.geojson | HTTP 404 | HTTP 404 | Not created | Unavailable | Unavailable | Unavailable |
| Iburi | reports/iburi_reports.geojson | HTTP 404 | HTTP 404 | Not created | Unavailable | Unavailable | Unavailable |
| Haiti | reports/haiti_reports.geojson | HTTP 404 | HTTP 404 | Not created | Unavailable | Unavailable | Unavailable |

Service A: https://maps.disasters.nasa.gov/ags01/rest/services/Hosted/nasa_glc_poly_point/FeatureServer/0/query

Service B: https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/COOLR_Reports_Points/FeatureServer/0/query

No alternative catalog was substituted and no empty GeoJSON was fabricated.

## Problems

- The supplied NASA Global Landslide Catalog and COOLR ArcGIS query endpoints both returned HTTP 404 for every requested envelope. The three report GeoJSON files could not be created, so feature counts, landslide_trigger counts, and event_date ranges are unavailable.
- Kumi Overpass returned HTTP 504 for both Haiti queries; the specified overpass-api.de fallback succeeded.
- Kumi Overpass returned HTTP 502 for Northridge quadrants 1 and 2; the specified overpass-api.de fallback succeeded.
- Northridge quadrants 3 and 4 were not downloaded because the user explicitly asked to skip the rest of Northridge and proceed to the NASA catalog.
- All eight requested raster files opened successfully as GeoTIFFs in EPSG:4326. All four completed OSM files parsed successfully as JSON.

