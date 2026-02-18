#!/usr/bin/env python
# coding: utf-8

import cf_xarray # noqa: F401
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import calendar
import hashlib
import json

import matplotlib.pyplot as plt
import xarray as xr

from big_geo_loader.utils import load_data_from_uri


zpath = "DATASET_CACHE/ecmwf-era5X_oper_an_sfc_2000_2020_2t_repack.kr1.0.zarr"

def load_zarr(zarr_path: str | Path) -> xr.Dataset:
    ds = load_data_from_uri(str(zarr_path), {})
    return ds


def main():
    ds = load_data_from_uri(zpath, {})
    print(ds)
    print(ds.cf["latitude"])
    print("Done loading Zarr dataset")

main()


class DynamicCroissantConverter:
    """Dynamic converter for NASA POWER data to GeoCroissant format"""

    def __init__(
        self,
        zarr_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the converter with the Zarr URL

        Args:
            zarr_url: URL or path to the Zarr dataset (e.g., S3 URL or local path)
            metadata: Optional dictionary of additional metadata to include in the GeoCroissant output
        """
        self.zarr_url = zarr_url
        self.metadata = metadata or {}
        self.ds = None

    def load_dataset(self) -> bool:
        """Load the dataset. Returns True if successful, False otherwise."""
        try:
            print(f"Loading NASA POWER dataset from {self.zarr_url}...")
            self.ds = load_data_from_uri(self.zarr_url, {})
            print(f"Dataset loaded successfully!")
            print(f"  - Dimensions: {self.ds.dims}")
            print(f"  - Total size: {self.ds.nbytes / 1e9:.2f} GB")
            print(f"  - Variables: {len(self.ds.data_vars)}")
            print(
                f"  - Time range: {self.ds.time.values[0]} to"
                f" {self.ds.time.values[-1]}"
            )
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def get_available_variables(self) -> Dict[str, Any]:
        """Get list of available variables with their metadata"""
        if not self.ds:
            return {}

        variables = {}
        for var_name, var in self.ds.data_vars.items():
            variables[var_name] = {
                "shape": list(var.shape),
                "dimensions": list(var.dims),
                "dtype": str(var.dtype),
                "size_mb": float(var.nbytes / 1e6),
                "attributes": (
                    dict(var.attrs) if hasattr(var, "attrs") and var.attrs else {}
                ),
            }
        return variables

    def generate_checksum(self, content: str) -> str:
        """Generate MD5 checksum for content"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def create_croissant_metadata(self, output_file: Optional[str] = "record.json") -> Dict[str, Any]:
        """
        Create GeoCroissant metadata for the data

        Args:
            output_file: Output file path (if None, auto-generated)

        Returns:
            dict: GeoCroissant metadata
        """
        if not self.ds:
            print("Error: No dataset available. Call load_dataset() first.")
            return {}

        # Generate checksum
        hash_input = f"{repr(self.ds.dims)}|{repr(self.ds.data_vars)}|{repr(self.ds.attrs)}"
        md5_hash = self.generate_checksum(hash_input)

        # Create time extent
        if "time" not in self.ds.coords:
            print("Error: 'time' coordinate not found in dataset. Setting dummy time extent.")
            start, end = "01-01-01T00:00:00Z", "9999-12-31T00:00:00Z"
        else:
            time_values = self.ds.time.values
            start, end = time_values[0].strftime("%Y-%m-%dT%H:%M:%SZ"), time_values[-1].strftime("%Y-%m-%dT%H:%M:%SZ")

        # Create GeoCroissant metadata
        croissant = {
            "@context": {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "citeAs": "cr:citeAs",
                "column": "cr:column",
                "conformsTo": "dct:conformsTo",
                "cr": "http://mlcommons.org/croissant/",
                "geocr": "http://mlcommons.org/croissant/geocr/",
                "rai": "http://mlcommons.org/croissant/RAI/",
                "dct": "http://purl.org/dc/terms/",
                "sc": "https://schema.org/",
                "data": {"@id": "cr:data", "@type": "@json"},
                "examples": {"@id": "cr:examples", "@type": "@json"},
                "dataBiases": "cr:dataBiases",
                "dataCollection": "cr:dataCollection",
                "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
                "extract": "cr:extract",
                "field": "cr:field",
                "fileProperty": "cr:fileProperty",
                "fileObject": "cr:fileObject",
                "fileSet": "cr:fileSet",
                "format": "cr:format",
                "includes": "cr:includes",
                "isLiveDataset": "cr:isLiveDataset",
                "jsonPath": "cr:jsonPath",
                "key": "cr:key",
                "md5": "cr:md5",
                "parentField": "cr:parentField",
                "path": "cr:path",
                "personalSensitiveInformation": "cr:personalSensitiveInformation",
                "recordSet": "cr:recordSet",
                "references": "cr:references",
                "regex": "cr:regex",
                "repeated": "cr:repeated",
                "replace": "cr:replace",
                "samplingRate": "cr:samplingRate",
                "separator": "cr:separator",
                "source": "cr:source",
                "subField": "cr:subField",
                "transform": "cr:transform",
            },
            "@type": "sc:Dataset",
            "name": self.metadata.get("name", "Unknown dataset"),
            "alternateName": self.metadata.get("alternateNames", []),
            "description": self.metadata.get("description", "No description provided."),
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "version": "1.0.0",
            "creator": {
                "@type": "Organization",
                "name": self.metadata.get("creator", "Unknown creator"),
                "url": self.metadata.get("creatorUrl", None),
            },
            "url": self.metadata.get("url", None),
            "keywords": self.metadata.get("keywords", []),
            "citeAs": self.metadata.get("citeAs", None),
            "datePublished": self.metadata.get("datePublished", datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "geocr:BoundingBox": [
                ds.cf["latitude"].values.min(),
                ds.cf["latitude"].values.min(),
                ds.cf["longitude"].values.max(),
                ds.cf["latitude"].values.max(),
            ],
            "geocr:temporalExtent": {"startDate": start_date, "endDate": end_date},
            "geocr:spatialResolution": "0.5° lat × 0.625° lon",
            "geocr:coordinateReferenceSystem": "EPSG:4326",
            "geocr:mlTask": {
                "@type": "geocr:Regression",
                "taskType": "climate_prediction",
                "evaluationMetric": "RMSE",
                "applicationDomain": "climate_monitoring",
            },
            "distribution": [
                {
                    "@type": "cr:FileObject",
                    "@id": (
                        f"zarr-store-{year}-{month:02d}"
                        if month
                        else f"zarr-store-{year}"
                    ),
                    "name": (
                        f"zarr-store-{year}-{month:02d}"
                        if month
                        else f"zarr-store-{year}"
                    ),
                    "description": (
                        f"Zarr datacube for NASA POWER data {description_suffix}"
                    ),
                    "contentUrl": self.zarr_url,
                    "encodingFormat": "application/x-zarr",
                    "md5": md5_hash,
                }
            ],
            "recordSet": [
                {
                    "@type": "cr:RecordSet",
                    "@id": (
                        f"nasa_power_data_{year}_{month:02d}"
                        if month
                        else f"nasa_power_data_{year}"
                    ),
                    "name": (
                        f"nasa_power_data_{year}_{month:02d}"
                        if month
                        else f"nasa_power_data_{year}"
                    ),
                    "description": f"NASA POWER climate data {description_suffix}",
                    "field": [],
                }
            ],
        }

        # Add fields for each variable
        fields = croissant["recordSet"][0]["field"]

        # Add coordinate fields
        for coord_name, coord in self.ds_subset.coords.items():
            coord_field = {
                "@type": "cr:Field",
                "@id": (
                    f"nasa_power_data_{year}_{month:02d}/{coord_name}"
                    if month
                    else f"nasa_power_data_{year}/{coord_name}"
                ),
                "name": (
                    f"nasa_power_data_{year}_{month:02d}/{coord_name}"
                    if month
                    else f"nasa_power_data_{year}/{coord_name}"
                ),
                "description": f"Coordinate: {coord_name}",
                "dataType": "sc:Float" if coord.dtype.kind == "f" else "sc:Date",
                "source": {
                    "fileObject": {
                        "@id": (
                            f"zarr-store-{year}-{month:02d}"
                            if month
                            else f"zarr-store-{year}"
                        )
                    },
                    "extract": {"jsonPath": f"$.{coord_name}"},
                },
                "geocr:dataShape": list(coord.shape),
                "geocr:validRange": (
                    {
                        "min": (
                            -90.0
                            if coord_name == "lat"
                            else -180.0
                            if coord_name == "lon"
                            else None
                        ),
                        "max": (
                            90.0
                            if coord_name == "lat"
                            else 180.0
                            if coord_name == "lon"
                            else None
                        ),
                    }
                    if coord_name in ["lat", "lon"]
                    else None
                ),
                "geocr:units": (
                    "degrees_north"
                    if coord_name == "lat"
                    else "degrees_east"
                    if coord_name == "lon"
                    else None
                ),
            }
            # Remove None values
            coord_field = {k: v for k, v in coord_field.items() if v is not None}
            fields.append(coord_field)

        # Add data variable fields
        for var_name, var in self.ds_subset.data_vars.items():
            var_field = {
                "@type": "cr:Field",
                "@id": (
                    f"nasa_power_data_{year}_{month:02d}/{var_name}"
                    if month
                    else f"nasa_power_data_{year}/{var_name}"
                ),
                "name": (
                    f"nasa_power_data_{year}_{month:02d}/{var_name}"
                    if month
                    else f"nasa_power_data_{year}/{var_name}"
                ),
                "description": var.attrs.get("long_name", var_name),
                "dataType": "sc:Float",
                "source": {
                    "fileObject": {
                        "@id": (
                            f"zarr-store-{year}-{month:02d}"
                            if month
                            else f"zarr-store-{year}"
                        )
                    },
                    "extract": {"jsonPath": f"$.{var_name}"},
                },
                "geocr:dataShape": list(var.shape),
                "geocr:validRange": (
                    {
                        "min": float(var.attrs.get("valid_min", 0.0)),
                        "max": float(var.attrs.get("valid_max", 100.0)),
                    }
                    if var.attrs.get("valid_min") is not None
                    and var.attrs.get("valid_max") is not None
                    else None
                ),
                "geocr:units": var.attrs.get("units", ""),
                "geocr:standardName": var.attrs.get("standard_name", ""),
                "geocr:definition": var.attrs.get("definition", ""),
                "geocr:cellMethods": var.attrs.get("cell_methods", ""),
            }
            # Remove None values
            var_field = {k: v for k, v in var_field.items() if v is not None}
            fields.append(var_field)

        # Save metadata
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(croissant, f, indent=2, ensure_ascii=False)

        print(f"GeoCroissant metadata saved to {output_file}")
        print(f"Total fields: {len(fields)}")

        return croissant

    def convert(
        self,
        year: int,
        month: Optional[int] = None,
        variables: Optional[list] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete conversion pipeline

        Args:
            year: Year to convert
            month: Month to convert (1-12), if None converts entire year
            variables: List of variables to include, if None includes all
            output_file: Output file path, if None auto-generated

        Returns:
            dict: GeoCroissant metadata
        """
        print(
            "Starting conversion for"
            f" {calendar.month_name[month] if month else 'year'} {year}..."
        )

        # Load dataset
        if not self.load_dataset():
            return {}

        # Subset data
        if not self.subset_data(year, month, variables):
            return {}

        # Generate metadata
        metadata = self.create_croissant_metadata(year, month, output_file)

        print("Conversion completed successfully!")
        return metadata

# In[13]:

print("\n\n\n----------------------------------\n  DEMONSTRATE USAGE\n----------------------------------\n")
converter = DynamicCroissantConverter()

# July 2021 (as we just demonstrated)
converter.convert(year=2021, month=7)

print("mlcroissant validate --jsonld=NASA_POWER_2021_07_croissant.json")

# Load metadata
with open("NASA_POWER_2021_07_croissant.json", "r") as f:
    metadata = json.load(f)

# Extract Zarr URL from metadata
zarr_url = None
for dist in metadata.get("distribution", []):
    if dist.get("encodingFormat") == "application/x-zarr":
        zarr_url = dist.get("contentUrl")
        break

print(f"Loading data from: {zarr_url}")

# Load data directly from S3
ds = xr.open_zarr(zarr_url, storage_options={"anon": True})

# Extract time range from metadata for subsetting
temporal_extent = metadata.get("geocr:temporalExtent", {})
start_date = temporal_extent.get("startDate", "").split("T")[0]
end_date = temporal_extent.get("endDate", "").split("T")[0]

if start_date and end_date:
    print(f"Subsetting data for {start_date} to {end_date}")
    ds = ds.sel(time=slice(start_date, end_date))

# Plot T2M Temperature
if "T2M" in ds.data_vars:
    fig, ax = plt.subplots(figsize=(12, 8))
    data = ds["T2M"].isel(time=0)
    im = data.plot(ax=ax, cmap="RdYlBu_r", robust=True)
    ax.set_title("Temperature (T2M) - 2020", fontweight="bold", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"T2M plot complete using metadata: NASA_POWER_2020_croissant.json")
    print(f"   - Data source: {zarr_url}")
    print(f"   - Time period: {start_date} to {end_date}")
else:
    print("Error: T2M variable not found in dataset")
