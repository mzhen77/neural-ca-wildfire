"""
Data Extractor for HDF5 Fire Dataset

Extracts dataset from the HDF5 file and saves it as numpy files.
Static datasets (without timestamps) are saved directly.
Time-series datasets (with timestamps) are concatenated along a new time dimension.
"""

from pathlib import Path

import h5py
import numpy as np


def extract_data(
    hdf5_path: str = "data/hdf5/dataset.hdf5",
    out_dir: str = "data/data_real",
    data_name: str = "Bear_2020",
) -> dict:
    """
    Extract data from HDF5 file and save as numpy files.

    Args:
        hdf5_path: Path to the HDF5 file
        out_dir: Output directory for numpy files

    Returns:
        Dictionary with dataset names and their shapes for README generation
    """
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Open HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        hdf5_group = f[data_name]

        # Extract metadata from HDF5 attributes
        fire_year = hdf5_group.attrs.get("fire_year", 2020)
        grid_height = hdf5_group.attrs.get("height", 0)
        grid_width = hdf5_group.attrs.get("width", 0)

        # Dictionary to store metadata for README
        metadata = {
            "static": {},
            "time_series": {},
            "fire_year": int(fire_year),
            "grid_height": int(grid_height),
            "grid_width": int(grid_width),
        }

        for key in hdf5_group.keys():
            item = hdf5_group[key]

            if isinstance(item, h5py.Dataset):
                # Static dataset (no timestamps)
                arr = item[:]
                np.save(out_path / f"{key}.npy", arr)
                metadata["static"][key] = {
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                    "description": get_variable_description(key),
                }
                print(f"Saved static dataset: {key}.npy, shape={arr.shape}")

            elif isinstance(item, h5py.Group):
                # Time-series dataset (has timestamps)
                # Sort timestamps to ensure chronological order
                timestamps = sorted(item.keys())

                # Stack all timestamps along axis 0
                arrays = [item[ts][:] for ts in timestamps]
                stacked_data = np.stack(arrays, axis=0)

                np.save(out_path / f"{key}.npy", stacked_data)

                # Also save timestamps as a separate file for reference
                timestamps_array = np.array(timestamps, dtype="U10")
                np.save(out_path / f"{key}_timestamps.npy", timestamps_array)

                metadata["time_series"][key] = {
                    "shape": stacked_data.shape,
                    "dtype": str(stacked_data.dtype),
                    "timestamps": timestamps,
                    "description": get_variable_description(key),
                }
                print(
                    f"Saved time-series dataset: {key}.npy, shape={stacked_data.shape} ({len(timestamps)} timestamps)"
                )

        return metadata


def get_variable_description(name: str) -> str:
    """
    Returns a human-readable description for each variable based on raw_dataset_desc.txt.
    """
    descriptions = {
        # LANDFIRE data products (230 prefix = 2023 version)
        "230CBD": "Density of available canopy fuel in a stand.",
        "230CC": "Proportion of the forest floor covered by the vertical projection of the tree crowns.",
        "230CH": "Average height of the top of the vegetated canopy",
        "230EVC": "Vertically projected percent cover of the dominant vegetation for a specific area.",
        "230EVH": "Average height of the dominant vegetation.",
        "230EVT": "Narrow sets of diagnostic plant species, including dominants and co-dominants, broadly similar composition, and diagnostic growth forms classified using the Ecological Systems Classification.",
        "230FBFM40": "A set of fire behavior fuel models that increases prediction accuracy by providing more models in the fuel types (grass, shrub, timber, slash) than Anderson's 13, captures moisture variations and unique fuel differences, allows user to plan or illustrate the effects of multiple or varying fuel and fire scenarios beyond the severe fire season, such as prescribed fire and fire use applications.",
        "230FVC": "Represents a modified version of EVC and more accurately leverages fuel transition assignments related to disturbed areas to properly align with logic developed from Fuels Calibration Workshops.",
        "230FVH": "Represents a modified version of EVH and more accurately leverages fuel transition assignments related to disturbed areas to properly align with logic developed from Fuels Calibration Workshops.",
        "230FVT": "Represents a modified version of EVT that re-establishes pre-disturbance vegetation in disturbed areas, allowing the application of fuel model transitions to properly align with logic developed from Fuels Calibration Workshops.",
        # Topographic data (2020 suffix = year of data)
        "ASP2020": "Azimuth of the sloped surfaces across a landscape.",
        "ELEV2020": "Land height above mean sea level.",
        "SLPD2020": "Percent change of elevation over a specific area.",
        "SLPP2020": "Percent change of elevation over a specific area.",
        # Fire progression data
        "fire": "Fire perimeter progression by date",
        # ERA5-Land weather data
        "leaf_area_index_high_vegetation": "One-half of the total green leaf area per unit horizontal ground surface area for high vegetation type.",
        "leaf_area_index_low_vegetation": "One-half of the total green leaf area per unit horizontal ground surface area for low vegetation type.",
        "surface_pressure": "Pressure (force per unit area) of the atmosphere on the surface of land, sea and in-land water. It is a measure of the weight of all the air in a column vertically above the area of the Earth's surface represented at a fixed point.",
        "temperature_2m": "Temperature of air at 2m above the surface of land, sea or in-land waters. 2m temperature is calculated by interpolating between the lowest model level and the Earth's surface, taking account of the atmospheric conditions.",
        "total_precipitation_sum": "Accumulated liquid and frozen water, including rain and snow, that falls to the Earth's surface. It is the sum of large-scale precipitation and convective precipitation.",
        "u_component_of_wind_10m": "Eastward component of the 10m wind. It is the horizontal speed of air moving towards the east, at a height of ten meters above the surface of the Earth, in meters per second.",
        "v_component_of_wind_10m": "Northward component of the 10m wind. It is the horizontal speed of air moving towards the north, at a height of ten meters above the surface of the Earth, in meters per second.",
    }
    return descriptions.get(name, "No description available")


def generate_readme(metadata: dict, out_dir: str, data_name: str) -> None:
    """
    Generate a README.md file with data descriptions and shapes.
    """
    out_path = Path(out_dir)

    # Extract grid information and year from metadata
    fire_year = metadata.get("fire_year", 2020)
    grid_height = metadata.get("grid_height", 0)
    grid_width = metadata.get("grid_width", 0)

    readme_content = """# {data_name} Fire Dataset

This directory contains extracted data from the {data_name} Fire in California, USA.
The data was extracted from the PyTorchFire HDF5 dataset.

## Dataset Overview

- **Fire Name**: {data_name}
- **Year**: {fire_year}
- **Grid Size**: {grid_height} × {grid_width} pixels
- **Spatial Resolution**: ~30m (LANDFIRE resolution)

---

## Static Datasets (No Temporal Dimension)

These datasets contain time-invariant terrain and vegetation characteristics.

| File | Shape | Data Type | Description |
|------|-------|-----------|-------------|
""".format(data_name=data_name, fire_year=fire_year, grid_height=grid_height, grid_width=grid_width)

    # Add static datasets to table
    for name, info in sorted(metadata["static"].items()):
        shape_str = str(info["shape"])
        readme_content += (
            f"| `{name}.npy` | {shape_str} | {info['dtype']} | {info['description']} |\n"
        )

    readme_content += """
---

## Time-Series Datasets (With Temporal Dimension)

These datasets contain daily observations throughout the fire event.
The first dimension (axis 0) represents time (days).

| File | Shape | Data Type | # Timestamps | Description |
|------|-------|-----------|--------------|-------------|
"""

    # Add time-series datasets to table
    for name, info in sorted(metadata["time_series"].items()):
        n_timestamps = len(info["timestamps"])
        shape_str = str(info["shape"])
        readme_content += f"| `{name}.npy` | {shape_str} | {info['dtype']} | {n_timestamps} | {info['description']} |\n"

    # Add timestamp details
    readme_content += """
### Timestamp Files

For each time-series dataset, a corresponding `*_timestamps.npy` file contains the date strings
in chronological order. These can be loaded to map array indices to dates.

| Time-Series | Date Range | Timestamps |
|-------------|------------|------------|
"""

    for name, info in sorted(metadata["time_series"].items()):
        timestamps = info["timestamps"]
        date_range = f"{timestamps[0]} to {timestamps[-1]}"
        readme_content += f"| `{name}` | {date_range} | {len(timestamps)} |\n"

    readme_content += """
---

## Usage Example

```python
import numpy as np

# Load static terrain data
elevation = np.load('ELEV2020.npy')  # Shape: (619, 748)
slope = np.load('SLPD2020.npy')       # Shape: (619, 748), values in degrees x 10

# Load time-series fire data
fire = np.load('fire.npy')            # Shape: (n_days, 619, 748)
timestamps = np.load('fire_timestamps.npy')  # Date strings

# Load weather data
temperature = np.load('temperature_2m.npy')  # Shape: (n_days, 619, 748), in Kelvin
wind_u = np.load('u_component_of_wind_10m.npy')  # Eastward wind component
wind_v = np.load('v_component_of_wind_10m.npy')  # Northward wind component

# Calculate wind speed and direction
wind_speed = np.sqrt(wind_u**2 + wind_v**2)
wind_direction = np.arctan2(wind_u, wind_v) * 180 / np.pi  # degrees from north
```

---

## Data Sources

- **LANDFIRE** (230 prefix): Landscape Fire and Resource Management Planning Tools
  - Source: https://landfire.gov/
  - Version: 2023 (LF 2023)
  
- **Topographic Data** (2020 suffix): Derived from LANDFIRE terrain products
  
- **Weather Data**: ERA5-Land reanalysis
  - Source: https://cds.climate.copernicus.eu/
  - Resolution: ~9km, interpolated to grid
  
- **Fire Perimeters**: NIFC (National Interagency Fire Center) daily fire perimeters

---

## Notes

1. **Scale factors**: Some LANDFIRE products use scale factors:
   - Heights (CH, EVH, FVH): values are in meters × 10
   - Slope degrees (SLPD): values are in degrees × 10
   - Aspect (ASP): values are in degrees × 10

2. **NoData values**: LANDFIRE uses specific NoData values (typically -9999 or similar)

3. **Coordinate System**: Data is in Albers Equal Area Conic projection (LANDFIRE standard)
"""

    # Write description file named after the dataset
    output_filename = f"{data_name}.md"
    with open(out_path / output_filename, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"\nGenerated {output_filename} in {out_dir}")


if __name__ == "__main__":
    # Configuration
    hdf5_path = "data/hdf5/dataset.hdf5"
    data_names = [
        "Bear_2020",
        "Brattain_2020",
        "Buck_2017",
        "Chimney_2016",
        "Ferguson_2018",
        "Pier_2017",
    ]
    for data_name in data_names:
        out_dir = Path(f"data/data_real/{data_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting data from: {hdf5_path}")
        print(f"Saving to: {out_dir}\n")

        # Extract data and get metadata
        metadata = extract_data(hdf5_path, str(out_dir), data_name)

        # Generate README
        generate_readme(metadata, str(out_dir), data_name)

    print("\nExtraction complete!")
