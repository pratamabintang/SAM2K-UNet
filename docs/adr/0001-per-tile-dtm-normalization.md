# Per-Tile Min-Max Normalization for DTM

In landslide hazard segmentation, Digital Terrain Model (DTM) rasters have varying absolute elevation baselines across disparate geographic regions in train and test splits. We decided to apply per-tile min-max normalization rather than dataset-wide Z-score or global min-max, ensuring local slope and elevation gradients are preserved without being distorted by regional baseline altitude shifts.
