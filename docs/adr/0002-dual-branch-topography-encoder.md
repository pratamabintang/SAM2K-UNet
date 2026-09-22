# Dual-Branch Architecture with ConvNeXt Topography Encoder

Optical RGB imagery and continuous topographic rasters (DTM, slope, aspect, hillshade) have fundamentally distinct physical distributions and spatial characteristics. We decided to separate them into two parallel encoder branches: SAM2 Hiera ViT for optical RGB, and a pretrained ConvNeXt CNN for topography, fusing their hierarchical multi-scale features via concatenation and 1x1 linear convolutions rather than RFB dilated convolutions.
