# CUDA Denoiser for Monte-Carlo Pathtraced Renders

Uses CUDA to implement SVGF (Spatiotemporal Variance-Guided Filtering) for denoising Monte-Carlo pathtraced renders.

## Features

- Real-time denoising of Monte-Carlo pathtraced images
- Temporal accumulation to reduce flickering
- Variance-guided filtering for high-quality results
- Fully implemented on CUDA for GPU acceleration

## Requirements

- CUDA Toolkit 11.0 or higher
- C++17 compatible compiler (gcc on linux)

## Usage

Build and run the test:

```bash
make clean
make test
```