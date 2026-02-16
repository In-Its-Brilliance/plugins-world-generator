use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct GeneratorSettings {
    pub sea_level: u16,
    /// Controls coastline detail frequency (higher = more jagged)
    pub elevation_noise_scale: f32,
    /// Land/water threshold - higher = more water
    pub water_threshold: f32,
    /// Island radius in blocks from center (0,0)
    pub island_radius: f32,
    /// Ocean ratio - base threshold (lower = more land, 0.2-0.4 typical)
    pub ocean_ratio: f32,
    /// Shape roundness: 0.0 = pure fractal noise, 1.0 = circular with noise
    pub shape_roundness: f32,
    /// Voronoi edge detection threshold
    pub edge_threshold: f32,
    /// Cell point jitter (0.0 = grid, 0.5 = max randomness)
    pub jitter: f32,
    /// Noise octaves for smoke-like detail (1 = blob, 4-6 = wispy smoke)
    pub noise_octaves: u32,
}
