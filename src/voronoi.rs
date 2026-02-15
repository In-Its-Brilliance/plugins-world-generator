use common::CHUNK_SIZE;
use fastnoise_lite::{FastNoiseLite, NoiseType};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const CELL_SIZE: f32 = CHUNK_SIZE as f32;

/// Compute jittered point position for a grid cell (deterministic from seed)
pub fn get_cell_point(seed: u64, cell_x: i32, cell_z: i32, jitter: f32) -> (f32, f32) {
    let x_bits = cell_x as u32 as u64;
    let z_bits = cell_z as u32 as u64;
    let cell_seed = seed
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(x_bits)
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(z_bits)
        .wrapping_mul(0x517cc1b727220a95);

    let mut rng = SmallRng::seed_from_u64(cell_seed);

    let center_x = (cell_x as f32 + 0.5) * CELL_SIZE;
    let center_z = (cell_z as f32 + 0.5) * CELL_SIZE;

    let offset_x = (rng.gen::<f32>() - 0.5) * 2.0 * jitter * CELL_SIZE;
    let offset_z = (rng.gen::<f32>() - 0.5) * 2.0 * jitter * CELL_SIZE;

    (center_x + offset_x, center_z + offset_z)
}

/// Get cell coordinates from world position
#[inline]
pub fn world_to_cell(x: f32, z: f32) -> (i32, i32) {
    ((x / CELL_SIZE).floor() as i32, (z / CELL_SIZE).floor() as i32)
}

/// Result of finding nearest Voronoi cells
pub struct VoronoiResult {
    /// Nearest cell center
    pub nearest: (f32, f32),
    /// Cell coordinates of nearest
    pub nearest_cell: (i32, i32),
    /// Distance squared to nearest
    pub dist_sq: f32,
    /// Distance squared to second nearest (for edge detection)
    pub second_dist_sq: f32,
}

/// Find nearest cell centers for a world position
/// Checks 3x3 grid of neighboring cells
pub fn find_nearest_cells(seed: u64, world_x: f32, world_z: f32, jitter: f32) -> VoronoiResult {
    let (cell_x, cell_z) = world_to_cell(world_x, world_z);

    let mut nearest = (0.0_f32, 0.0_f32);
    let mut nearest_cell = (0_i32, 0_i32);
    let mut min_dist_sq = f32::MAX;
    let mut second_min_dist_sq = f32::MAX;

    // Check 3x3 neighborhood
    for dx in -1..=1 {
        for dz in -1..=1 {
            let cx = cell_x + dx;
            let cz = cell_z + dz;
            let point = get_cell_point(seed, cx, cz, jitter);

            let dist_sq = (world_x - point.0).powi(2) + (world_z - point.1).powi(2);

            if dist_sq < min_dist_sq {
                second_min_dist_sq = min_dist_sq;
                min_dist_sq = dist_sq;
                nearest = point;
                nearest_cell = (cx, cz);
            } else if dist_sq < second_min_dist_sq {
                second_min_dist_sq = dist_sq;
            }
        }
    }

    VoronoiResult {
        nearest,
        nearest_cell,
        dist_sq: min_dist_sq,
        second_dist_sq: second_min_dist_sq,
    }
}

/// Check if position is on a Voronoi edge (equidistant to two cells)
#[inline]
pub fn is_on_voronoi_edge(result: &VoronoiResult, threshold: f32) -> bool {
    let dist1 = result.dist_sq.sqrt();
    let dist2 = result.second_dist_sq.sqrt();
    (dist2 - dist1).abs() < threshold
}

/// Check if position is at a cell center
#[inline]
pub fn is_at_cell_center(world_x: f32, world_z: f32, center: (f32, f32)) -> bool {
    (center.0.floor() as i32) == (world_x.floor() as i32)
        && (center.1.floor() as i32) == (world_z.floor() as i32)
}

/// Fractal Brownian Motion noise for complex coastlines
fn fbm_noise(noise: &FastNoiseLite, x: f32, z: f32, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_value = 0.0;

    for _ in 0..octaves {
        value += noise.get_noise_2d(x * frequency, z * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    value / max_value
}

/// Get cell elevation (0.0 = deep water, 1.0 = high mountains)
/// shape_roundness controls: 0 = pure fractal, 1 = circular
pub fn get_cell_elevation(seed: u64, cell: (i32, i32), noise_scale: f32, island_radius: f32, ocean_ratio: f32, shape_roundness: f32, jitter: f32) -> f32 {
    let center = get_cell_point(seed, cell.0, cell.1, jitter);

    // Sample noise at WORLD coordinates
    // noise_scale controls feature size: higher = smaller features
    let mut noise = FastNoiseLite::with_seed(seed as i32);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_frequency(Some(noise_scale * 0.01)); // Direct frequency control
    let noise_val = noise.get_noise_2d(center.0, center.1);
    let noise_normalized = (noise_val + 1.0) / 2.0; // 0 to 1

    // Distance from origin, normalized to island_radius
    let dist = (center.0 * center.0 + center.1 * center.1).sqrt() / island_radius;

    // shape_roundness controls how much distance affects the shape:
    // 0.0 = weak distance influence (irregular coastline)
    // 1.0 = strong distance influence (more circular)
    // Always keep minimum 0.2 to ensure island shape exists
    let dist_strength = 0.2 + shape_roundness * 0.8;
    let threshold = ocean_ratio * (1.0 + dist * dist * dist_strength);

    // Land if noise > threshold, water otherwise
    if noise_normalized > threshold {
        // Land: scale to 0.5-1.0 range
        let excess = (noise_normalized - threshold) / (1.0 - threshold).max(0.01);
        0.5 + excess * 0.5
    } else {
        // Water: scale to 0.0-0.5 range
        let deficit = (threshold - noise_normalized) / threshold.max(0.01);
        (0.5 - deficit * 0.5).max(0.0)
    }
}

/// Determine if a cell is land based on elevation threshold
pub fn is_cell_land(seed: u64, cell: (i32, i32), noise_scale: f32, water_threshold: f32, island_radius: f32, ocean_ratio: f32, shape_roundness: f32, jitter: f32) -> bool {
    get_cell_elevation(seed, cell, noise_scale, island_radius, ocean_ratio, shape_roundness, jitter) > water_threshold
}
