use common::CHUNK_SIZE;
use fastnoise_lite::{FastNoiseLite, NoiseType};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const JITTER: f32 = 0.4;
pub const CELL_SIZE: f32 = CHUNK_SIZE as f32;

/// Island shape parameters
const NOISE_SCALE: f32 = 0.003;
const LAND_THRESHOLD: f32 = 0.1;

/// Compute jittered point position for a grid cell (deterministic from seed)
pub fn get_cell_point(seed: u64, cell_x: i32, cell_z: i32) -> (f32, f32) {
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

    let offset_x = (rng.gen::<f32>() - 0.5) * 2.0 * JITTER * CELL_SIZE;
    let offset_z = (rng.gen::<f32>() - 0.5) * 2.0 * JITTER * CELL_SIZE;

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
pub fn find_nearest_cells(seed: u64, world_x: f32, world_z: f32) -> VoronoiResult {
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
            let point = get_cell_point(seed, cx, cz);

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

/// Determine if a cell is land based on noise + radial gradient
/// Returns true for land, false for water
pub fn is_cell_land(seed: u64, cell: (i32, i32), world_radius: f32) -> bool {
    let center = get_cell_point(seed, cell.0, cell.1);

    // Distance from world center (0, 0)
    let dist = (center.0 * center.0 + center.1 * center.1).sqrt();
    let normalized_dist = (dist / world_radius).min(1.0);

    // Radial gradient: 1.0 at center, 0.0 at edge
    let radial = 1.0 - normalized_dist;

    // Noise for irregular coastline
    let mut noise = FastNoiseLite::with_seed(seed as i32);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    let noise_val = noise.get_noise_2d(center.0 * NOISE_SCALE, center.1 * NOISE_SCALE);

    // Combine: radial gradient + noise
    // noise_val is -1..1, normalize to 0..1
    let noise_normalized = (noise_val + 1.0) / 2.0;

    // Land if combined value exceeds threshold
    let combined = radial * 0.6 + noise_normalized * 0.4;
    combined > LAND_THRESHOLD
}
