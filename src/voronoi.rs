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
pub fn get_cell_elevation(seed: u64, cell: (i32, i32), noise_scale: f32, island_radius: f32, ocean_ratio: f32, shape_roundness: f32, jitter: f32, noise_octaves: u32) -> f32 {
    let center = get_cell_point(seed, cell.0, cell.1, jitter);

    // Sample FBM noise at WORLD coordinates for smoke-like detail
    // noise_scale controls feature size: higher = smaller features
    // noise_octaves adds fine detail layers (more octaves = more wispy)
    let mut noise = FastNoiseLite::with_seed(seed as i32);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_frequency(Some(noise_scale * 0.01)); // Base frequency

    let noise_val = fbm_noise(&noise, center.0, center.1, noise_octaves);
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
pub fn is_cell_land(seed: u64, cell: (i32, i32), noise_scale: f32, water_threshold: f32, island_radius: f32, ocean_ratio: f32, shape_roundness: f32, jitter: f32, noise_octaves: u32) -> bool {
    get_cell_elevation(seed, cell, noise_scale, island_radius, ocean_ratio, shape_roundness, jitter, noise_octaves) > water_threshold
}

/// Cell terrain type
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellType {
    Ocean,      // Deep water far from land
    Coast,      // Land cell adjacent to water
    Inland,     // Land cell surrounded by land
}

/// Parameters for terrain generation (to reduce function arguments)
#[derive(Clone, Copy)]
pub struct TerrainParams {
    pub seed: u64,
    pub noise_scale: f32,
    pub water_threshold: f32,
    pub island_radius: f32,
    pub ocean_ratio: f32,
    pub shape_roundness: f32,
    pub jitter: f32,
    pub noise_octaves: u32,
}

impl TerrainParams {
    fn is_water(&self, cell: (i32, i32)) -> bool {
        get_cell_elevation(self.seed, cell, self.noise_scale, self.island_radius, self.ocean_ratio, self.shape_roundness, self.jitter, self.noise_octaves) <= self.water_threshold
    }
}

/// 8-directional neighbors
const NEIGHBORS: [(i32, i32); 8] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
];

/// Calculate distance from coast for a cell (BFS with limited depth)
/// Returns: 0 for coast, positive for inland (higher = further from coast)
/// max_depth limits search to prevent performance issues
pub fn get_coast_distance(cell: (i32, i32), params: &TerrainParams, max_depth: u32) -> u32 {
    // Water cells have no coast distance (they ARE water)
    if params.is_water(cell) {
        return 0;
    }

    // Check if this is a coast cell (has water neighbor)
    for (dx, dz) in NEIGHBORS {
        if params.is_water((cell.0 + dx, cell.1 + dz)) {
            return 0; // Coast cell
        }
    }

    // BFS to find nearest coast
    use std::collections::{HashSet, VecDeque};

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(cell);
    queue.push_back((cell, 1u32));

    while let Some((current, depth)) = queue.pop_front() {
        if depth > max_depth {
            return max_depth; // Cap at max depth
        }

        for (dx, dz) in NEIGHBORS {
            let neighbor = (current.0 + dx, current.1 + dz);

            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);

            // Found water = this neighbor is coast distance away
            if params.is_water(neighbor) {
                return depth;
            }

            // Check if neighbor is coast (has water neighbor)
            for (dx2, dz2) in NEIGHBORS {
                if params.is_water((neighbor.0 + dx2, neighbor.1 + dz2)) {
                    return depth; // Found coast at this depth
                }
            }

            queue.push_back((neighbor, depth + 1));
        }
    }

    max_depth // Fallback
}

/// Pre-compute coast distances for all land cells using multi-source BFS
/// Much faster than calling get_coast_distance() per cell
/// Returns HashMap with cell -> distance (0 = coast, higher = inland)
pub fn generate_coast_distances(params: &TerrainParams, max_depth: u32) -> std::collections::HashMap<(i32, i32), u32> {
    use std::collections::{HashMap, VecDeque};

    let mut distances: HashMap<(i32, i32), u32> = HashMap::new();
    let mut queue: VecDeque<((i32, i32), u32)> = VecDeque::new();

    // Scan area based on island_radius
    let scan_radius = (params.island_radius / CELL_SIZE) as i32 + 5;

    // First pass: find all coast cells (land cells adjacent to water)
    for x in -scan_radius..=scan_radius {
        for z in -scan_radius..=scan_radius {
            let cell = (x, z);
            if params.is_water(cell) {
                continue; // Skip water cells
            }

            // Check if this land cell is adjacent to water (coast)
            let mut is_coast = false;
            for (dx, dz) in NEIGHBORS {
                if params.is_water((cell.0 + dx, cell.1 + dz)) {
                    is_coast = true;
                    break;
                }
            }

            if is_coast {
                distances.insert(cell, 0);
                queue.push_back((cell, 0));
            }
        }
    }

    // Multi-source BFS from all coast cells
    while let Some((current, dist)) = queue.pop_front() {
        if dist >= max_depth {
            continue;
        }

        for (dx, dz) in NEIGHBORS {
            let neighbor = (current.0 + dx, current.1 + dz);

            // Skip if already visited or is water
            if distances.contains_key(&neighbor) || params.is_water(neighbor) {
                continue;
            }

            let new_dist = dist + 1;
            distances.insert(neighbor, new_dist);
            queue.push_back((neighbor, new_dist));
        }
    }

    distances
}

/// Get downslope direction - neighbor with lowest elevation (closest to coast)
/// Returns the cell coordinates of the lowest neighbor, or None if this is coast/water
pub fn get_downslope(cell: (i32, i32), params: &TerrainParams, max_depth: u32) -> Option<(i32, i32)> {
    let my_dist = get_coast_distance(cell, params, max_depth);

    // Coast or water - no downslope
    if my_dist == 0 {
        return None;
    }

    let mut lowest_neighbor = None;
    let mut lowest_dist = my_dist;

    for (dx, dz) in NEIGHBORS {
        let neighbor = (cell.0 + dx, cell.1 + dz);
        let neighbor_dist = get_coast_distance(neighbor, params, max_depth);

        if neighbor_dist < lowest_dist {
            lowest_dist = neighbor_dist;
            lowest_neighbor = Some(neighbor);
        }
    }

    lowest_neighbor
}

/// Generate all river cells once and return as a HashSet for O(1) lookup
/// Call this once per chunk generation, not per cell
pub fn generate_river_cells(params: &TerrainParams, max_depth: u32) -> std::collections::HashSet<(i32, i32)> {
    use std::collections::HashSet;

    let mut river_cells = HashSet::new();
    let mut rng = SmallRng::seed_from_u64(params.seed.wrapping_mul(0x9E3779B97F4A7C15));

    // Number of river sources
    let num_rivers = 12;

    for _ in 0..num_rivers {
        // Generate a random starting point in a wider area
        let start_x: i32 = (rng.gen::<f32>() * 60.0 - 30.0) as i32;
        let start_z: i32 = (rng.gen::<f32>() * 60.0 - 30.0) as i32;

        // Find nearest mountain cell (high coast distance)
        let mut best_start = (start_x, start_z);
        let mut best_dist = 0u32;

        for dx in -2..=2 {
            for dz in -2..=2 {
                let check = (start_x + dx, start_z + dz);
                if !params.is_water(check) {
                    let dist = get_coast_distance(check, params, max_depth);
                    if dist > best_dist {
                        best_dist = dist;
                        best_start = check;
                    }
                }
            }
        }

        // Only start rivers from high elevation
        if best_dist < max_depth / 2 {
            continue;
        }

        // Trace river path downhill and collect all cells
        let mut current = best_start;
        let mut steps = 0;

        while steps < 50 {
            river_cells.insert(current);

            match get_downslope(current, params, max_depth) {
                Some(next) => {
                    current = next;
                    steps += 1;
                }
                None => break, // Reached coast/water
            }
        }
    }

    river_cells
}

/// Check if a land cell is on the coastline (has water neighbor)
/// Returns CellType based on cell's position relative to water
pub fn get_cell_type(seed: u64, cell: (i32, i32), noise_scale: f32, water_threshold: f32, island_radius: f32, ocean_ratio: f32, shape_roundness: f32, jitter: f32, noise_octaves: u32) -> CellType {
    let elevation = get_cell_elevation(seed, cell, noise_scale, island_radius, ocean_ratio, shape_roundness, jitter, noise_octaves);

    // Water cell
    if elevation <= water_threshold {
        return CellType::Ocean;
    }

    // Land cell - check if any neighbor is water (8-directional)
    for (dx, dz) in NEIGHBORS {
        let neighbor = (cell.0 + dx, cell.1 + dz);
        let neighbor_elevation = get_cell_elevation(seed, neighbor, noise_scale, island_radius, ocean_ratio, shape_roundness, jitter, noise_octaves);
        if neighbor_elevation <= water_threshold {
            return CellType::Coast;
        }
    }

    CellType::Inland
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn test_params() -> TerrainParams {
        TerrainParams {
            seed: 12345,
            noise_scale: 0.3,
            water_threshold: 0.2,
            island_radius: 150.0,
            ocean_ratio: 0.5,
            shape_roundness: 0.4,
            jitter: 0.4,
            noise_octaves: 4,
        }
    }

    #[test]
    fn bench_generate_river_cells() {
        let params = test_params();
        let max_depth = 10;

        let start = Instant::now();
        let rivers = generate_river_cells(&params, max_depth);
        let elapsed = start.elapsed();

        println!("generate_river_cells: {:?} ({} cells)", elapsed, rivers.len());
    }

    #[test]
    fn bench_get_coast_distance() {
        let params = test_params();
        let max_depth = 10;

        // Test 100 cells
        let start = Instant::now();
        for x in -5..5 {
            for z in -5..5 {
                get_coast_distance((x, z), &params, max_depth);
            }
        }
        let elapsed = start.elapsed();

        println!("get_coast_distance (100 cells): {:?}", elapsed);
    }

    #[test]
    fn bench_get_cell_elevation() {
        let params = test_params();

        // Test 256 cells (one chunk worth)
        let start = Instant::now();
        for x in 0..16 {
            for z in 0..16 {
                get_cell_elevation(params.seed, (x, z), params.noise_scale, params.island_radius, params.ocean_ratio, params.shape_roundness, params.jitter, params.noise_octaves);
            }
        }
        let elapsed = start.elapsed();

        println!("get_cell_elevation (256 cells): {:?}", elapsed);
    }

    #[test]
    fn bench_find_nearest_cells() {
        let seed = 12345u64;
        let jitter = 0.4;

        // Test 256 positions (one chunk)
        let start = Instant::now();
        for x in 0..16 {
            for z in 0..16 {
                find_nearest_cells(seed, x as f32, z as f32, jitter);
            }
        }
        let elapsed = start.elapsed();

        println!("find_nearest_cells (256 positions): {:?}", elapsed);
    }

    #[test]
    fn bench_full_chunk_simulation() {
        let params = test_params();
        let max_depth = 10;

        println!("\n=== Full chunk simulation (OLD - per-cell BFS) ===");

        // 1. Generate rivers (once per chunk)
        let start = Instant::now();
        let river_cells = generate_river_cells(&params, max_depth);
        let river_time = start.elapsed();
        println!("1. generate_river_cells: {:?} ({} cells)", river_time, river_cells.len());

        // 2. Process all 256 cells in chunk (OLD way - BFS per cell)
        let start = Instant::now();
        let mut coast_dist_time = std::time::Duration::ZERO;

        for x in 0..16 {
            for z in 0..16 {
                let voronoi = find_nearest_cells(params.seed, x as f32, z as f32, params.jitter);
                let cell_type = get_cell_type(params.seed, voronoi.nearest_cell, params.noise_scale, params.water_threshold, params.island_radius, params.ocean_ratio, params.shape_roundness, params.jitter, params.noise_octaves);

                let t = Instant::now();
                if matches!(cell_type, CellType::Inland) {
                    get_coast_distance(voronoi.nearest_cell, &params, max_depth);
                }
                coast_dist_time += t.elapsed();
            }
        }
        let total_old = start.elapsed();

        println!("2. OLD per-cell BFS coast_distance: {:?}", coast_dist_time);
        println!("   OLD total loop: {:?}", total_old);
    }

    #[test]
    fn bench_full_chunk_simulation_cached() {
        let params = test_params();
        let max_depth = 10;

        println!("\n=== Full chunk simulation (NEW - cached) ===");

        // 1. Pre-compute ALL data once
        let start = Instant::now();
        let river_cells = generate_river_cells(&params, max_depth);
        let river_time = start.elapsed();

        let start = Instant::now();
        let coast_distances = generate_coast_distances(&params, max_depth);
        let coast_cache_time = start.elapsed();

        println!("1. Pre-compute:");
        println!("   - generate_river_cells: {:?} ({} cells)", river_time, river_cells.len());
        println!("   - generate_coast_distances: {:?} ({} cells)", coast_cache_time, coast_distances.len());

        // 2. Process all 256 cells with O(1) lookups
        let start = Instant::now();
        for x in 0..16 {
            for z in 0..16 {
                let voronoi = find_nearest_cells(params.seed, x as f32, z as f32, params.jitter);
                let cell_type = get_cell_type(params.seed, voronoi.nearest_cell, params.noise_scale, params.water_threshold, params.island_radius, params.ocean_ratio, params.shape_roundness, params.jitter, params.noise_octaves);

                if matches!(cell_type, CellType::Inland) {
                    let _dist = coast_distances.get(&voronoi.nearest_cell).copied().unwrap_or(max_depth);
                }
                let _is_river = river_cells.contains(&voronoi.nearest_cell);
            }
        }
        let total_loop = start.elapsed();

        println!("2. Per-cell loop (O(1) lookups): {:?}", total_loop);
        println!("\n   NEW TOTAL: {:?}", river_time + coast_cache_time + total_loop);
    }

    #[test]
    fn bench_get_cell_elevation_detailed() {
        let params = test_params();

        // Test FBM noise cost
        let start = Instant::now();
        let mut noise = FastNoiseLite::with_seed(params.seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(params.noise_scale * 0.01));
        let noise_setup_time = start.elapsed();

        let start = Instant::now();
        for x in 0..256 {
            let _ = fbm_noise(&noise, x as f32, 0.0, params.noise_octaves);
        }
        let fbm_time = start.elapsed();

        println!("\n=== FBM Noise breakdown ===");
        println!("Noise setup: {:?}", noise_setup_time);
        println!("FBM 256 calls: {:?}", fbm_time);
        println!("Per FBM call: {:?}", fbm_time / 256);
    }
}
