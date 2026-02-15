use common::{chunks::chunk_data::WorldMacroData, CHUNK_SIZE};
use delaunator::{triangulate, Point};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::settings::GeneratorSettings;

const JITTER: f32 = 0.4;
const CELL_SIZE: f32 = CHUNK_SIZE as f32;

/// Stores points and Delaunay triangulation
#[derive(serde::Serialize, serde::Deserialize)]
pub struct MacroData {
    pub seed: u64,
    pub points: Vec<(f32, f32)>,
    /// Triangle indices (3 per triangle, referencing points)
    pub triangles: Vec<usize>,
}

/// Compute jittered point position for a grid cell
fn get_cell_point(seed: u64, cell_x: i32, cell_z: i32) -> (f32, f32) {
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

/// Generate world macro data with Delaunay triangulation
pub fn generate_world_macro(seed: u64, settings: &GeneratorSettings) -> WorldMacroData {
    // Generate all points for world area centered at 0,0
    let half_cells = (settings.world_size / CELL_SIZE / 2.0).ceil() as i32;
    let mut points: Vec<(f32, f32)> = Vec::new();

    for cell_x in -half_cells..half_cells {
        for cell_z in -half_cells..half_cells {
            let (x, z) = get_cell_point(seed, cell_x, cell_z);
            points.push((x, z));
        }
    }

    extism_pdk::log!(
        extism_pdk::LogLevel::Info,
        "Generated {} points ({}x{} cells centered at 0,0)",
        points.len(),
        half_cells * 2,
        half_cells * 2
    );

    // Run Delaunay triangulation
    let delaunay_points: Vec<Point> = points
        .iter()
        .map(|(x, z)| Point { x: *x as f64, y: *z as f64 })
        .collect();

    let triangles = triangulate(&delaunay_points)
        .map(|t| t.triangles)
        .unwrap_or_default();

    extism_pdk::log!(
        extism_pdk::LogLevel::Info,
        "Delaunay: {} triangles ({} indices)",
        triangles.len() / 3,
        triangles.len()
    );

    let macro_data = MacroData {
        seed,
        points,
        triangles,
    };
    let data = serde_yaml::to_value(&macro_data).unwrap();
    WorldMacroData::create(data)
}
