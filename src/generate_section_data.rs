use common::{
    chunks::{
        block_position::ChunkBlockPosition,
        chunk_data::{BlockDataInfo, ChunkSectionData},
        chunk_position::ChunkPosition,
    },
    default_blocks_ids::BlockID,
    CHUNK_SIZE,
};

use crate::generate_world_macro::MacroData;
use crate::settings::GeneratorSettings;

const GROUND_LEVEL: usize = 60;
const LINE_THRESHOLD: f32 = 0.7;

/// Check if point (px, pz) is on line segment from (x1, z1) to (x2, z2)
fn is_on_edge(px: f32, pz: f32, x1: f32, z1: f32, x2: f32, z2: f32) -> bool {
    // Bounding box check
    let min_x = x1.min(x2) - LINE_THRESHOLD;
    let max_x = x1.max(x2) + LINE_THRESHOLD;
    let min_z = z1.min(z2) - LINE_THRESHOLD;
    let max_z = z1.max(z2) + LINE_THRESHOLD;

    if px < min_x || px > max_x || pz < min_z || pz > max_z {
        return false;
    }

    // Distance from point to line segment
    let dx = x2 - x1;
    let dz = z2 - z1;
    let len_sq = dx * dx + dz * dz;

    if len_sq < 0.001 {
        return false;
    }

    // Project point onto line
    let t = ((px - x1) * dx + (pz - z1) * dz) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let proj_x = x1 + t * dx;
    let proj_z = z1 + t * dz;

    let dist_sq = (px - proj_x) * (px - proj_x) + (pz - proj_z) * (pz - proj_z);
    dist_sq < LINE_THRESHOLD * LINE_THRESHOLD
}

/// Check if position is on any triangle edge
fn is_on_triangle_edge(world_x: f32, world_z: f32, macro_data: &MacroData) -> bool {
    let px = world_x + 0.5;
    let pz = world_z + 0.5;

    for tri in macro_data.triangles.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (x1, z1) = macro_data.points[tri[0]];
        let (x2, z2) = macro_data.points[tri[1]];
        let (x3, z3) = macro_data.points[tri[2]];

        if is_on_edge(px, pz, x1, z1, x2, z2)
            || is_on_edge(px, pz, x2, z2, x3, z3)
            || is_on_edge(px, pz, x3, z3, x1, z1)
        {
            return true;
        }
    }
    false
}

/// Check if position is at a vertex point
fn is_point_at(world_x: f32, world_z: f32, macro_data: &MacroData) -> bool {
    for (px, pz) in &macro_data.points {
        if (px.floor() as i32) == (world_x as i32) && (pz.floor() as i32) == (world_z as i32) {
            return true;
        }
    }
    false
}

/// Step 3: Delaunay triangulation visualization
pub fn generate_section_data(
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    macro_data: &MacroData,
    _settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();

    let chunk_x = chunk_position.x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_position.z as f32 * CHUNK_SIZE as f32;

    for x in 0_u8..(CHUNK_SIZE as u8) {
        for z in 0_u8..(CHUNK_SIZE as u8) {
            let world_x = chunk_x + x as f32;
            let world_z = chunk_z + z as f32;

            let is_point = is_point_at(world_x, world_z, macro_data);
            let is_edge = is_on_triangle_edge(world_x, world_z, macro_data);

            for y in 0_u8..(CHUNK_SIZE as u8) {
                let y_global = y as usize + (vertical_index * CHUNK_SIZE as usize);
                let pos = ChunkBlockPosition::new(x, y, z);

                if y_global < GROUND_LEVEL {
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Grass.id()));
                } else if y_global == GROUND_LEVEL {
                    if is_point {
                        section_data.insert(&pos, BlockDataInfo::create(BlockID::Stone.id()));
                    } else if is_edge {
                        section_data.insert(&pos, BlockDataInfo::create(BlockID::Cobblestone.id()));
                    }
                }
            }
        }
    }

    section_data
}
