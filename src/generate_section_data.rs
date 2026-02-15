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
use crate::voronoi::{find_nearest_cells, is_at_cell_center, is_cell_land, is_on_voronoi_edge};

const GROUND_LEVEL: usize = 60;
const EDGE_THRESHOLD: f32 = 0.7;

/// Runtime Voronoi visualization with Land/Water
pub fn generate_section_data(
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    macro_data: &MacroData,
    settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();

    let chunk_x = chunk_position.x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_position.z as f32 * CHUNK_SIZE as f32;
    let world_radius = settings.world_size / 2.0;

    for x in 0_u8..(CHUNK_SIZE as u8) {
        for z in 0_u8..(CHUNK_SIZE as u8) {
            let world_x = chunk_x + x as f32;
            let world_z = chunk_z + z as f32;

            // Runtime Voronoi computation
            let voronoi = find_nearest_cells(macro_data.seed, world_x + 0.5, world_z + 0.5);
            let is_edge = is_on_voronoi_edge(&voronoi, EDGE_THRESHOLD);
            let is_center = is_at_cell_center(world_x, world_z, voronoi.nearest);
            let is_land = is_cell_land(macro_data.seed, voronoi.nearest_cell, world_radius);

            for y in 0_u8..(CHUNK_SIZE as u8) {
                let y_global = y as usize + (vertical_index * CHUNK_SIZE as usize);
                let pos = ChunkBlockPosition::new(x, y, z);

                if y_global < GROUND_LEVEL {
                    if is_land {
                        section_data.insert(&pos, BlockDataInfo::create(BlockID::Grass.id()));
                    } else {
                        section_data.insert(&pos, BlockDataInfo::create(BlockID::Water.id()));
                    }
                } else if y_global == GROUND_LEVEL {
                    if is_center {
                        // Cell centers - different color for land/water
                        if is_land {
                            section_data.insert(&pos, BlockDataInfo::create(BlockID::Stone.id()));
                        } else {
                            section_data.insert(&pos, BlockDataInfo::create(BlockID::Gravel.id()));
                        }
                    } else if is_edge {
                        // Voronoi edges
                        section_data.insert(&pos, BlockDataInfo::create(BlockID::Sand.id()));
                    }
                }
            }
        }
    }

    section_data
}
