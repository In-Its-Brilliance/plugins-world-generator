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
use crate::voronoi::{find_nearest_cells, get_cell_elevation, get_cell_type, is_on_voronoi_edge, CellType};

/// Map elevation (0-1) to block type for gradient visualization
/// Dark (low) -> Light (high)
fn elevation_to_block(elevation: f32) -> BlockID {
    if elevation < 0.1 {
        BlockID::Bedrock // Darkest - deep water
    } else if elevation < 0.2 {
        BlockID::Blackstone
    } else if elevation < 0.3 {
        BlockID::Deepslate // Water threshold
    } else if elevation < 0.4 {
        BlockID::Andesite
    } else if elevation < 0.5 {
        BlockID::Cobblestone
    } else if elevation < 0.6 {
        BlockID::Stone
    } else if elevation < 0.7 {
        BlockID::Gravel
    } else if elevation < 0.8 {
        BlockID::Sand
    } else if elevation < 0.9 {
        BlockID::Sandstone
    } else {
        BlockID::SmoothStone // Brightest - mountain peaks
    }
}

/// Runtime Voronoi visualization with elevation gradient
pub fn generate_section_data(
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    macro_data: &MacroData,
    settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();

    let chunk_x = chunk_position.x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_position.z as f32 * CHUNK_SIZE as f32;

    for x in 0_u8..(CHUNK_SIZE as u8) {
        for z in 0_u8..(CHUNK_SIZE as u8) {
            let world_x = chunk_x + x as f32;
            let world_z = chunk_z + z as f32;

            // Runtime Voronoi computation
            let voronoi = find_nearest_cells(macro_data.seed, world_x + 0.5, world_z + 0.5, settings.jitter);
            let is_edge = is_on_voronoi_edge(&voronoi, settings.edge_threshold);
            let elevation = get_cell_elevation(macro_data.seed, voronoi.nearest_cell, settings.elevation_noise_scale, settings.island_radius, settings.ocean_ratio, settings.shape_roundness, settings.jitter, settings.noise_octaves);
            let cell_type = get_cell_type(macro_data.seed, voronoi.nearest_cell, settings.elevation_noise_scale, settings.water_threshold, settings.island_radius, settings.ocean_ratio, settings.shape_roundness, settings.jitter, settings.noise_octaves);

            // Get block based on cell type and elevation
            let block = match cell_type {
                CellType::Ocean => elevation_to_block(elevation),
                CellType::Coast => BlockID::AmethystBlock,  // Bright coastline for visibility
                CellType::Inland => elevation_to_block(elevation),
            };

            for y in 0_u8..(CHUNK_SIZE as u8) {
                let y_global = y as usize + (vertical_index * CHUNK_SIZE as usize);
                let pos = ChunkBlockPosition::new(x, y, z);

                if y_global < settings.sea_level as usize {
                    section_data.insert(&pos, BlockDataInfo::create(block.id()));
                } else if y_global == settings.sea_level as usize && is_edge {
                    // Voronoi edges for visualization
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::OakPlanks.id()));
                }
            }
        }
    }

    section_data
}
