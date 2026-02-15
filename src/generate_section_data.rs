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
use crate::voronoi::{
    find_nearest_cells, is_on_voronoi_edge,
    find_corners_in_region, world_to_cell, assign_corner_elevations, interpolate_elevation,
    TerrainParams,
};

/// Map interpolated elevation to block type for visualization
/// Ocean (0.0) -> Coast (0.08) -> Inland (0.2-1.0)
fn interpolated_elevation_to_block(elevation: f32) -> BlockID {
    if elevation < 0.03 {
        BlockID::Deepslate // Ocean only (very close to 0.0)
    } else if elevation < 0.12 {
        BlockID::Sand // Beach - only near coast corners (0.08)
    } else if elevation < 0.30 {
        BlockID::Grass // Plains
    } else if elevation < 0.50 {
        BlockID::Podzol // Forest
    } else if elevation < 0.70 {
        BlockID::Stone // Hills
    } else if elevation < 0.88 {
        BlockID::Gravel // Mountains
    } else {
        BlockID::SmoothStone // Peaks
    }
}

/// Runtime Voronoi visualization with corners
pub fn generate_section_data(
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    macro_data: &MacroData,
    settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();

    let chunk_x = chunk_position.x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_position.z as f32 * CHUNK_SIZE as f32;

    // Terrain params for coast distance calculation
    let terrain_params = TerrainParams {
        seed: macro_data.seed,
        noise_scale: settings.elevation_noise_scale,
        water_threshold: settings.water_threshold,
        island_radius: settings.island_radius,
        ocean_ratio: settings.ocean_ratio,
        shape_roundness: settings.shape_roundness,
        jitter: settings.jitter,
        noise_octaves: settings.noise_octaves,
    };

    const MAX_COAST_DISTANCE: u32 = 10;

    // Compute corners for this chunk area (+ buffer)
    let center_cell = world_to_cell(chunk_x + CHUNK_SIZE as f32 / 2.0, chunk_z + CHUNK_SIZE as f32 / 2.0);
    let mut corners = find_corners_in_region(
        macro_data.seed,
        center_cell.0,
        center_cell.1,
        2, // radius in cells
        settings.jitter,
    );

    // Phase 2: Assign elevations to corners
    assign_corner_elevations(macro_data.seed, &mut corners, &terrain_params, MAX_COAST_DISTANCE);

    for x in 0_u8..(CHUNK_SIZE as u8) {
        for z in 0_u8..(CHUNK_SIZE as u8) {
            let world_x = chunk_x + x as f32;
            let world_z = chunk_z + z as f32;
            let world_point = (world_x + 0.5, world_z + 0.5);

            // Phase 3: Interpolate elevation from corners
            let elevation = interpolate_elevation(world_point, &corners);
            let block = interpolated_elevation_to_block(elevation);

            // Optional: still detect edges for visualization
            let voronoi = find_nearest_cells(macro_data.seed, world_point.0, world_point.1, settings.jitter);
            let is_edge = is_on_voronoi_edge(&voronoi, settings.edge_threshold);

            for y in 0_u8..(CHUNK_SIZE as u8) {
                let y_global = y as usize + (vertical_index * CHUNK_SIZE as usize);
                let pos = ChunkBlockPosition::new(x, y, z);

                if y_global < settings.sea_level as usize {
                    // Use interpolated elevation for terrain color
                    section_data.insert(&pos, BlockDataInfo::create(block.id()));
                } else if y_global == settings.sea_level as usize {
                    // Show edges on top layer
                    if is_edge {
                        section_data.insert(&pos, BlockDataInfo::create(BlockID::OakPlanks.id()));
                    }
                }
            }
        }
    }

    section_data
}
