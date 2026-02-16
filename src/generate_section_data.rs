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
    find_corners_in_region, world_to_cell, assign_corner_elevations, interpolate_elevation,
    TerrainParams,
};

// ============================================================================
// PHASE 6: 3D VOXELIZATION
// ============================================================================

/// Convert elevation (0.0-1.0) to surface Y coordinate
/// Ocean floor below sea_level, land rises above
/// water_threshold defines the land/water boundary in elevation space
fn elevation_to_surface_y(elevation: f32, water_threshold: f32, sea_level: u16, max_terrain_height: u16) -> u16 {
    const OCEAN_FLOOR_DEPTH: u16 = 15; // Ocean floor depth below sea_level

    if elevation < water_threshold {
        // Underwater: map 0.0-water_threshold to (sea_level - depth) to sea_level
        let t = elevation / water_threshold;
        let min_y = sea_level.saturating_sub(OCEAN_FLOOR_DEPTH);
        min_y + ((sea_level - min_y) as f32 * t) as u16
    } else {
        // Land: map water_threshold-1.0 to sea_level to max_terrain_height
        let t = (elevation - water_threshold) / (1.0 - water_threshold);
        let height_range = max_terrain_height - sea_level;
        sea_level + (height_range as f32 * t) as u16
    }
}

/// Get surface block based on elevation/biome
/// water_threshold defines the land/water boundary
fn get_surface_block(elevation: f32, water_threshold: f32) -> BlockID {
    if elevation < water_threshold {
        // Underwater - ocean floor
        if elevation < water_threshold * 0.5 {
            BlockID::Gravel // Deep ocean floor
        } else {
            BlockID::Sand // Shallow ocean floor
        }
    } else {
        // Above water - land
        let land_elevation = elevation - water_threshold;
        if land_elevation < 0.08 {
            BlockID::Sand // Beach (just above water line)
        } else if land_elevation < 0.20 {
            BlockID::Grass // Plains
        } else if land_elevation < 0.40 {
            BlockID::Podzol // Forest
        } else if land_elevation < 0.55 {
            BlockID::Stone // Hills (exposed rock)
        } else if land_elevation < 0.70 {
            BlockID::Gravel // Mountains
        } else {
            BlockID::SmoothStone // Peaks
        }
    }
}

/// Get subsurface block (layer below surface)
fn get_subsurface_block(elevation: f32, water_threshold: f32) -> BlockID {
    if elevation < water_threshold {
        // Underwater
        BlockID::Sand // Ocean floor has sand below
    } else {
        // Land
        let land_elevation = elevation - water_threshold;
        if land_elevation < 0.08 {
            BlockID::Sand // Beach has sand below
        } else if land_elevation < 0.40 {
            BlockID::CoarseDirt // Plains/forest have dirt
        } else {
            BlockID::Stone // Hills/mountains have stone
        }
    }
}

/// Phase 6: 3D Voxelization - Generate terrain with actual height
pub fn generate_section_data(
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    macro_data: &MacroData,
    settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();

    let chunk_x = chunk_position.x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_position.z as f32 * CHUNK_SIZE as f32;
    let section_y_start = vertical_index * CHUNK_SIZE as usize;

    // Terrain params
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

            // Interpolate elevation from corners
            let elevation = interpolate_elevation(world_point, &corners);

            // Convert elevation to Y coordinate
            let surface_y = elevation_to_surface_y(
                elevation,
                settings.water_threshold,
                settings.sea_level,
                settings.max_terrain_height,
            ) as usize;

            // Get blocks for this column
            // Coastline is defined by elevation isoline at water_threshold
            let surface_block = get_surface_block(elevation, settings.water_threshold);
            let subsurface_block = get_subsurface_block(elevation, settings.water_threshold);
            let is_underwater = elevation < settings.water_threshold;

            for y in 0_u8..(CHUNK_SIZE as u8) {
                let y_global = section_y_start + y as usize;
                let pos = ChunkBlockPosition::new(x, y, z);

                let sea_level = settings.sea_level as usize;

                if y_global == 0 {
                    // Bedrock at bottom
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Bedrock.id()));
                } else if y_global < surface_y.saturating_sub(3) {
                    // Deep underground: stone
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Stone.id()));
                } else if y_global < surface_y {
                    // Subsurface layers (3 blocks below surface)
                    section_data.insert(&pos, BlockDataInfo::create(subsurface_block.id()));
                } else if y_global == surface_y {
                    // Surface block
                    section_data.insert(&pos, BlockDataInfo::create(surface_block.id()));
                } else if y_global > surface_y && y_global <= sea_level {
                    // Water: fill from surface+1 to sea_level (ocean areas)
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Water.id()));
                }
                // Above sea_level: air (no block)
            }
        }
    }

    section_data
}
