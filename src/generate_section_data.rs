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
    TerrainParams, find_nearest_cells, is_on_voronoi_edge, get_cell_type, CellType,
    get_continuous_elevation,
};

// ============================================================================
// PHASE 6: 3D VOXELIZATION
// ============================================================================

/// Convert elevation (0.0-1.0) to surface Y coordinate
/// Ocean floor below sea_level, land rises above
/// Water boundary is at elevation 0.5 (matches get_continuous_elevation)
/// Mountain height is derived from island_radius (bigger island = higher mountains)
fn elevation_to_surface_y(elevation: f32, sea_level: u16) -> u16 {
    const OCEAN_FLOOR_DEPTH: u16 = 15; // Ocean floor depth below sea_level
    const MAX_MOUNTAIN_HEIGHT: u16 = 40; // Fixed max height above sea level
    const WATER_LEVEL: f32 = 0.5; // Water boundary in elevation space

    let max_terrain_height = sea_level + MAX_MOUNTAIN_HEIGHT;

    if elevation < WATER_LEVEL {
        // Underwater: map 0.0-0.5 to (sea_level - depth) to sea_level
        let t = elevation / WATER_LEVEL;
        let min_y = sea_level.saturating_sub(OCEAN_FLOOR_DEPTH);
        min_y + ((sea_level - min_y) as f32 * t) as u16
    } else {
        // Land: map 0.5-1.0 to sea_level to max_terrain_height
        let t = (elevation - WATER_LEVEL) / (1.0 - WATER_LEVEL);
        let height_range = max_terrain_height - sea_level;
        sea_level + (height_range as f32 * t) as u16
    }
}

/// Get surface block based on slope and proximity to water
/// slope: 0.0 = flat, higher = steeper
/// is_beach: true if close to water level
fn get_surface_block(elevation: f32, slope: f32, is_beach: bool) -> BlockID {
    const WATER_LEVEL: f32 = 0.5;

    if elevation < WATER_LEVEL {
        // Underwater - ocean floor
        if elevation < 0.25 {
            BlockID::Gravel // Deep ocean floor
        } else {
            BlockID::Sand // Shallow ocean floor
        }
    } else if is_beach {
        BlockID::Sand // Beach near water
    } else if slope > 0.06 {
        BlockID::Stone // Steep slope = exposed rock
    } else if slope > 0.03 {
        BlockID::Gravel // Moderate slope = rocky
    } else {
        BlockID::Grass // Flat = grass
    }
}

/// Get subsurface block based on slope
fn get_subsurface_block(elevation: f32, slope: f32, is_beach: bool) -> BlockID {
    const WATER_LEVEL: f32 = 0.5;

    if elevation < WATER_LEVEL || is_beach {
        BlockID::Sand // Ocean/beach has sand below
    } else if slope > 0.03 {
        BlockID::Stone // Rocky areas have stone below
    } else {
        BlockID::CoarseDirt // Flat areas have dirt
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

            // Calculate slope by sampling neighbors
            let elev_px = interpolate_elevation((world_x + 1.5, world_z + 0.5), &corners);
            let elev_mx = interpolate_elevation((world_x - 0.5, world_z + 0.5), &corners);
            let elev_pz = interpolate_elevation((world_x + 0.5, world_z + 1.5), &corners);
            let elev_mz = interpolate_elevation((world_x + 0.5, world_z - 0.5), &corners);

            let dx = (elev_px - elev_mx).abs();
            let dz = (elev_pz - elev_mz).abs();
            let slope = (dx * dx + dz * dz).sqrt();

            // Get Voronoi cell info for this position
            let voronoi = find_nearest_cells(macro_data.seed, world_x, world_z, settings.jitter);
            let cell_type = get_cell_type(
                macro_data.seed,
                voronoi.nearest_cell,
                settings.elevation_noise_scale,
                settings.water_threshold,
                settings.island_radius,
                settings.ocean_ratio,
                settings.shape_roundness,
                settings.jitter,
                settings.noise_octaves,
            );

            // Beach = land just above water level (elevation 0.5 to 0.55)
            // This creates a smooth sand line along the actual coastline
            let is_beach = elevation >= 0.5 && elevation < 0.55;

            // Convert elevation to Y coordinate
            let surface_y = elevation_to_surface_y(
                elevation,
                settings.sea_level,
            ) as usize;

            // Get blocks for this column
            let surface_block = get_surface_block(elevation, slope, is_beach);
            let subsurface_block = get_subsurface_block(elevation, slope, is_beach);

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
                    // Surface block with coast visualization
                    let is_edge = is_on_voronoi_edge(&voronoi, settings.edge_threshold);
                    let is_coast_cell = cell_type == CellType::Coast;

                    let block_id = if is_edge {
                        BlockID::Stone.id()
                    } else if is_coast_cell && elevation < 0.5 {
                        // Coast cell underwater - purple debug
                        BlockID::AmethystBlock.id()
                    } else if is_beach {
                        // Coast cell on land - sand beach
                        BlockID::Sand.id()
                    } else {
                        surface_block.id()
                    };

                    section_data.insert(&pos, BlockDataInfo::create(block_id));
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
