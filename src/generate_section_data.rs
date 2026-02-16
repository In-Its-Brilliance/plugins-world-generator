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
};

fn elevation_to_surface_y(elevation: f32, sea_level: u16) -> u16 {
    const OCEAN_FLOOR_DEPTH: u16 = 15;
    const MAX_MOUNTAIN_HEIGHT: u16 = 40;
    const WATER_LEVEL: f32 = 0.5;

    let max_terrain_height = sea_level + MAX_MOUNTAIN_HEIGHT;

    if elevation < WATER_LEVEL {
        let t = elevation / WATER_LEVEL;
        let min_y = sea_level.saturating_sub(OCEAN_FLOOR_DEPTH);
        min_y + ((sea_level - min_y) as f32 * t) as u16
    } else {
        let t = (elevation - WATER_LEVEL) / (1.0 - WATER_LEVEL);
        let height_range = max_terrain_height - sea_level;
        sea_level + (height_range as f32 * t) as u16
    }
}

fn get_surface_block(elevation: f32, is_beach: bool) -> BlockID {
    const WATER_LEVEL: f32 = 0.5;

    if elevation < WATER_LEVEL {
        if elevation < 0.25 {
            BlockID::Gravel
        } else {
            BlockID::Sand
        }
    } else if is_beach {
        BlockID::Sand
    } else {
        BlockID::Grass
    }
}

fn get_subsurface_block(elevation: f32, is_beach: bool) -> BlockID {
    const WATER_LEVEL: f32 = 0.5;

    if elevation < WATER_LEVEL || is_beach {
        BlockID::Sand
    } else {
        BlockID::CoarseDirt
    }
}

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

    let center_cell = world_to_cell(chunk_x + CHUNK_SIZE as f32 / 2.0, chunk_z + CHUNK_SIZE as f32 / 2.0);
    let mut corners = find_corners_in_region(
        macro_data.seed,
        center_cell.0,
        center_cell.1,
        2,
        settings.jitter,
    );

    assign_corner_elevations(macro_data.seed, &mut corners, &terrain_params, MAX_COAST_DISTANCE);

    for x in 0_u8..(CHUNK_SIZE as u8) {
        for z in 0_u8..(CHUNK_SIZE as u8) {
            let world_x = chunk_x + x as f32;
            let world_z = chunk_z + z as f32;
            let world_point = (world_x + 0.5, world_z + 0.5);

            let elevation = interpolate_elevation(world_point, &corners);

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

            let is_beach = elevation >= 0.5 && elevation < 0.55;

            let surface_y = elevation_to_surface_y(elevation, settings.sea_level) as usize;

            let surface_block = get_surface_block(elevation, is_beach);
            let subsurface_block = get_subsurface_block(elevation, is_beach);

            for y in 0_u8..(CHUNK_SIZE as u8) {
                let y_global = section_y_start + y as usize;
                let pos = ChunkBlockPosition::new(x, y, z);
                let sea_level = settings.sea_level as usize;

                if y_global == 0 {
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Bedrock.id()));
                } else if y_global < surface_y.saturating_sub(3) {
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Stone.id()));
                } else if y_global < surface_y {
                    section_data.insert(&pos, BlockDataInfo::create(subsurface_block.id()));
                } else if y_global == surface_y {
                    let is_edge = is_on_voronoi_edge(&voronoi, settings.edge_threshold);
                    let is_coast_cell = cell_type == CellType::Coast;

                    let block_id = if is_edge {
                        BlockID::Stone.id()
                    } else if is_coast_cell && elevation < 0.5 {
                        BlockID::AmethystBlock.id()
                    } else if is_beach {
                        BlockID::Sand.id()
                    } else {
                        surface_block.id()
                    };

                    section_data.insert(&pos, BlockDataInfo::create(block_id));
                } else if y_global > surface_y && y_global <= sea_level {
                    section_data.insert(&pos, BlockDataInfo::create(BlockID::Water.id()));
                }
            }
        }
    }

    section_data
}
