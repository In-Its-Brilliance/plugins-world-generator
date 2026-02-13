use common::{
    chunks::{
        block_position::ChunkBlockPosition,
        chunk_data::{BlockDataInfo, ChunkSectionData},
        chunk_position::ChunkPosition,
    },
    default_blocks_ids::BlockID,
    CHUNK_SIZE,
};
use fastnoise_lite::{FastNoiseLite, NoiseType};

use crate::generate_world_macro::MacroData;

pub fn generate_section_data(
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    macro_data: &MacroData,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();
    let section_y_offset = vertical_index as i32 * CHUNK_SIZE as i32;

    for x in 0..(CHUNK_SIZE as u8) {
        for z in 0..(CHUNK_SIZE as u8) {
            let world_x = chunk_position.x as f32 * CHUNK_SIZE as f32 + x as f32;
            let world_z = chunk_position.z as f32 * CHUNK_SIZE as f32 + z as f32;

            let height = calc_height(world_x, world_z, macro_data);

            for y in 0..(CHUNK_SIZE as u8) {
                let world_y = section_y_offset + y as i32;

                let block = pick_block(world_y, height, macro_data.sea_level as i32);
                if let Some(block_id) = block {
                    let pos = ChunkBlockPosition::new(x, y, z);
                    section_data.insert(&pos, BlockDataInfo::create(block_id));
                }
            }
        }
    }

    section_data
}

fn calc_height(world_x: f32, world_z: f32, macro_data: &MacroData) -> i32 {
    let mut best_influence = 0.0_f32;
    let mut best_island = None;

    for island in &macro_data.islands {
        let dx = world_x - island.x as f32;
        let dz = world_z - island.z as f32;
        let dist = (dx * dx + dz * dz).sqrt();

        // Noise-deformed coastline
        let mut noise = FastNoiseLite::with_seed(island.seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(0.005));
        let deform = noise.get_noise_2d(world_x, world_z) * 0.3 + 1.0;
        let effective_radius = island.radius as f32 * deform;

        if dist < effective_radius {
            let t = 1.0 - (dist / effective_radius);
            let falloff = t * t * (3.0 - 2.0 * t);
            if falloff > best_influence {
                best_influence = falloff;
                best_island = Some(island);
            }
        }
    }

    if let Some(island) = best_island {
        // fBM: multiple octaves for natural terrain
        let mut terrain_noise = FastNoiseLite::with_seed(island.seed as i32 + 1);
        terrain_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        terrain_noise.set_frequency(Some(island.noise_scale));
        terrain_noise.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
        terrain_noise.set_fractal_octaves(Some(5));
        terrain_noise.set_fractal_lacunarity(Some(2.0));
        terrain_noise.set_fractal_gain(Some(0.5));

        let n = (terrain_noise.get_noise_2d(world_x, world_z) + 1.0) * 0.5;

        let height = macro_data.sea_level as f32
            + best_influence * island.peak_height as f32 * n;
        height as i32
    } else {
        macro_data.sea_level as i32 - 1
    }
}

fn pick_block(world_y: i32, height: i32, sea_level: i32) -> Option<u16> {
    if world_y > height && world_y > sea_level {
        return None; // air
    }
    if world_y > height && world_y <= sea_level {
        return Some(BlockID::Water.id());
    }
    if world_y == height {
        if height <= sea_level + 2 {
            return Some(BlockID::Sand.id());
        }
        return Some(BlockID::Grass.id());
    }
    if world_y >= height - 3 {
        return Some(BlockID::CoarseDirt.id());
    }
    Some(BlockID::Stone.id())
}
