use common::{
    chunks::{
        block_position::ChunkBlockPosition,
        chunk_data::{BlockDataInfo, ChunkSectionData},
        chunk_position::ChunkPosition,
    },
    default_blocks_ids::BlockID,
};

use crate::{generate_world_macro::MacroData, settings::GeneratorSettings};

pub fn generate_section_data(
    _seed: u64,
    _chunk_position: &ChunkPosition,
    _vertical_index: usize,
    _macro_data: &MacroData,
    _settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();
    section_data.insert(
        &ChunkBlockPosition::new(0, 0, 0),
        BlockDataInfo::create(BlockID::Grass.id()),
    );
    section_data
}
