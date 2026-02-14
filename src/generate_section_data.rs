use common::chunks::{chunk_data::ChunkSectionData, chunk_position::ChunkPosition};
use serde::{Deserialize, Serialize};

use crate::settings::GeneratorSettings;

#[derive(Serialize, Deserialize)]
pub struct MacroData {}

pub fn generate_section_data(
    _chunk_position: &ChunkPosition,
    _vertical_index: usize,
    _macro_data: &MacroData,
    _settings: &GeneratorSettings,
) -> ChunkSectionData {
    let section_data = ChunkSectionData::default();
    section_data
}
