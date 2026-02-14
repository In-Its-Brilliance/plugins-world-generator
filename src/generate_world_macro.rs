use common::chunks::chunk_data::WorldMacroData;

use crate::settings::GeneratorSettings;

pub fn generate_world_macro(_seed: u64, _settings: &GeneratorSettings) -> WorldMacroData {
    WorldMacroData::default()
}
