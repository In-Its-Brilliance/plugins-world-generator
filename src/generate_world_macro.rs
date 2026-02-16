use common::chunks::chunk_data::WorldMacroData;
use serde::{Deserialize, Serialize};

use crate::settings::GeneratorSettings;

#[derive(Serialize, Deserialize)]
pub struct MacroData {}

pub fn generate_world_macro(_seed: u64, _settings: &GeneratorSettings) -> WorldMacroData {
    WorldMacroData::default()
}
