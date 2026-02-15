use common::chunks::chunk_data::WorldMacroData;

/// Minimal macro data - just the seed for deterministic generation
#[derive(serde::Serialize, serde::Deserialize)]
pub struct MacroData {
    pub seed: u64,
}

/// Generate world macro data (minimal - just seed)
pub fn generate_world_macro(seed: u64) -> WorldMacroData {
    extism_pdk::log!(
        extism_pdk::LogLevel::Info,
        "World macro created with seed: {}",
        seed
    );

    let macro_data = MacroData { seed };
    let data = serde_yaml::to_value(&macro_data).unwrap();
    WorldMacroData::create(data)
}
