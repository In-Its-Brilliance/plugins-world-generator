use common::{
    chunks::chunk_data::{ChunkData, WorldMacroData},
    event_handler,
    plugin_api::events::{
        generage_chunk::ChunkGenerateEvent, generage_world_macro::GenerateWorldMacroEvent,
        plugin_load::PluginLoadEvent, plugin_unload::PluginUnloadEvent,
    },
    VERTICAL_SECTIONS,
};
use extism_pdk::Error;
use generate_section_data::generate_section_data;
use generate_world_macro::{generate_world_macro, MacroData};

mod generate_section_data;
mod generate_world_macro;

/// Two-tier generation: macro-level stores island positions and parameters,
/// micro-level computes terrain per chunk.
///
/// Macro data is compact and does not grow with world size.
/// Islands define landmass shape only — biomes are computed per-chunk
/// via temperature/moisture noise combined with elevation.

#[event_handler]
pub fn on_plugin_load(event: PluginLoadEvent) -> Result<(), Error> {
    event.register_world_generator("default")?;
    extism_pdk::log!(
        extism_pdk::LogLevel::Info,
        "World Generator plugin enabled!"
    );
    Ok(())
}

#[event_handler]
pub fn on_generate_world_macro(event: GenerateWorldMacroEvent) -> Result<WorldMacroData, Error> {
    let world_macro_data = generate_world_macro(event.get_seed(), event.get_settings());
    Ok(world_macro_data)
}

#[event_handler]
pub fn on_chunk_generate(event: ChunkGenerateEvent) -> Result<ChunkData, Error> {
    let chunk_position = event.get_chunk_position();
    let world_settings = event.get_world_settings();

    let macro_data: MacroData = serde_yaml::from_value(
        serde_yaml::to_value(world_settings.get_world_macro_data().get_data()).unwrap()
    ).map_err(|e| Error::msg(format!("MacroData parse error: {}", e)))?;

    let mut chunk_data = ChunkData::default();
    for y in 0..VERTICAL_SECTIONS {
        let chunk_section = generate_section_data(&chunk_position, y, &macro_data);
        chunk_data.push_section(chunk_section);
    }

    Ok(chunk_data)
}

#[event_handler]
pub fn on_plugin_unload(_event: PluginUnloadEvent) {
    extism_pdk::log!(
        extism_pdk::LogLevel::Info,
        "World Generator plugin disabled!"
    );
}
