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
use settings::GeneratorSettings;

mod generate_section_data;
mod generate_world_macro;
mod settings;
mod utils;
mod voronoi;

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
    let world_macro_data = generate_world_macro(event.get_seed());
    Ok(world_macro_data)
}

#[event_handler]
pub fn on_chunk_generate(event: ChunkGenerateEvent) -> Result<ChunkData, Error> {
    let chunk_position = event.get_chunk_position();
    let world_settings = event.get_world_settings();

    let settings = GeneratorSettings::from_option(world_settings.get_settings());

    let macro_data: MacroData = serde_yaml::from_value(
        world_settings.get_world_macro_data().get_data().clone(),
    )
    .map_err(|e| Error::msg(format!("MacroData parse error: {}", e)))?;

    extism_pdk::log!(
        extism_pdk::LogLevel::Debug,
        "MacroData seed: {}",
        macro_data.seed
    );

    let mut chunk_data = ChunkData::default();
    for y in 0..VERTICAL_SECTIONS {
        let chunk_section = generate_section_data(&chunk_position, y, &macro_data, &settings);
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
