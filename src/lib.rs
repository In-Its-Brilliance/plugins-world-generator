use common::{
    event_handler,
    plugin_api::events::{plugin_load::PluginLoadEvent, plugin_unload::PluginUnloadEvent},
};

#[event_handler]
pub fn on_plugin_load(event: PluginLoadEvent) {
    extism_pdk::log!(extism_pdk::LogLevel::Info, "World Generator enabled!");
}

#[event_handler]
pub fn on_plugin_unload(event: PluginUnloadEvent) {
    extism_pdk::log!(extism_pdk::LogLevel::Info, "World Generator disabled!");
}
