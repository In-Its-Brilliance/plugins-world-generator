use common::plugin_api::BrilliancePlugin;

struct WorldGenerator;

impl BrilliancePlugin for WorldGenerator {
    fn on_enable() {
        extism_pdk::log!(extism_pdk::LogLevel::Info, "World Generator enabled!");
    }

    fn on_disable() {
    }
}

common::brilliance_plugin!(WorldGenerator);
