use crate::settings::GeneratorSettings;

impl GeneratorSettings
{
    pub fn from_option(settings: &Option<serde_yaml::Value>) -> Self {
        match settings {
            Some(value) => serde_yaml::from_value(value.clone()).unwrap_or_default(),
            None => Self::default(),
        }
    }
}

impl Default for GeneratorSettings {
    fn default() -> Self {
        let text = include_str!("default_settings.yml");
        serde_yaml::from_str(text).expect("default_settings.yml is invalid")
    }
}
