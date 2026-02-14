use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct GeneratorSettings {
    pub sea_level: u16,
}
