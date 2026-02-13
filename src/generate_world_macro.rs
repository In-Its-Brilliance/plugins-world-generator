use common::chunks::chunk_data::WorldMacroData;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[serde_inline_default]
#[derive(Serialize, Deserialize)]
pub struct GeneratorSettings {
    #[serde_inline_default(3)]
    pub island_count: usize,
    #[serde_inline_default(200)]
    pub min_radius: u32,
    #[serde_inline_default(500)]
    pub max_radius: u32,
    #[serde_inline_default(2000)]
    pub world_spread: i32,
    #[serde_inline_default(32)]
    pub sea_level: u16,
    #[serde_inline_default(0.002)]
    pub noise_scale: f32,
    #[serde_inline_default(48)]
    pub peak_height: u16,
}

impl GeneratorSettings {
    pub fn from_option(settings: &Option<serde_yaml::Value>) -> Self {
        match settings {
            Some(value) => serde_yaml::from_value(value.clone()).unwrap_or_default(),
            None => Self::default(),
        }
    }
}

impl Default for GeneratorSettings {
    fn default() -> Self {
        serde_yaml::from_value(serde_yaml::Value::Mapping(Default::default())).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
pub struct IslandMacro {
    pub x: i32,
    pub z: i32,
    pub radius: u32,
    pub seed: u64,
    pub peak_height: u16,
    pub noise_scale: f32,
}

#[derive(Serialize, Deserialize)]
pub struct MacroData {
    pub sea_level: u16,
    pub islands: Vec<IslandMacro>,
}

pub fn generate_world_macro(seed: u64, settings: &Option<serde_yaml::Value>) -> WorldMacroData {
    let settings = GeneratorSettings::from_option(settings);
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut islands: Vec<IslandMacro> = Vec::new();

    // Спавн-остров всегда в центре
    islands.push(IslandMacro {
        x: 0,
        z: 0,
        radius: settings.max_radius,
        seed: rng.gen(),
        peak_height: settings.peak_height,
        noise_scale: settings.noise_scale,
    });

    // Остальные рандомно
    for _ in 1..settings.island_count {
        islands.push(IslandMacro {
            x: rng.gen_range(-settings.world_spread..settings.world_spread),
            z: rng.gen_range(-settings.world_spread..settings.world_spread),
            radius: rng.gen_range(settings.min_radius..settings.max_radius),
            seed: rng.gen(),
            peak_height: settings.peak_height,
            noise_scale: settings.noise_scale,
        });
    }

    let macro_data = MacroData {
        sea_level: settings.sea_level,
        islands,
    };

    let data = serde_yaml::to_value(&macro_data).unwrap();
    WorldMacroData::create(data)
}
