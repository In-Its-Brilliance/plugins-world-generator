use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct GeneratorSettings {
    pub sea_level: u16,
    pub island: IslandSettings,
}

/// Основные параметры острова.
#[derive(Serialize, Deserialize)]
pub struct IslandSettings {
    /// Радиус от центра до дальнего берега (в блоках).
    pub size: f64,
    /// Максимальный сдвиг центра от начала координат (в блоках).
    pub center_offset: f64,
    /// Включить отображение границ Voronoi-ячеек.
    pub voronoi_borders: bool,

    pub coastline: CoastlineSettings,
    pub mountains: MountainSettings,
    pub terrain: TerrainSettings,
    pub voronoi: VoronoiSettings,
    pub ridge: RidgeSettings,
    pub shore: ShoreSettings,
}

/// Параметры береговой линии.
#[derive(Serialize, Deserialize)]
pub struct CoastlineSettings {
    /// Сила влияния noise на форму берега.
    pub spread: f64,
    /// Базовая частота noise береговой линии.
    pub fractal: f64,
}

/// Параметры горной гряды.
#[derive(Serialize, Deserialize)]
pub struct MountainSettings {
    /// Расстояние от хребта для пика горной гряды (в блоках).
    pub width: f64,
    /// Максимальная высота горной гряды над sea_level.
    pub max_peak_height: i32,
    /// Максимальная глубина океанского дна ниже sea_level.
    pub max_ocean_depth: f64,
    /// Соотношение высоты гор к ширине суши.
    pub height_ratio: f64,
    /// Шаг изменения высоты на сегмент хребта (в блоках).
    pub height_step: f64,
    /// Базовая вероятность подъёма на сегменте.
    pub base_up_chance: f64,
    /// Максимальная вероятность подъёма.
    pub max_up_chance: f64,
}

/// Параметры микрорельефа поверхности.
#[derive(Serialize, Deserialize)]
pub struct TerrainSettings {
    /// Максимальная амплитуда поверхностного шума (в блоках).
    pub surface_noise_amplitude: f64,
    /// Ширина пляжной полосы от линии воды (в блоках).
    pub beach_width: f64,
}

/// Параметры Voronoi-сетки.
#[derive(Serialize, Deserialize)]
pub struct VoronoiSettings {
    /// Средний размер ячейки (в блоках).
    pub cell_size: f64,
    /// Порог расстояния до границы (в блоках).
    pub border_thickness: f64,
    /// Высота границы над поверхностью (в блоках).
    pub border_height: i32,
}

/// Параметры формы хребта.
#[derive(Serialize, Deserialize)]
pub struct RidgeSettings {
    /// Количество сегментов в главном хребте.
    pub spine_segments: usize,
    /// Количество сегментов в рукаве.
    pub arm_segments: usize,
    /// Количество рукавов от главного хребта.
    pub arm_count: usize,
    /// Извилистость хребта (макс. отклонение на сегмент).
    pub spine_wobble: f64,
    /// Извилистость рукавов.
    pub arm_wobble: f64,
}

/// Параметры берегового модификатора высоты.
#[derive(Serialize, Deserialize)]
pub struct ShoreSettings {
    /// Максимальное значение modifier на суше (в блоках).
    pub mod_up: f64,
    /// Максимальное значение modifier под водой (в блоках).
    pub mod_down: f64,
    /// Расстояние на котором modifier достигает максимума (в блоках).
    pub mod_radius: f64,
}
