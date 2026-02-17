use common::{
    chunks::{
        block_position::ChunkBlockPosition,
        chunk_data::{BlockDataInfo, ChunkSectionData},
        chunk_position::ChunkPosition,
    },
    default_blocks_ids::BlockID,
    CHUNK_SIZE,
};
use delaunator::Point;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::generate_world_macro::MacroData;
use crate::settings::GeneratorSettings;

// ─── Единственный параметр размера ──────────────────────────────────────────

/// Общий размер острова (радиус от центра до дальнего берега, в блоках).
/// Все производные параметры рассчитываются из этого значения.
const ISLAND_SIZE: f64 = 600.0;

// ─── Постоянные параметры (не зависят от размера) ───────────────────────────

/// Средний размер Voronoi-ячейки в блоках (шаг jittered grid)
const VORONOI_CELL_SIZE: f64 = 13.0;

/// Порог расстояния до границы между ячейками (в блоках)
const BORDER_THICKNESS: f64 = 0.45;

/// Высота границы над поверхностью ячейки (в блоках)
const BORDER_HEIGHT: i32 = 1;

/// Расстояние от хребта для пика горной гряды (ширина ~1 ячейки)
const MOUNTAIN_WIDTH: f64 = 7.0;

/// Максимальная высота горной гряды над sea_level
const MAX_PEAK_HEIGHT: i32 = 48;

/// Шаг изменения высоты на сегмент (в блоках)
const HEIGHT_STEP: f64 = 6.0;

/// Базовая вероятность подъёма (остальное -- спуск)
const BASE_UP_CHANCE: f64 = 0.1;

/// Максимальная вероятность подъёма (для больших островов)
const MAX_UP_CHANCE: f64 = 0.4;

/// Скорость спада высоты от хребта (exp falloff)
const RIDGE_FALLOFF: f64 = 20.0;

/// Количество сегментов в главном хребте
const SPINE_SEGMENTS: usize = 12;

/// Сегментов в рукаве
const ARM_SEGMENTS: usize = 8;

/// Количество рукавов от главного хребта
const ARM_COUNT: usize = 6;

/// Степень извилистости хребта (макс. отклонение на сегмент)
const SPINE_WOBBLE: f64 = 40.0;

/// Извилистость рукавов
const ARM_WOBBLE: f64 = 30.0;

// ─── Производные параметры из ISLAND_SIZE ───────────────────────────────────

/// Все размеры острова, рассчитанные из единого ISLAND_SIZE
struct IslandParams {
    /// Длина главного хребта от центра в каждую сторону
    spine_length: f64,
    /// Длина рукава (в блоках)
    arm_length: f64,
    /// Максимальное расстояние от хребта, в пределах которого есть суша
    land_width: f64,
    /// Расстояние от хребта для высокогорья
    highland_width: f64,
    /// Ширина пляжной зоны (в блоках)
    beach_width: f64,
}

impl IslandParams {
    /// Создаёт все параметры пропорционально из одного размера.
    /// ISLAND_SIZE = spine_length + arm_length + land_width
    fn from_size(size: f64) -> Self {
        let spine_length = size * 0.5;
        let arm_length = size * 0.3;
        let land_width = size * 0.2;
        Self {
            spine_length,
            arm_length,
            land_width,
            highland_width: land_width * 0.42,
            beach_width: 16.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum CellZone {
    Ocean,
    Beach,
    Lowland,
    Highland,
    Mountain,
}

/// Точка на хребте с координатами и высотой
#[derive(Clone, Copy)]
struct RidgePoint {
    x: f64,
    z: f64,
    h: f64,
}

/// Цельная ломаная линия хребта/рукава
struct Polyline {
    points: Vec<RidgePoint>,
}

impl Polyline {
    /// Находит ближайшую точку на всей ломаной.
    /// Возвращает (расстояние, интерполированная высота).
    /// Проекция плавно скользит по цепочке сегментов без скачков.
    fn nearest_point(&self, px: f64, pz: f64) -> (f64, f64) {
        let mut best_dist = f64::MAX;
        let mut best_h = 0.0;

        for w in self.points.windows(2) {
            let (d, h) = project_on_segment(
                px, pz,
                w[0].x, w[0].z, w[0].h,
                w[1].x, w[1].z, w[1].h,
            );
            if d < best_dist {
                best_dist = d;
                best_h = h;
            }
        }

        (best_dist, best_h)
    }
}

/// Проекция точки на отрезок с интерполяцией высоты.
/// Возвращает (расстояние до отрезка, высота в точке проекции).
fn project_on_segment(
    px: f64, pz: f64,
    x0: f64, z0: f64, h0: f64,
    x1: f64, z1: f64, h1: f64,
) -> (f64, f64) {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let len_sq = dx * dx + dz * dz;

    if len_sq < 1e-10 {
        let ex = px - x0;
        let ez = pz - z0;
        return ((ex * ex + ez * ez).sqrt(), h0);
    }

    let t = ((px - x0) * dx + (pz - z0) * dz) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let cx = x0 + t * dx;
    let cz = z0 + t * dz;
    let ex = px - cx;
    let ez = pz - cz;

    ((ex * ex + ez * ez).sqrt(), h0 + t * (h1 - h0))
}

struct IslandSkeleton {
    polylines: Vec<Polyline>,
}

impl IslandSkeleton {
    /// Генерирует хребет + рукава из seed.
    /// Вероятность подъёма зависит от размера острова,
    /// но всегда меньше вероятности спуска.
    fn generate(seed: u64, params: &IslandParams) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut polylines = Vec::new();

        let island_size = params.spine_length + params.arm_length;
        let up_chance = (BASE_UP_CHANCE
            + (island_size / 2000.0) * (MAX_UP_CHANCE - BASE_UP_CHANCE))
            .min(MAX_UP_CHANCE);

        let spine_angle: f64 = rng.gen::<f64>() * std::f64::consts::PI;

        // Хребет вперёд от центра
        let spine_fwd = generate_ridge(
            &mut rng,
            0.0, 0.0,
            MAX_PEAK_HEIGHT as f64,
            spine_angle,
            params.spine_length,
            SPINE_SEGMENTS,
            SPINE_WOBBLE,
            up_chance,
        );

        // Хребет назад от центра
        let spine_back = generate_ridge(
            &mut rng,
            0.0, 0.0,
            MAX_PEAK_HEIGHT as f64,
            spine_angle + std::f64::consts::PI,
            params.spine_length,
            SPINE_SEGMENTS,
            SPINE_WOBBLE,
            up_chance,
        );

        // Объединяем в одну polyline: back(reversed) + fwd
        // Так центр острова -- середина polyline, без стыка
        let mut spine_points: Vec<RidgePoint> = spine_back.iter()
            .rev()
            .map(|&(x, z, h)| RidgePoint { x, z, h })
            .collect();
        // Пропускаем первую точку fwd (она == последняя back, т.е. (0,0))
        for &(x, z, h) in spine_fwd.iter().skip(1) {
            spine_points.push(RidgePoint { x, z, h });
        }

        // Собираем все точки хребта для ответвлений рукавов
        let all_spine: Vec<(f64, f64, f64)> = spine_points.iter()
            .map(|p| (p.x, p.z, p.h))
            .collect();

        polylines.push(Polyline { points: spine_points });

        // Рукава как отдельные polyline
        for _ in 0..ARM_COUNT {
            let idx = rng.gen_range(1..all_spine.len());
            let (ax, az, ah) = all_spine[idx];

            let arm_angle = rng.gen::<f64>() * std::f64::consts::TAU;
            let arm_raw = generate_ridge(
                &mut rng,
                ax, az,
                ah * 0.7,
                arm_angle,
                params.arm_length,
                ARM_SEGMENTS,
                ARM_WOBBLE,
                up_chance,
            );

            let arm_points: Vec<RidgePoint> = arm_raw.iter()
                .map(|&(x, z, h)| RidgePoint { x, z, h })
                .collect();

            polylines.push(Polyline { points: arm_points });
        }

        Self { polylines }
    }

    /// Расстояние до хребта + elevation с exp-спадом.
    /// Для каждой polyline находит ближайшую проекцию,
    /// потом берёт ту, что даёт максимальную effective height.
    fn query_elevation(&self, px: f64, pz: f64) -> (f64, f64) {
        let mut min_dist = f64::MAX;
        let mut best_elev = 0.0;

        for poly in &self.polylines {
            let (d, h) = poly.nearest_point(px, pz);

            if d < min_dist {
                min_dist = d;
            }

            let falloff = (-d / RIDGE_FALLOFF).exp();
            let elev = falloff * h;
            if elev > best_elev {
                best_elev = elev;
            }
        }

        (min_dist, best_elev)
    }
}

/// Генерирует ломаную линию хребта/рукава с высотой.
/// Высота на каждом сегменте: с большей вероятностью спуск,
/// с меньшей -- подъём. Последние 3 сегмента принудительно спускаются.
fn generate_ridge(
    rng: &mut SmallRng,
    start_x: f64, start_z: f64,
    start_height: f64,
    angle: f64,
    length: f64,
    num_segments: usize,
    wobble: f64,
    up_chance: f64,
) -> Vec<(f64, f64, f64)> {
    let mut points = Vec::with_capacity(num_segments + 1);
    points.push((start_x, start_z, start_height));

    let seg_length = length / num_segments as f64;
    let mut current_angle = angle;
    let mut cx = start_x;
    let mut cz = start_z;
    let mut h = start_height;

    for i in 0..num_segments {
        current_angle += (rng.gen::<f64>() - 0.5) * wobble / length * std::f64::consts::TAU;
        cx += current_angle.cos() * seg_length;
        cz += current_angle.sin() * seg_length;

        if rng.gen::<f64>() < up_chance {
            h += HEIGHT_STEP;
        } else {
            h -= HEIGHT_STEP;
        }

        // Последние 3 сегмента принудительно спускаются к нулю
        if i >= num_segments - 3 {
            h -= HEIGHT_STEP;
        }

        h = h.clamp(0.0, MAX_PEAK_HEIGHT as f64);
        points.push((cx, cz, h));
    }

    points
}

pub fn generate_section_data(
    seed: u64,
    chunk_position: &ChunkPosition,
    vertical_index: usize,
    _macro_data: &MacroData,
    settings: &GeneratorSettings,
) -> ChunkSectionData {
    let mut section_data = ChunkSectionData::default();
    let section_y_offset = vertical_index as i32 * CHUNK_SIZE as i32;
    let base_y = settings.sea_level as i32;

    if section_y_offset > base_y + MAX_PEAK_HEIGHT + BORDER_HEIGHT + 2 {
        return section_data;
    }
    if section_y_offset + CHUNK_SIZE as i32 <= base_y - 1 {
        return section_data;
    }

    let params = IslandParams::from_size(ISLAND_SIZE);
    let skeleton = IslandSkeleton::generate(seed, &params);

    let chunk_wx = chunk_position.x as f64 * CHUNK_SIZE as f64;
    let chunk_wz = chunk_position.z as f64 * CHUNK_SIZE as f64;

    let extent = params.spine_length + params.arm_length + params.land_width;
    let points = generate_voronoi_points(
        seed, -extent, -extent, extent, extent,
    );

    if points.len() < 3 {
        return section_data;
    }

    // Зона для каждой Voronoi-точки (для выбора текстуры)
    let cell_zones: Vec<CellZone> = points.iter().map(|p| {
        let (dist, _) = skeleton.query_elevation(p.x, p.y);
        if dist > params.land_width {
            CellZone::Ocean
        } else if dist > params.land_width - params.beach_width {
            CellZone::Beach
        } else if dist > params.highland_width {
            CellZone::Lowland
        } else if dist > MOUNTAIN_WIDTH {
            CellZone::Highland
        } else {
            CellZone::Mountain
        }
    }).collect();

    for x in 0..(CHUNK_SIZE as u8) {
        for z in 0..(CHUNK_SIZE as u8) {
            let wx = chunk_wx + x as f64;
            let wz = chunk_wz + z as f64;

            let (nearest, second, nearest_idx) =
                find_two_nearest_with_index(&points, wx, wz);
            let dist_to_border = (second - nearest) * 0.5;
            let is_border = dist_to_border < BORDER_THICKNESS;

            let zone = cell_zones[nearest_idx];

            if zone == CellZone::Ocean {
                for y in 0..(CHUNK_SIZE as u8) {
                    let world_y = section_y_offset + y as i32;
                    if world_y <= base_y {
                        let pos = ChunkBlockPosition::new(x, y, z);
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(BlockID::Water.id()),
                        );
                    }
                }
                continue;
            }

            let (_, elevation_f) = skeleton.query_elevation(wx, wz);
            let elevation = elevation_f as i32;

            let top_y = if is_border {
                base_y + elevation + BORDER_HEIGHT
            } else {
                base_y + elevation
            };

            for y in 0..(CHUNK_SIZE as u8) {
                let world_y = section_y_offset + y as i32;
                if world_y > top_y {
                    continue;
                }

                let block_id = if world_y == top_y {
                    if is_border {
                        border_block(zone)
                    } else {
                        surface_block(zone)
                    }
                } else if world_y >= top_y - 3 {
                    subsurface_block(zone)
                } else {
                    BlockID::Stone.id()
                };

                let pos = ChunkBlockPosition::new(x, y, z);
                section_data.insert(&pos, BlockDataInfo::create(block_id));
            }
        }
    }

    section_data
}

fn surface_block(zone: CellZone) -> u16 {
    match zone {
        CellZone::Ocean => BlockID::Water.id(),
        CellZone::Beach => BlockID::Sand.id(),
        CellZone::Lowland => BlockID::Grass.id(),
        CellZone::Highland => BlockID::Podzol.id(),
        CellZone::Mountain => BlockID::IronBlock.id(),
    }
}

fn border_block(zone: CellZone) -> u16 {
    match zone {
        CellZone::Ocean => BlockID::Water.id(),
        CellZone::Beach => BlockID::Sandstone.id(),
        CellZone::Lowland => BlockID::CoarseDirt.id(),
        CellZone::Highland => BlockID::Cobblestone.id(),
        CellZone::Mountain => BlockID::Cobblestone.id(),
    }
}

fn subsurface_block(zone: CellZone) -> u16 {
    match zone {
        CellZone::Ocean => BlockID::Stone.id(),
        CellZone::Beach => BlockID::Sandstone.id(),
        CellZone::Lowland => BlockID::CoarseDirt.id(),
        CellZone::Highland => BlockID::Stone.id(),
        CellZone::Mountain => BlockID::Stone.id(),
    }
}

fn find_two_nearest_with_index(
    points: &[Point], wx: f64, wz: f64,
) -> (f64, f64, usize) {
    let mut nearest = f64::MAX;
    let mut second = f64::MAX;
    let mut nearest_idx = 0;

    for (i, p) in points.iter().enumerate() {
        let dx = wx - p.x;
        let dz = wz - p.y;
        let dist = dx * dx + dz * dz;

        if dist < nearest {
            second = nearest;
            nearest = dist;
            nearest_idx = i;
        } else if dist < second {
            second = dist;
        }
    }

    (nearest.sqrt(), second.sqrt(), nearest_idx)
}

fn generate_voronoi_points(
    seed: u64,
    min_x: f64, min_z: f64,
    max_x: f64, max_z: f64,
) -> Vec<Point> {
    let mut points = Vec::new();
    let step = VORONOI_CELL_SIZE;

    let gx0 = (min_x / step).floor() as i64;
    let gz0 = (min_z / step).floor() as i64;
    let gx1 = (max_x / step).ceil() as i64;
    let gz1 = (max_z / step).ceil() as i64;

    for gx in gx0..=gx1 {
        for gz in gz0..=gz1 {
            let cell_seed = seed
                ^ (gx as u64).wrapping_mul(6364136223846793005)
                ^ (gz as u64).wrapping_mul(1442695040888963407);
            let mut rng = SmallRng::seed_from_u64(cell_seed);

            points.push(Point {
                x: gx as f64 * step + rng.gen::<f64>() * step * 0.8,
                y: gz as f64 * step + rng.gen::<f64>() * step * 0.8,
            });
        }
    }

    points
}
