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

// --- Единственный параметр размера -------------------------------------------

/// Общий размер острова (радиус от центра до дальнего берега, в блоках).
/// Все производные параметры рассчитываются из этого значения.
const ISLAND_SIZE: f64 = 400.0;

// --- Параметры береговой линии -----------------------------------------------

/// Сила влияния noise на береговую линию.
const COAST_SPREAD: f64 = 0.8;

/// Извилистость береговой линии (базовая частота noise).
const COAST_FRACTAL: f64 = 0.006;

// --- Постоянные параметры (не зависят от размера) ----------------------------

/// Средний размер Voronoi-ячейки в блоках (шаг jittered grid)
const VORONOI_CELL_SIZE: f64 = 13.0;

/// Порог расстояния до границы между ячейками (в блоках)
const BORDER_THICKNESS: f64 = 0.45;

/// Высота границы над поверхностью ячейки (в блоках)
const BORDER_HEIGHT: i32 = 1;

/// Расстояние от хребта для пика горной гряды (ширина ~1 ячейки)
const MOUNTAIN_WIDTH: f64 = 7.0;

/// Максимальная высота горной гряды над sea_level
const MAX_PEAK_HEIGHT: i32 = 80;

/// Максимальная глубина океанского дна ниже sea_level
const MAX_OCEAN_DEPTH: f64 = 20.0;

/// Шаг изменения высоты на сегмент (в блоках)
const HEIGHT_STEP: f64 = 6.0;

/// Базовая вероятность подъёма (остальное -- спуск)
const BASE_UP_CHANCE: f64 = 0.35;

/// Максимальная вероятность подъёма (для больших островов)
const MAX_UP_CHANCE: f64 = 0.4;

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

/// Ширина пляжной полосы от линии воды вглубь суши (в блоках).
const BEACH_WIDTH: f64 = 12.0;

/// Максимальное значение shore modifier на суше (в блоках).
const SHORE_MOD_UP: f64 = 6.0;

/// Максимальное значение shore modifier под водой (в блоках).
const SHORE_MOD_DOWN: f64 = 3.0;

/// Расстояние (в блоках) на котором modifier достигает максимума.
const SHORE_MOD_RADIUS: f64 = 40.0;

// --- Производные параметры из ISLAND_SIZE ------------------------------------

struct IslandParams {
    spine_length: f64,
    arm_length: f64,
    land_width: f64,
    highland_width: f64,
    max_height: f64,
    /// Порог field-значения для пляжной зоны.
    beach_field_threshold: f64,
}

impl IslandParams {
    fn from_size(size: f64) -> Self {
        let spine_length = size * 0.5;
        let arm_length = size * 0.3;
        let land_width = size * 0.4;
        let max_height = (land_width * 0.8).min(MAX_PEAK_HEIGHT as f64);
        Self {
            spine_length,
            arm_length,
            land_width,
            highland_width: land_width * 0.42,
            max_height,
            beach_field_threshold: BEACH_WIDTH / land_width,
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

#[derive(Clone, Copy)]
struct RidgePoint {
    x: f64,
    z: f64,
    h: f64,
}

struct Polyline {
    points: Vec<RidgePoint>,
}

impl Polyline {
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
    fn generate(seed: u64, params: &IslandParams) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut polylines = Vec::new();

        let island_size = params.spine_length + params.arm_length;
        let up_chance = (BASE_UP_CHANCE
            + (island_size / 2000.0) * (MAX_UP_CHANCE - BASE_UP_CHANCE))
            .min(MAX_UP_CHANCE);

        let spine_angle: f64 = rng.gen::<f64>() * std::f64::consts::PI;

        let spine_fwd = generate_ridge(
            &mut rng,
            0.0, 0.0,
            params.max_height,
            spine_angle,
            params.spine_length,
            SPINE_SEGMENTS,
            SPINE_WOBBLE,
            up_chance,
            params.max_height,
        );

        let spine_back = generate_ridge(
            &mut rng,
            0.0, 0.0,
            params.max_height,
            spine_angle + std::f64::consts::PI,
            params.spine_length,
            SPINE_SEGMENTS,
            SPINE_WOBBLE,
            up_chance,
            params.max_height,
        );

        let mut spine_points: Vec<RidgePoint> = spine_back.iter()
            .rev()
            .map(|&(x, z, h)| RidgePoint { x, z, h })
            .collect();
        for &(x, z, h) in spine_fwd.iter().skip(1) {
            spine_points.push(RidgePoint { x, z, h });
        }

        let all_spine: Vec<(f64, f64, f64)> = spine_points.iter()
            .map(|p| (p.x, p.z, p.h))
            .collect();

        polylines.push(Polyline { points: spine_points });

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
                params.max_height,
            );

            let arm_points: Vec<RidgePoint> = arm_raw.iter()
                .map(|&(x, z, h)| RidgePoint { x, z, h })
                .collect();

            polylines.push(Polyline { points: arm_points });
        }

        Self { polylines }
    }

    fn query_elevation(&self, px: f64, pz: f64, params: &IslandParams) -> (f64, f64) {
        let mut min_dist = f64::MAX;
        let mut best_elev = 0.0;

        for poly in &self.polylines {
            let (d, h) = poly.nearest_point(px, pz);

            if d < min_dist {
                min_dist = d;
            }

            let height_ratio = h / params.max_height;
            let base = ISLAND_SIZE * 0.05;
            let extra = height_ratio * height_ratio * ISLAND_SIZE * 0.07;
            let dynamic_falloff = base + extra;

            let falloff = (-d / dynamic_falloff).exp();
            let elev = falloff * h;
            if elev > best_elev {
                best_elev = elev;
            }
        }

        (min_dist, best_elev)
    }
}

fn generate_ridge(
    rng: &mut SmallRng,
    start_x: f64, start_z: f64,
    start_height: f64,
    angle: f64,
    length: f64,
    num_segments: usize,
    wobble: f64,
    up_chance: f64,
    max_height: f64,
) -> Vec<(f64, f64, f64)> {
    let mut points = Vec::with_capacity(num_segments + 1);
    let h0 = start_height.min(max_height);
    points.push((start_x, start_z, h0));

    let seg_length = length / num_segments as f64;
    let mut current_angle = angle;
    let mut cx = start_x;
    let mut cz = start_z;
    let mut h = h0;

    for i in 0..num_segments {
        current_angle += (rng.gen::<f64>() - 0.5) * wobble / length * std::f64::consts::TAU;
        cx += current_angle.cos() * seg_length;
        cz += current_angle.sin() * seg_length;

        if rng.gen::<f64>() < up_chance {
            h += HEIGHT_STEP;
        } else {
            h -= HEIGHT_STEP;
        }

        if i >= num_segments - 3 {
            h -= HEIGHT_STEP;
        }

        h = h.clamp(0.0, max_height);
        points.push((cx, cz, h));
    }

    points
}

// --- Когерентный noise для береговой линии ------------------------------------

fn hash_2d(ix: i64, iz: i64, seed: u64) -> f64 {
    let mut h = seed;
    h = h.wrapping_add((ix as u64).wrapping_mul(6364136223846793005));
    h = h.wrapping_add((iz as u64).wrapping_mul(1442695040888963407));
    h = h.wrapping_mul(h.wrapping_shr(16).wrapping_add(1376312589));
    h = h ^ h.wrapping_shr(13);
    h = h.wrapping_mul(h.wrapping_shr(16).wrapping_add(0x45d9f3b));
    (h & 0x7FFFFFFF) as f64 / 0x7FFFFFFF as f64 * 2.0 - 1.0
}

fn coherent_noise_2d(x: f64, z: f64, seed: u64) -> f64 {
    let ix = x.floor() as i64;
    let iz = z.floor() as i64;
    let fx = x - x.floor();
    let fz = z - z.floor();

    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);

    let n00 = hash_2d(ix, iz, seed);
    let n10 = hash_2d(ix + 1, iz, seed);
    let n01 = hash_2d(ix, iz + 1, seed);
    let n11 = hash_2d(ix + 1, iz + 1, seed);

    let nx0 = n00 + sx * (n10 - n00);
    let nx1 = n01 + sx * (n11 - n01);
    nx0 + sz * (nx1 - nx0)
}

fn fbm_noise(x: f64, z: f64, seed: u64, octaves: usize) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;

    for i in 0..octaves {
        value += coherent_noise_2d(
            x * frequency,
            z * frequency,
            seed.wrapping_add(i as u64 * 31),
        ) * amplitude;
        max_amp += amplitude;
        amplitude *= 0.65;
        frequency *= 2.0;
    }

    value / max_amp
}

fn island_field(
    x: f64, z: f64,
    raw_dist: f64,
    land_width: f64,
    seed: u64,
) -> f64 {
    let ridge_field = ((land_width - raw_dist) / land_width).clamp(0.0, 1.0);

    let noise_field = fbm_noise(
        x * COAST_FRACTAL,
        z * COAST_FRACTAL,
        seed.wrapping_add(7777),
        6,
    );

    ridge_field + noise_field * COAST_SPREAD
}

/// Оценка горизонтального расстояния до береговой линии (field=0) в блоках.
///
/// Использует средний градиент field (~ 1/land_width per block).
/// Результат со знаком: положительный на суше, отрицательный в океане.
/// Гладкий, без шума от noise.
fn distance_to_shore(field: f64, land_width: f64) -> f64 {
    field * land_width
}

/// Единая высота для любой точки мира.
fn compute_elevation(
    wx: f64, wz: f64,
    skeleton: &IslandSkeleton,
    params: &IslandParams,
    seed: u64,
) -> f64 {
    let (raw_dist, ridge_elev) = skeleton.query_elevation(wx, wz, params);
    let field = island_field(wx, wz, raw_dist, params.land_width, seed);

    let base = if field > 0.0 {
        let shore_factor = field.min(1.0);
        ridge_elev * shore_factor
    } else {
        field * MAX_OCEAN_DEPTH
    };

    // Shore modifier: плавный подъём от берега
    let shore_dist = distance_to_shore(field, params.land_width);
    let abs_dist = shore_dist.abs().min(SHORE_MOD_RADIUS);
    let t = abs_dist / SHORE_MOD_RADIUS;
    let ease = t * (2.0 - t);
    let modifier = if shore_dist > 0.0 {
        ease * SHORE_MOD_UP
    } else {
        -ease * SHORE_MOD_DOWN
    };

    base + modifier
}

fn classify_by_field(
    wx: f64, wz: f64,
    field: f64,
    skeleton: &IslandSkeleton,
    params: &IslandParams,
) -> CellZone {
    if field <= 0.0 {
        return CellZone::Ocean;
    }

    if field < params.beach_field_threshold {
        return CellZone::Beach;
    }

    let (raw_dist, _) = skeleton.query_elevation(wx, wz, params);
    if raw_dist <= MOUNTAIN_WIDTH {
        CellZone::Mountain
    } else if raw_dist <= params.highland_width {
        CellZone::Highland
    } else {
        CellZone::Lowland
    }
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

    let max_depth = MAX_OCEAN_DEPTH as i32 + SHORE_MOD_DOWN as i32;
    if section_y_offset > base_y + MAX_PEAK_HEIGHT + SHORE_MOD_UP as i32 + BORDER_HEIGHT + 2 {
        return section_data;
    }
    if section_y_offset + CHUNK_SIZE as i32 <= base_y - max_depth - 4 {
        return section_data;
    }

    let params = IslandParams::from_size(ISLAND_SIZE);
    let skeleton = IslandSkeleton::generate(seed, &params);

    let chunk_wx = chunk_position.x as f64 * CHUNK_SIZE as f64;
    let chunk_wz = chunk_position.z as f64 * CHUNK_SIZE as f64;

    let max_noise_reach = COAST_SPREAD * params.land_width;
    let extent = params.spine_length + params.arm_length
        + params.land_width + max_noise_reach;
    let points = generate_voronoi_points(
        seed, -extent, -extent, extent, extent,
    );

    if points.len() < 3 {
        return section_data;
    }

    for x in 0..(CHUNK_SIZE as u8) {
        for z in 0..(CHUNK_SIZE as u8) {
            let wx = chunk_wx + x as f64;
            let wz = chunk_wz + z as f64;

            let (nearest, second, _nearest_idx) =
                find_two_nearest_with_index(&points, wx, wz);
            let dist_to_border = (second - nearest) * 0.5;

            let (raw_dist, _ridge_elev) = skeleton.query_elevation(wx, wz, &params);
            let field = island_field(wx, wz, raw_dist, params.land_width, seed);
            let elevation_f = compute_elevation(wx, wz, &skeleton, &params, seed);
            let elevation = elevation_f as i32;
            let is_underwater = field <= 0.0;

            let zone = classify_by_field(
                wx, wz, field, &skeleton, &params,
            );

            let is_border = dist_to_border < BORDER_THICKNESS
                && !is_underwater
                && zone != CellZone::Beach
                && zone != CellZone::Ocean;

            let top_y = if is_border {
                base_y + elevation + BORDER_HEIGHT
            } else {
                base_y + elevation
            };

            for y in 0..(CHUNK_SIZE as u8) {
                let world_y = section_y_offset + y as i32;

                if world_y > top_y && world_y > base_y {
                    continue;
                }

                let pos = ChunkBlockPosition::new(x, y, z);

                if world_y > top_y && world_y <= base_y {
                    section_data.insert(
                        &pos,
                        BlockDataInfo::create(BlockID::Water.id()),
                    );
                } else if world_y == top_y {
                    if is_underwater {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(BlockID::Sand.id()),
                        );
                    } else if is_border {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(border_block(zone)),
                        );
                    } else {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(surface_block(zone)),
                        );
                    }
                } else if world_y >= top_y - 3 {
                    if is_underwater {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(BlockID::Sandstone.id()),
                        );
                    } else {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(subsurface_block(zone)),
                        );
                    }
                } else {
                    section_data.insert(
                        &pos,
                        BlockDataInfo::create(BlockID::Stone.id()),
                    );
                }
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
