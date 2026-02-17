use common::{
    chunks::{
        block_position::ChunkBlockPosition,
        chunk_data::{BlockDataInfo, ChunkSectionData},
        chunk_position::ChunkPosition,
    },
    default_blocks_ids::BlockID,
    CHUNK_SIZE,
};
use delaunator::{Point};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::generate_world_macro::MacroData;
use crate::settings::GeneratorSettings;

/// Средний размер Voronoi-ячейки в блоках (шаг jittered grid)
const VORONOI_CELL_SIZE: f64 = 13.0;

/// Порог расстояния до границы между ячейками (в блоках)
const BORDER_THICKNESS: f64 = 0.45;

/// Высота границы над поверхностью ячейки (в блоках)
const BORDER_HEIGHT: i32 = 1;

/// Длина главного хребта от центра в каждую сторону
const SPINE_LENGTH: f64 = 300.0;

/// Количество сегментов в главном хребте
const SPINE_SEGMENTS: usize = 12;

/// Степень извилистости хребта (макс. отклонение на сегмент)
const SPINE_WOBBLE: f64 = 40.0;

/// Количество рукавов от главного хребта
const ARM_COUNT: usize = 6;

/// Длина рукава (в блоках)
const ARM_LENGTH: f64 = 180.0;

/// Сегментов в рукаве
const ARM_SEGMENTS: usize = 8;

/// Извилистость рукавов
const ARM_WOBBLE: f64 = 30.0;

/// Максимальное расстояние от хребта, в пределах которого есть суша
const LAND_WIDTH: f64 = 120.0;

/// Расстояние от хребта для горной зоны
const MOUNTAIN_WIDTH: f64 = 20.0;

/// Расстояние от хребта для высокогорья
const HIGHLAND_WIDTH: f64 = 50.0;

#[derive(Clone, Copy, PartialEq)]
enum CellZone {
    Ocean,
    Beach,
    Lowland,
    Highland,
    Mountain,
}

/// Отрезок горного хребта
struct Segment {
    x0: f64, z0: f64,
    x1: f64, z1: f64,
}

impl Segment {
    /// Минимальное расстояние от точки до отрезка
    fn distance_to(&self, px: f64, pz: f64) -> f64 {
        let dx = self.x1 - self.x0;
        let dz = self.z1 - self.z0;
        let len_sq = dx * dx + dz * dz;

        if len_sq < 1e-10 {
            let ex = px - self.x0;
            let ez = pz - self.z0;
            return (ex * ex + ez * ez).sqrt();
        }

        let t = ((px - self.x0) * dx + (pz - self.z0) * dz) / len_sq;
        let t = t.clamp(0.0, 1.0);

        let cx = self.x0 + t * dx;
        let cz = self.z0 + t * dz;
        let ex = px - cx;
        let ez = pz - cz;
        (ex * ex + ez * ez).sqrt()
    }
}

struct IslandSkeleton {
    segments: Vec<Segment>,
}

impl IslandSkeleton {
    /// Генерирует хребет + рукава из seed
    fn generate(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut segments = Vec::new();

        // Главный хребет: ломаная через (0,0)
        let spine_angle: f64 = rng.gen::<f64>() * std::f64::consts::PI;
        let spine_points = generate_ridge(
            &mut rng,
            0.0, 0.0,
            spine_angle,
            SPINE_LENGTH,
            SPINE_SEGMENTS,
            SPINE_WOBBLE,
        );

        // Хребет идёт в обе стороны от центра
        let spine_points_back = generate_ridge(
            &mut rng,
            0.0, 0.0,
            spine_angle + std::f64::consts::PI,
            SPINE_LENGTH,
            SPINE_SEGMENTS,
            SPINE_WOBBLE,
        );

        add_segments(&spine_points, &mut segments);
        add_segments(&spine_points_back, &mut segments);

        // Все точки хребта для ответвлений
        let mut all_spine: Vec<(f64, f64)> = Vec::new();
        all_spine.extend_from_slice(&spine_points);
        all_spine.extend_from_slice(&spine_points_back);

        // Рукава от случайных точек хребта
        for _ in 0..ARM_COUNT {
            let idx = rng.gen_range(1..all_spine.len());
            let (ax, az) = all_spine[idx];

            let arm_angle = rng.gen::<f64>() * std::f64::consts::TAU;
            let arm_points = generate_ridge(
                &mut rng,
                ax, az,
                arm_angle,
                ARM_LENGTH,
                ARM_SEGMENTS,
                ARM_WOBBLE,
            );
            add_segments(&arm_points, &mut segments);
        }

        Self { segments }
    }

    /// Минимальное расстояние от точки до любого сегмента хребта
    fn distance_to(&self, px: f64, pz: f64) -> f64 {
        let mut min_dist = f64::MAX;
        for seg in &self.segments {
            let d = seg.distance_to(px, pz);
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }
}

/// Генерирует ломаную линию хребта/рукава
fn generate_ridge(
    rng: &mut SmallRng,
    start_x: f64, start_z: f64,
    angle: f64,
    length: f64,
    num_segments: usize,
    wobble: f64,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(num_segments + 1);
    points.push((start_x, start_z));

    let seg_length = length / num_segments as f64;
    let mut current_angle = angle;
    let mut cx = start_x;
    let mut cz = start_z;

    for _ in 0..num_segments {
        current_angle += (rng.gen::<f64>() - 0.5) * wobble / length * std::f64::consts::TAU;
        cx += current_angle.cos() * seg_length;
        cz += current_angle.sin() * seg_length;
        points.push((cx, cz));
    }

    points
}

/// Превращает набор точек в отрезки
fn add_segments(points: &[(f64, f64)], segments: &mut Vec<Segment>) {
    for w in points.windows(2) {
        segments.push(Segment {
            x0: w[0].0, z0: w[0].1,
            x1: w[1].0, z1: w[1].1,
        });
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

    if section_y_offset > base_y + BORDER_HEIGHT + 2 {
        return section_data;
    }
    if section_y_offset + CHUNK_SIZE as i32 <= base_y - 1 {
        return section_data;
    }

    let skeleton = IslandSkeleton::generate(seed);

    let chunk_wx = chunk_position.x as f64 * CHUNK_SIZE as f64;
    let chunk_wz = chunk_position.z as f64 * CHUNK_SIZE as f64;

    // Voronoi-точки для всего острова
    let extent = SPINE_LENGTH + ARM_LENGTH + LAND_WIDTH;
    let points = generate_voronoi_points(
        seed, -extent, -extent, extent, extent,
    );

    if points.len() < 3 {
        return section_data;
    }

    // Зоны по расстоянию от скелета
    let zones: Vec<CellZone> = points.iter().map(|p| {
        let dist = skeleton.distance_to(p.x, p.y);
        if dist > LAND_WIDTH {
            CellZone::Ocean
        } else if dist > LAND_WIDTH - 16.0 {
            CellZone::Beach
        } else if dist > HIGHLAND_WIDTH {
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

            let zone = zones[nearest_idx];

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

            let top_y = if is_border {
                base_y + BORDER_HEIGHT
            } else {
                base_y
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
        CellZone::Mountain => BlockID::Stone.id(),
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
