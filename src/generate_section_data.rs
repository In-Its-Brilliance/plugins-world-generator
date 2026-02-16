use common::{
    chunks::{
        block_position::ChunkBlockPosition,
        chunk_data::{BlockDataInfo, ChunkSectionData},
        chunk_position::ChunkPosition,
    },
    default_blocks_ids::BlockID,
    CHUNK_SIZE,
};
use delaunator::{triangulate, Point};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::generate_world_macro::MacroData;
use crate::settings::GeneratorSettings;

/// Средний размер Voronoi-ячейки в блоках (шаг jittered grid)
const VORONOI_CELL_SIZE: f64 = 16.0;

/// Порог расстояния до границы между ячейками (в блоках)
const BORDER_THICKNESS: f64 = 0.45;

/// Высота границы над поверхностью ячейки (в блоках)
const BORDER_HEIGHT: i32 = 1;

/// Voronoi-сетка: ячейки из травы, границы приподняты на BORDER_HEIGHT,
/// кусты в центрах ячеек. Точки детерминистичны из seed через jittered grid.
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

    let chunk_wx = chunk_position.x as f64 * CHUNK_SIZE as f64;
    let chunk_wz = chunk_position.z as f64 * CHUNK_SIZE as f64;

    let margin = VORONOI_CELL_SIZE * 2.0;
    let points = generate_voronoi_points(
        seed,
        chunk_wx - margin,
        chunk_wz - margin,
        chunk_wx + CHUNK_SIZE as f64 + margin,
        chunk_wz + CHUNK_SIZE as f64 + margin,
    );

    if points.len() < 3 {
        return section_data;
    }

    // Делоне-триангуляция -- понадобится для elevation/rivers
    let _triangulation = triangulate(&points);

    for x in 0..(CHUNK_SIZE as u8) {
        for z in 0..(CHUNK_SIZE as u8) {
            let wx = chunk_wx + x as f64;
            let wz = chunk_wz + z as f64;

            let (nearest, second) = find_two_nearest(&points, wx, wz);
            let dist_to_border = (second - nearest) * 0.5;
            let is_border = dist_to_border < BORDER_THICKNESS;

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
                        BlockID::Sand.id()
                    } else {
                        BlockID::Grass.id()
                    }
                } else if world_y >= top_y - 3 {
                    if is_border {
                        BlockID::Sandstone.id()
                    } else {
                        BlockID::CoarseDirt.id()
                    }
                } else {
                    BlockID::Stone.id()
                };

                let pos = ChunkBlockPosition::new(x, y, z);
                section_data.insert(&pos, BlockDataInfo::create(block_id));
            }

            // Куст в центре ячейки
            if !is_border && nearest < 1.5 {
                let bush_y = base_y + 1;
                if bush_y >= section_y_offset
                    && bush_y < section_y_offset + CHUNK_SIZE as i32
                {
                    let pos = ChunkBlockPosition::new(
                        x, (bush_y - section_y_offset) as u8, z,
                    );
                    section_data.insert(
                        &pos,
                        BlockDataInfo::create(BlockID::Andesite.id()),
                    );
                }
            }
        }
    }

    section_data
}

/// Jittered grid: одна точка на ячейку, seed из координат ячейки.
/// Соседние чанки генерируют те же точки в зоне перекрытия.
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

/// Расстояния до двух ближайших точек.
/// Разница определяет близость к границе Voronoi-ячеек.
fn find_two_nearest(points: &[Point], wx: f64, wz: f64) -> (f64, f64) {
    let mut nearest = f64::MAX;
    let mut second = f64::MAX;

    for p in points {
        let dx = wx - p.x;
        let dz = wz - p.y;
        let dist = dx * dx + dz * dz;

        if dist < nearest {
            second = nearest;
            nearest = dist;
        } else if dist < second {
            second = dist;
        }
    }

    (nearest.sqrt(), second.sqrt())
}
