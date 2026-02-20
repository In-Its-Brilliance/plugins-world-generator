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

// --- Производные параметры ---------------------------------------------------

struct IslandParams {
    island_size: f64,
    spine_length: f64,
    arm_length: f64,
    land_width: f64,
    highland_width: f64,
    max_height: f64,
    beach_field_threshold: f64,
}

impl IslandParams {
    fn from_settings(s: &GeneratorSettings) -> Self {
        let size = s.island.size;
        let spine_length = size * 0.5;
        let arm_length = size * 0.3;
        let land_width = size * 0.4;
        let max_height = (land_width * s.island.mountains.height_ratio)
            .min(s.island.mountains.max_peak_height as f64);
        Self {
            island_size: size,
            spine_length,
            arm_length,
            land_width,
            highland_width: land_width * 0.42,
            max_height,
            beach_field_threshold: s.island.terrain.beach_width / land_width,
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
    fn nearest_point(&self, px: f64, pz: f64) -> (f64, f64, f64, f64) {
        let mut best_dist = f64::MAX;
        let mut best_h = 0.0;
        let mut best_cx = 0.0;
        let mut best_cz = 0.0;

        for w in self.points.windows(2) {
            let (d, h, cx, cz) = project_on_segment(
                px, pz,
                w[0].x, w[0].z, w[0].h,
                w[1].x, w[1].z, w[1].h,
            );
            if d < best_dist {
                best_dist = d;
                best_h = h;
                best_cx = cx;
                best_cz = cz;
            }
        }

        (best_dist, best_h, best_cx, best_cz)
    }
}

fn project_on_segment(
    px: f64, pz: f64,
    x0: f64, z0: f64, h0: f64,
    x1: f64, z1: f64, h1: f64,
) -> (f64, f64, f64, f64) {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let len_sq = dx * dx + dz * dz;

    if len_sq < 1e-10 {
        let ex = px - x0;
        let ez = pz - z0;
        return ((ex * ex + ez * ez).sqrt(), h0, x0, z0);
    }

    let t = ((px - x0) * dx + (pz - z0) * dz) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let cx = x0 + t * dx;
    let cz = z0 + t * dz;
    let ex = px - cx;
    let ez = pz - cz;

    ((ex * ex + ez * ez).sqrt(), h0 + t * (h1 - h0), cx, cz)
}

struct IslandSkeleton {
    polylines: Vec<Polyline>,
}

impl IslandSkeleton {
    fn generate(seed: u64, params: &IslandParams, settings: &GeneratorSettings) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut polylines = Vec::new();

        let offset = settings.island.center_offset;
        let ox = (rng.gen::<f64>() - 0.5) * offset * 2.0;
        let oz = (rng.gen::<f64>() - 0.5) * offset * 2.0;

        let island_size = params.spine_length + params.arm_length;
        let mtn = &settings.island.mountains;
        let up_chance = (mtn.base_up_chance
            + (island_size / 2000.0) * (mtn.max_up_chance - mtn.base_up_chance))
            .min(mtn.max_up_chance);

        let ridge = &settings.island.ridge;
        let spine_angle: f64 = rng.gen::<f64>() * std::f64::consts::PI;

        let spine_fwd = generate_ridge(
            &mut rng,
            ox, oz,
            params.max_height,
            spine_angle,
            params.spine_length,
            ridge.spine_segments,
            ridge.spine_wobble,
            up_chance,
            params.max_height,
            mtn.height_step,
        );

        let spine_back = generate_ridge(
            &mut rng,
            ox, oz,
            params.max_height,
            spine_angle + std::f64::consts::PI,
            params.spine_length,
            ridge.spine_segments,
            ridge.spine_wobble,
            up_chance,
            params.max_height,
            mtn.height_step,
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

        for _ in 0..ridge.arm_count {
            let idx = rng.gen_range(1..all_spine.len());
            let (ax, az, ah) = all_spine[idx];

            let arm_angle = rng.gen::<f64>() * std::f64::consts::TAU;
            let arm_raw = generate_ridge(
                &mut rng,
                ax, az,
                ah * 0.7,
                arm_angle,
                params.arm_length,
                ridge.arm_segments,
                ridge.arm_wobble,
                up_chance,
                params.max_height,
                mtn.height_step,
            );

            let arm_points: Vec<RidgePoint> = arm_raw.iter()
                .map(|&(x, z, h)| RidgePoint { x, z, h })
                .collect();

            polylines.push(Polyline { points: arm_points });
        }

        Self { polylines }
    }

    fn query_elevation(&self, px: f64, pz: f64, params: &IslandParams) -> (f64, f64, f64, f64) {
        let mut min_dist = f64::MAX;
        let mut best_elev = 0.0;
        let mut ridge_cx = 0.0;
        let mut ridge_cz = 0.0;

        for poly in &self.polylines {
            let (d, h, cx, cz) = poly.nearest_point(px, pz);

            if d < min_dist {
                min_dist = d;
                ridge_cx = cx;
                ridge_cz = cz;
            }

            let height_ratio = h / params.max_height;
            let base = params.island_size * 0.05;
            let extra = height_ratio * height_ratio * params.island_size * 0.07;
            let dynamic_falloff = base + extra;

            let falloff = (-d / dynamic_falloff).exp();
            let elev = falloff * h;
            if elev > best_elev {
                best_elev = elev;
            }
        }

        (min_dist, best_elev, ridge_cx, ridge_cz)
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
    height_step: f64,
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
            h += height_step;
        } else {
            h -= height_step;
        }

        if i >= num_segments - 3 {
            h -= height_step;
        }

        h = h.clamp(0.0, max_height);
        points.push((cx, cz, h));
    }

    points
}

// --- Когерентный noise -------------------------------------------------------

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
    settings: &GeneratorSettings,
) -> f64 {
    let ridge_field = ((land_width - raw_dist) / land_width).clamp(0.0, 1.0);

    let coast = &settings.island.coastline;
    let noise_field = fbm_noise(
        x * coast.fractal,
        z * coast.fractal,
        seed.wrapping_add(7777),
        6,
    );

    ridge_field + noise_field * coast.spread
}

fn distance_to_shore(field: f64, land_width: f64) -> f64 {
    field * land_width
}

/// Информация о дереве для размещения.
struct TreeInstance {
    /// Мировая координата X основания ствола.
    wx: i32,
    /// Мировая координата Z основания ствола.
    wz: i32,
    /// Мировая координата Y основания ствола (top_y + 1).
    wy: i32,
    /// Seed для генерации формы этого дерева.
    tree_seed: u64,
}

/// Блок дерева относительно основания ствола.
struct TreeBlock {
    dx: i32,
    dy: i32,
    dz: i32,
    block_id: u16,
}

/// Генерирует форму ели: ствол + конусообразная крона с ярусами.
fn generate_tree(tree_seed: u64) -> Vec<TreeBlock> {
    let mut rng = SmallRng::seed_from_u64(tree_seed);
    let mut blocks = Vec::new();

    let trunk_height = 6 + (rng.gen::<f64>() * 6.0) as i32;
    let crown_start = 2 + (rng.gen::<f64>() * 2.0) as i32;
    let crown_top = trunk_height + 2;
    let max_radius = 3 + (rng.gen::<f64>() * 2.0) as i32;

    // Ствол
    for y in 0..trunk_height {
        blocks.push(TreeBlock {
            dx: 0, dy: y, dz: 0,
            block_id: BlockID::SpruceLog.id(),
        });
    }

    // Верхушка -- 1-2 блока листвы над стволом
    for y in trunk_height..=crown_top {
        blocks.push(TreeBlock {
            dx: 0, dy: y, dz: 0,
            block_id: BlockID::SpruceLeaves.id(),
        });
    }

    // Ярусы кроны снизу вверх
    let crown_height = crown_top - crown_start;
    let mut y = crown_start;
    while y <= trunk_height {
        let t = (y - crown_start) as f64 / crown_height as f64;
        // Радиус: максимальный внизу, сужается к верху
        let r = ((1.0 - t) * max_radius as f64 + 0.5) as i32;
        if r < 1 {
            y += 1;
            continue;
        }

        for dx in -r..=r {
            for dz in -r..=r {
                let dist_sq = dx * dx + dz * dz;
                if dist_sq > r * r {
                    continue;
                }
                if dx == 0 && dz == 0 {
                    continue;
                }
                // Случайные дыры на краях
                let hole_hash = tree_seed
                    .wrapping_add(dx as u64 * 73856093)
                    .wrapping_add(y as u64 * 19349663)
                    .wrapping_add(dz as u64 * 83492791);
                let edge = dist_sq as f64 / (r * r) as f64;
                if edge > 0.7 && (hole_hash % 10) < 3 {
                    continue;
                }
                blocks.push(TreeBlock {
                    dx, dy: y, dz,
                    block_id: BlockID::SpruceLeaves.id(),
                });
            }
        }

        // Пробел между ярусами через каждые 2 слоя
        y += if r > 2 { 2 } else { 1 };
    }

    blocks
}

/// Собирает все деревья, которые могут повлиять на данный чанк.
/// Проверяет ячейки grid в радиусе MAX_TREE_REACH от границ чанка.
fn collect_trees_near_chunk(
    chunk_wx: f64,
    chunk_wz: f64,
    seed: u64,
    skeleton: &IslandSkeleton,
    params: &IslandParams,
    settings: &GeneratorSettings,
    base_y: i32,
) -> Vec<TreeInstance> {
    let cell_size = 4.0;
    let reach = 10.0;
    let chunk_size = CHUNK_SIZE as f64;

    let min_x = chunk_wx - reach;
    let min_z = chunk_wz - reach;
    let max_x = chunk_wx + chunk_size + reach;
    let max_z = chunk_wz + chunk_size + reach;

    let gx0 = (min_x / cell_size).floor() as i64;
    let gz0 = (min_z / cell_size).floor() as i64;
    let gx1 = (max_x / cell_size).ceil() as i64;
    let gz1 = (max_z / cell_size).ceil() as i64;

    let mtn = &settings.island.mountains;
    let mut trees = Vec::new();

    for gx in gx0..=gx1 {
        for gz in gz0..=gz1 {
            let cx = gx as f64 * cell_size + cell_size * 0.5;
            let cz = gz as f64 * cell_size + cell_size * 0.5;
            let forest_noise = fbm_noise(
                cx * 0.025,
                cz * 0.025,
                seed.wrapping_add(33333),
                2,
            );
            if forest_noise < 0.15 {
                continue;
            }
            let density = ((forest_noise - 0.15) / 0.25).min(1.0) * 0.95;

            let cell_seed = seed.wrapping_add(12345)
                ^ (gx as u64).wrapping_mul(6364136223846793005)
                ^ (gz as u64).wrapping_mul(1442695040888963407);
            let mut rng = SmallRng::seed_from_u64(cell_seed);

            if rng.gen::<f64>() > density {
                continue;
            }

            let tree_x = (gx as f64 * cell_size + 1.0 + rng.gen::<f64>() * (cell_size - 2.0)).floor();
            let tree_z = (gz as f64 * cell_size + 1.0 + rng.gen::<f64>() * (cell_size - 2.0)).floor();

            // Проверка: дерево на суше, плоское место
            let (raw_dist, _, _, _) = skeleton.query_elevation(tree_x, tree_z, params);
            let field = island_field(tree_x, tree_z, raw_dist, params.land_width, seed, settings);
            if field <= params.beach_field_threshold {
                continue;
            }

            let elevation_f = compute_elevation(tree_x, tree_z, skeleton, params, seed, settings);
            let elev_px = compute_elevation(tree_x + 1.0, tree_z, skeleton, params, seed, settings);
            let elev_pz = compute_elevation(tree_x, tree_z + 1.0, skeleton, params, seed, settings);
            let slope = ((elev_px - elevation_f).powi(2) + (elev_pz - elevation_f).powi(2)).sqrt();
            if slope > 0.3 {
                continue;
            }

            let zone = classify_by_field(
                tree_x, tree_z, field, skeleton, params, mtn.width,
            );
            if zone != CellZone::Lowland {
                continue;
            }

            trees.push(TreeInstance {
                wx: tree_x as i32,
                wz: tree_z as i32,
                wy: base_y + elevation_f as i32 + 1,
                tree_seed: cell_seed.wrapping_add(99999),
            });
        }
    }

    trees
}

/// Расстояние до ближайшего дерева (для затемнения травы).
/// Проверяет 9 ячеек вокруг точки.
fn distance_to_nearest_tree(wx: f64, wz: f64, seed: u64) -> f64 {
    let cell_size = 4.0;
    let gx = (wx / cell_size).floor() as i64;
    let gz = (wz / cell_size).floor() as i64;

    let mut min_dist_sq = f64::MAX;

    for dg_x in -1..=1 {
        for dg_z in -1..=1 {
            let cgx = gx + dg_x;
            let cgz = gz + dg_z;

            // Проверка forest noise для этой ячейки
            let cx = cgx as f64 * cell_size + cell_size * 0.5;
            let cz = cgz as f64 * cell_size + cell_size * 0.5;
            let forest_noise = fbm_noise(
                cx * 0.025,
                cz * 0.025,
                seed.wrapping_add(33333),
                2,
            );
            if forest_noise < 0.15 {
                continue;
            }
            let density = ((forest_noise - 0.15) / 0.25).min(1.0) * 0.95;

            let cell_seed = seed.wrapping_add(12345)
                ^ (cgx as u64).wrapping_mul(6364136223846793005)
                ^ (cgz as u64).wrapping_mul(1442695040888963407);
            let mut rng = SmallRng::seed_from_u64(cell_seed);

            if rng.gen::<f64>() > density {
                continue;
            }

            let tree_x = (cgx as f64 * cell_size + 1.0 + rng.gen::<f64>() * (cell_size - 2.0)).floor() + 0.5;
            let tree_z = (cgz as f64 * cell_size + 1.0 + rng.gen::<f64>() * (cell_size - 2.0)).floor() + 0.5;

            let dx = wx - tree_x;
            let dz = wz - tree_z;
            let d = dx * dx + dz * dz;
            if d < min_dist_sq {
                min_dist_sq = d;
            }
        }
    }

    min_dist_sq.sqrt()
}

fn compute_elevation(
    wx: f64, wz: f64,
    skeleton: &IslandSkeleton,
    params: &IslandParams,
    seed: u64,
    settings: &GeneratorSettings,
) -> f64 {
    let (raw_dist, ridge_elev, ridge_cx, ridge_cz) = skeleton.query_elevation(wx, wz, params);
    let field = island_field(wx, wz, raw_dist, params.land_width, seed, settings);

    let mtn = &settings.island.mountains;
    let shore = &settings.island.shore;

    let base = if field > 0.0 {
        let shore_factor = field.min(1.0);
        ridge_elev * shore_factor
    } else {
        field * mtn.max_ocean_depth
    };

    let shore_dist = distance_to_shore(field, params.land_width);
    let abs_dist = shore_dist.abs().min(shore.mod_radius);
    let t = abs_dist / shore.mod_radius;
    let ease = t * (2.0 - t);
    let modifier = if shore_dist > 0.0 {
        ease * shore.mod_up
    } else {
        -ease * shore.mod_down
    };

    let raw_noise = fbm_noise(
        wx * 0.012,
        wz * 0.012,
        seed.wrapping_add(55555),
        4,
    );
    let hills = raw_noise.max(0.0) * raw_noise.max(0.0) * 30.0;

    let detail_noise = fbm_noise(
        wx * 0.025,
        wz * 0.025,
        seed.wrapping_add(88888),
        2,
    );
    let detail = detail_noise.max(0.0) * 4.0;

    let shore_fade = if field <= params.beach_field_threshold {
        0.0
    } else {
        ((field - params.beach_field_threshold) * 8.0).min(1.0)
    };

    let surface_mod = (hills + detail) * shore_fade;

    let total = base + modifier + surface_mod;

    // --- Эрозионные борозды на горах ---
    // Радиальные канавки, расходящиеся от хребта как кулуары
    let erosion = if total > 6.0 {
        let to_x = wx - ridge_cx;
        let to_z = wz - ridge_cz;
        let dist_from_ridge = (to_x * to_x + to_z * to_z).sqrt().max(0.1);

        // Угол от хребта - определяет "спицу"
        let angle = to_z.atan2(to_x);

        // Noise по углу создаёт борозды-кулуары
        let angular_coord = angle * 12.0;
        let radial_coord = dist_from_ridge * 0.015;

        let groove_noise = fbm_noise(
            angular_coord,
            radial_coord,
            seed.wrapping_add(77777),
            3,
        );

        // Второй слой - мелкие борозды
        let detail_groove = fbm_noise(
            angular_coord * 2.5,
            radial_coord * 3.0,
            seed.wrapping_add(78888),
            2,
        );

        let groove = (-groove_noise).max(0.0) + (-detail_groove).max(0.0) * 0.4;

        // Сила зависит от высоты
        let height_factor = ((total - 6.0) / 34.0).clamp(0.0, 1.0);

        // Сильнее на склонах, слабее у самой вершины и у подножия
        let dist_norm = (dist_from_ridge / params.highland_width).clamp(0.0, 1.0);
        let slope_factor = (dist_norm * 3.0).min(1.0) * (1.0 - dist_norm * 0.5);

        let raw_erosion = groove * height_factor * slope_factor * total * 0.35;

        raw_erosion.min((total - 3.0).max(0.0))
    } else {
        0.0
    };

    total - erosion
}

/// Сила эрозии в точке (0.0 .. ~1.0). Используется и для высоты, и для выбора блока.
fn erosion_strength(
    wx: f64, wz: f64,
    raw_dist: f64,
    ridge_elev: f64,
    ridge_cx: f64, ridge_cz: f64,
    params: &IslandParams,
    seed: u64,
) -> f64 {
    if ridge_elev <= 15.0 || raw_dist >= params.highland_width {
        return 0.0;
    }

    let to_ridge_x = ridge_cx - wx;
    let to_ridge_z = ridge_cz - wz;
    let to_ridge_len = (to_ridge_x * to_ridge_x + to_ridge_z * to_ridge_z)
        .sqrt()
        .max(1.0);

    let dir_x = to_ridge_x / to_ridge_len;
    let dir_z = to_ridge_z / to_ridge_len;

    let along = wx * dir_x + wz * dir_z;
    let across = wx * (-dir_z) + wz * dir_x;

    let erosion_noise = fbm_noise(
        along * 0.01,
        across * 0.08,
        seed.wrapping_add(77777),
        4,
    );

    let groove = (-erosion_noise).max(0.0);

    let height_factor = ((ridge_elev - 15.0) / 30.0).clamp(0.0, 1.0);
    let dist_factor = (1.0 - raw_dist / params.highland_width).clamp(0.0, 1.0);

    groove * groove * height_factor * dist_factor
}

fn classify_by_field(
    wx: f64, wz: f64,
    field: f64,
    skeleton: &IslandSkeleton,
    params: &IslandParams,
    mountain_width: f64,
) -> CellZone {
    if field <= 0.0 {
        return CellZone::Ocean;
    }

    if field < params.beach_field_threshold {
        return CellZone::Beach;
    }

    let (raw_dist, _, _, _) = skeleton.query_elevation(wx, wz, params);
    if raw_dist <= mountain_width {
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

    let mtn = &settings.island.mountains;
    let shore = &settings.island.shore;
    let voronoi = &settings.island.voronoi;
    let noise_amp = settings.island.terrain.surface_noise_amplitude as i32;

    let max_depth = mtn.max_ocean_depth as i32 + shore.mod_down as i32;
    let max_tree_height = 22;
    if section_y_offset > base_y + mtn.max_peak_height + shore.mod_up as i32 + noise_amp + voronoi.border_height + max_tree_height {
        return section_data;
    }
    if section_y_offset + CHUNK_SIZE as i32 <= base_y - max_depth - noise_amp - 4 {
        return section_data;
    }

    let params = IslandParams::from_settings(settings);
    let skeleton = IslandSkeleton::generate(seed, &params, settings);

    let chunk_wx = chunk_position.x as f64 * CHUNK_SIZE as f64;
    let chunk_wz = chunk_position.z as f64 * CHUNK_SIZE as f64;

    let max_noise_reach = settings.island.coastline.spread * params.land_width;
    let extent = params.spine_length + params.arm_length
        + params.land_width + max_noise_reach;

    let show_borders = settings.island.voronoi_borders;
    let points = if show_borders {
        generate_voronoi_points(seed, -extent, -extent, extent, extent, voronoi.cell_size)
    } else {
        Vec::new()
    };

    if show_borders && points.len() < 3 {
        return section_data;
    }

    for x in 0..(CHUNK_SIZE as u8) {
        for z in 0..(CHUNK_SIZE as u8) {
            let wx = chunk_wx + x as f64;
            let wz = chunk_wz + z as f64;

            let (raw_dist, ridge_elev_val, ridge_cx, ridge_cz) = skeleton.query_elevation(wx, wz, &params);
            let field = island_field(wx, wz, raw_dist, params.land_width, seed, settings);
            let elevation_f = compute_elevation(wx, wz, &skeleton, &params, seed, settings);
            let elevation = elevation_f as i32;
            let is_underwater = field <= 0.0;

            let elev_px = compute_elevation(wx + 1.0, wz, &skeleton, &params, seed, settings);
            let elev_pz = compute_elevation(wx, wz + 1.0, &skeleton, &params, seed, settings);
            let slope = ((elev_px - elevation_f).powi(2) + (elev_pz - elevation_f).powi(2)).sqrt();

            let zone = classify_by_field(
                wx, wz, field, &skeleton, &params, mtn.width,
            );

            // Сила эрозии для выбора блока поверхности
            let e_str = erosion_strength(
                wx, wz, raw_dist, ridge_elev_val,
                ridge_cx, ridge_cz, &params, seed,
            );
            let in_erosion_groove = e_str > 0.15;

            // Цвет травы: noise + затемнение около деревьев
            let color_noise = fbm_noise(
                wx * 0.02,
                wz * 0.02,
                seed.wrapping_add(44444),
                3,
            );
            let base_color = (color_noise + 0.5) * 8.0;

            // Затемнение травы рядом с деревьями
            let tree_dist = distance_to_nearest_tree(wx, wz, seed);
            let tree_darken = if tree_dist < 5.0 {
                (1.0 - tree_dist / 5.0) * 4.0
            } else {
                0.0
            };
            let grass_color = (base_color + tree_darken).clamp(0.0, 7.0) as u8;

            let is_border = if show_borders {
                let (nearest, second, _) = find_two_nearest_with_index(&points, wx, wz);
                let dist_to_border = (second - nearest) * 0.5;
                dist_to_border < voronoi.border_thickness
                    && !is_underwater
                    && zone != CellZone::Beach
                    && zone != CellZone::Ocean
            } else {
                false
            };

            let top_y = if is_border {
                base_y + elevation + voronoi.border_height
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
                            BlockDataInfo::create(border_block(zone, slope)),
                        );
                    } else if in_erosion_groove {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(BlockID::Cobblestone.id()),
                        );
                    } else {
                        let (block_id, color) = surface_block(zone, slope, grass_color);
                        let mut block = BlockDataInfo::create(block_id);
                        if let Some(c) = color {
                            block = block.color(c);
                        }
                        section_data.insert(&pos, block);
                    }
                } else if world_y >= top_y - 3 {
                    if is_underwater {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(BlockID::Sandstone.id()),
                        );
                    } else if in_erosion_groove {
                        section_data.insert(
                            &pos,
                            BlockDataInfo::create(BlockID::Cobblestone.id()),
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

    // --- Деревья: собираем все в радиусе, генерируем блоки ---
    let trees = collect_trees_near_chunk(
        chunk_wx, chunk_wz, seed,
        &skeleton, &params, settings, base_y,
    );

    let chunk_x0 = chunk_wx as i32;
    let chunk_z0 = chunk_wz as i32;
    let chunk_x1 = chunk_x0 + CHUNK_SIZE as i32;
    let chunk_z1 = chunk_z0 + CHUNK_SIZE as i32;
    let section_y_min = section_y_offset;
    let section_y_max = section_y_offset + CHUNK_SIZE as i32;

    for tree in &trees {
        let tree_blocks = generate_tree(tree.tree_seed);
        for tb in &tree_blocks {
            let bx = tree.wx + tb.dx;
            let by = tree.wy + tb.dy;
            let bz = tree.wz + tb.dz;

            if bx < chunk_x0 || bx >= chunk_x1 { continue; }
            if bz < chunk_z0 || bz >= chunk_z1 { continue; }
            if by < section_y_min || by >= section_y_max { continue; }

            let lx = (bx - chunk_x0) as u8;
            let ly = (by - section_y_min) as u8;
            let lz = (bz - chunk_z0) as u8;

            let pos = ChunkBlockPosition::new(lx, ly, lz);
            section_data.insert(
                &pos,
                BlockDataInfo::create(tb.block_id),
            );
        }
    }

    // --- Фолидж: цветы на полях, трава/мох в лесах ---
    for x in 0..(CHUNK_SIZE as u8) {
        for z in 0..(CHUNK_SIZE as u8) {
            let wx = chunk_wx + x as f64;
            let wz = chunk_wz + z as f64;

            let (raw_dist, _, _, _) = skeleton.query_elevation(wx, wz, &params);
            let field = island_field(wx, wz, raw_dist, params.land_width, seed, settings);
            if field <= params.beach_field_threshold {
                continue;
            }

            let elevation_f = compute_elevation(wx, wz, &skeleton, &params, seed, settings);
            let elevation = elevation_f as i32;
            let surface_y = base_y + elevation + 1;

            if surface_y < section_y_min || surface_y >= section_y_max {
                continue;
            }

            let zone = classify_by_field(
                wx, wz, field, &skeleton, &params, mtn.width,
            );

            let elev_px = compute_elevation(wx + 1.0, wz, &skeleton, &params, seed, settings);
            let elev_pz = compute_elevation(wx, wz + 1.0, &skeleton, &params, seed, settings);
            let slope = ((elev_px - elevation_f).powi(2) + (elev_pz - elevation_f).powi(2)).sqrt();

            // Фолидж только на траве
            let (surface_id, _) = surface_block(zone, slope, 0);
            if surface_id != BlockID::Grass.id() {
                continue;
            }

            // Хэш для рандома на блок
            let iwx = wx.floor() as i64;
            let iwz = wz.floor() as i64;
            let cell_hash = seed.wrapping_add(55555)
                ^ (iwx as u64).wrapping_mul(6364136223846793005)
                ^ (iwz as u64).wrapping_mul(1442695040888963407);
            let roll = (cell_hash % 1000) as f64 / 1000.0;

            let forest_noise = fbm_noise(
                wx * 0.025,
                wz * 0.025,
                seed.wrapping_add(33333),
                2,
            );
            let in_forest = forest_noise > 0.15 && zone == CellZone::Lowland;

            let foliage_id = if zone != CellZone::Lowland {
                // Highland/Mountain: просто трава на траве
                if roll > 0.12 { continue; }
                match cell_hash % 4 {
                    0 => BlockID::Grass1.id(),
                    1 => BlockID::Grass2.id(),
                    2 => BlockID::Grass3.id(),
                    _ => BlockID::Grass4.id(),
                }
            } else if in_forest {
                // Лес: только трава, редко
                let forest_depth = ((forest_noise - 0.15) / 0.35).clamp(0.0, 1.0);
                if roll > forest_depth * 0.15 { continue; }
                match cell_hash % 4 {
                    0 => BlockID::Grass1.id(),
                    1 => BlockID::Grass2.id(),
                    2 => BlockID::Grass3.id(),
                    _ => BlockID::Grass4.id(),
                }
            } else {
                // Поле: отдельные поляны цветов среди травы
                let field_depth = ((0.15 - forest_noise) / 0.4).clamp(0.0, 1.0);

                // Noise для цветочных полян (крупные пятна)
                let meadow_noise = fbm_noise(
                    wx * 0.015,
                    wz * 0.015,
                    seed.wrapping_add(66666),
                    2,
                );
                let is_meadow = meadow_noise > 0.15;

                if is_meadow {
                    // Цветочная поляна -- один вид на всю поляну
                    let meadow_density = ((meadow_noise - 0.15) / 0.3).clamp(0.0, 1.0);
                    if roll > meadow_density * field_depth * 0.3 { continue; }

                    // Вид цветка определяется очень крупным noise
                    let flower_type = fbm_noise(
                        wx * 0.008,
                        wz * 0.008,
                        seed.wrapping_add(88888),
                        1,
                    );
                    let fi = ((flower_type + 0.5) * 5.0).clamp(0.0, 4.99) as u32;
                    match fi {
                        0 => BlockID::FlowerWhite.id(),
                        1 => BlockID::FlowerYellow.id(),
                        2 => BlockID::FlowerRose.id(),
                        3 => BlockID::FlowerLupin.id(),
                        _ => BlockID::FlowerOrchid.id(),
                    }
                } else {
                    // Между полянами -- трава
                    if roll > field_depth * 0.1 { continue; }
                    match cell_hash % 4 {
                        0 => BlockID::Grass1.id(),
                        1 => BlockID::Grass2.id(),
                        2 => BlockID::TallGrass1.id(),
                        _ => BlockID::TallGrass2.id(),
                    }
                }
            };

            let ly = (surface_y - section_y_min) as u8;
            let pos = ChunkBlockPosition::new(x, ly, z);
            section_data.insert(
                &pos,
                BlockDataInfo::create(foliage_id),
            );
        }
    }

    section_data
}

fn surface_block(zone: CellZone, slope: f64, grass_color: u8) -> (u16, Option<u8>) {
    match zone {
        CellZone::Ocean => (BlockID::Water.id(), None),
        CellZone::Beach => (BlockID::Sand.id(), None),
        CellZone::Lowland => {
            if slope > 0.8 {
                (BlockID::CoarseDirt.id(), None)
            } else {
                (BlockID::Grass.id(), Some(grass_color))
            }
        }
        CellZone::Highland | CellZone::Mountain => {
            if slope > 1.2 {
                (BlockID::Stone.id(), None)
            } else if slope > 0.6 {
                (BlockID::Cobblestone.id(), None)
            } else if slope > 0.3 {
                (BlockID::CoarseDirt.id(), None)
            } else {
                (BlockID::Grass.id(), Some(grass_color))
            }
        }
    }
}

fn border_block(zone: CellZone, slope: f64) -> u16 {
    match zone {
        CellZone::Ocean => BlockID::Water.id(),
        CellZone::Beach => BlockID::Sandstone.id(),
        CellZone::Lowland => BlockID::CoarseDirt.id(),
        CellZone::Highland | CellZone::Mountain => {
            if slope > 0.6 {
                BlockID::Stone.id()
            } else {
                BlockID::Cobblestone.id()
            }
        }
    }
}

fn subsurface_block(zone: CellZone) -> u16 {
    match zone {
        CellZone::Ocean => BlockID::Stone.id(),
        CellZone::Beach => BlockID::Sandstone.id(),
        CellZone::Lowland => BlockID::CoarseDirt.id(),
        CellZone::Highland | CellZone::Mountain => BlockID::Stone.id(),
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
    cell_size: f64,
) -> Vec<Point> {
    let mut points = Vec::new();
    let step = cell_size;

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
