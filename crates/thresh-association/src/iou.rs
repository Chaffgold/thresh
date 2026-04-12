//! IoU (Intersection over Union) computation for bounding boxes.

/// Compute 2D IoU between two axis-aligned bounding boxes.
///
/// Each box is (x_min, y_min, x_max, y_max).
pub fn iou_2d(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x_overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let y_overlap = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = x_overlap * y_overlap;

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// 3D bounding box for IoU computation.
#[derive(Debug, Clone, Copy)]
pub struct Box3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub length: f64,
    pub width: f64,
    pub height: f64,
    pub yaw: f64,
}

/// Compute approximate 3D IoU between two oriented bounding boxes.
///
/// Uses BEV (Bird's Eye View) rotated rectangle intersection for the x-y plane,
/// multiplied by the z-axis overlap fraction. This is the standard approximation
/// used in nuScenes and KITTI evaluation.
pub fn iou_3d(a: &Box3D, b: &Box3D) -> f64 {
    // Get the 4 corners of each box in BEV
    let corners_a = rotated_corners(a);
    let corners_b = rotated_corners(b);

    // Compute BEV intersection area via Sutherland-Hodgman polygon clipping
    let intersection_area = polygon_intersection_area(&corners_a, &corners_b);

    let area_a = a.length * a.width;
    let area_b = b.length * b.width;
    let bev_union = area_a + area_b - intersection_area;

    if bev_union <= 1e-15 {
        return 0.0;
    }

    // Z-axis overlap
    let a_z_min = a.z - a.height / 2.0;
    let a_z_max = a.z + a.height / 2.0;
    let b_z_min = b.z - b.height / 2.0;
    let b_z_max = b.z + b.height / 2.0;

    let z_overlap = (a_z_max.min(b_z_max) - a_z_min.max(b_z_min)).max(0.0);

    // 3D IoU = intersection volume / union volume
    let vol_intersection = intersection_area * z_overlap;
    let vol_a = area_a * a.height;
    let vol_b = area_b * b.height;
    let vol_union = vol_a + vol_b - vol_intersection;

    if vol_union <= 1e-15 {
        0.0
    } else {
        vol_intersection / vol_union
    }
}

/// Compute the 4 BEV corners of a rotated box.
fn rotated_corners(b: &Box3D) -> [(f64, f64); 4] {
    let cos_y = b.yaw.cos();
    let sin_y = b.yaw.sin();
    let hl = b.length / 2.0;
    let hw = b.width / 2.0;

    // Counter-clockwise winding
    let dx = [hl, -hl, -hl, hl];
    let dy = [hw, hw, -hw, -hw];

    let mut corners = [(0.0, 0.0); 4];
    for i in 0..4 {
        corners[i] = (
            b.x + dx[i] * cos_y - dy[i] * sin_y,
            b.y + dx[i] * sin_y + dy[i] * cos_y,
        );
    }
    corners
}

/// Sutherland-Hodgman polygon clipping: clip `subject` by `clip` polygon.
fn polygon_intersection_area(subject: &[(f64, f64); 4], clip: &[(f64, f64); 4]) -> f64 {
    let mut output: Vec<(f64, f64)> = subject.to_vec();

    for i in 0..4 {
        if output.is_empty() {
            return 0.0;
        }
        let input = output.clone();
        output.clear();

        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % 4];

        for j in 0..input.len() {
            let current = input[j];
            let previous = input[(j + input.len() - 1) % input.len()];

            let curr_inside = cross_2d(edge_start, edge_end, current) >= 0.0;
            let prev_inside = cross_2d(edge_start, edge_end, previous) >= 0.0;

            if curr_inside {
                if !prev_inside
                    && let Some(p) = line_intersection(edge_start, edge_end, previous, current)
                {
                    output.push(p);
                }
                output.push(current);
            } else if prev_inside
                && let Some(p) = line_intersection(edge_start, edge_end, previous, current)
            {
                output.push(p);
            }
        }
    }

    polygon_area(&output)
}

fn cross_2d(a: (f64, f64), b: (f64, f64), p: (f64, f64)) -> f64 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

fn line_intersection(
    a1: (f64, f64),
    a2: (f64, f64),
    b1: (f64, f64),
    b2: (f64, f64),
) -> Option<(f64, f64)> {
    let d1x = a2.0 - a1.0;
    let d1y = a2.1 - a1.1;
    let d2x = b2.0 - b1.0;
    let d2y = b2.1 - b1.1;

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-15 {
        return None;
    }

    let t = ((b1.0 - a1.0) * d2y - (b1.1 - a1.1) * d2x) / denom;
    Some((a1.0 + t * d1x, a1.1 + t * d1y))
}

fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].0 * vertices[j].1;
        area -= vertices[j].0 * vertices[i].1;
    }
    area.abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_boxes_iou_1() {
        let b = Box3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            length: 4.0,
            width: 2.0,
            height: 1.5,
            yaw: 0.0,
        };
        let iou = iou_3d(&b, &b);
        assert!(
            (iou - 1.0).abs() < 1e-6,
            "Expected IoU 1.0 for identical boxes, got {iou}"
        );
    }

    #[test]
    fn non_overlapping_iou_0() {
        let a = Box3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            length: 2.0,
            width: 2.0,
            height: 2.0,
            yaw: 0.0,
        };
        let b = Box3D {
            x: 100.0,
            y: 100.0,
            z: 100.0,
            length: 2.0,
            width: 2.0,
            height: 2.0,
            yaw: 0.0,
        };
        let iou = iou_3d(&a, &b);
        assert!(iou.abs() < 1e-10, "Expected IoU 0.0, got {iou}");
    }

    #[test]
    fn iou_2d_identical() {
        let b = [0.0, 0.0, 2.0, 2.0];
        assert!((iou_2d(b, b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn iou_2d_no_overlap() {
        let a = [0.0, 0.0, 1.0, 1.0];
        let b = [2.0, 2.0, 3.0, 3.0];
        assert!(iou_2d(a, b).abs() < 1e-10);
    }

    #[test]
    fn iou_2d_half_overlap() {
        let a = [0.0, 0.0, 2.0, 2.0];
        let b = [1.0, 0.0, 3.0, 2.0];
        // Intersection = 1*2 = 2, Union = 4+4-2 = 6
        let iou = iou_2d(a, b);
        assert!((iou - 2.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn iou_3d_z_no_overlap() {
        let a = Box3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            length: 2.0,
            width: 2.0,
            height: 2.0,
            yaw: 0.0,
        };
        let b = Box3D {
            x: 0.0,
            y: 0.0,
            z: 100.0,
            length: 2.0,
            width: 2.0,
            height: 2.0,
            yaw: 0.0,
        };
        let iou = iou_3d(&a, &b);
        assert!(iou.abs() < 1e-10);
    }
}
