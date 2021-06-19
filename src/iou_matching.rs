use crate::*;

use ndarray::*;

pub fn iou(bbox: &Array1<f32>, candidates: &Array2<f32>) -> Array1<f32> {
    let bbox_tl = bbox.slice(s![..2]).to_owned();
    let bbox_br = &bbox_tl + bbox.slice(s![2..4]).to_owned();
    let candidates_tl = candidates.slice(s![.., 0..2]).to_owned();
    let candidates_br = &candidates_tl + candidates.slice(s![.., 2..4]).to_owned();

    let tl = stack!(
        Axis(1),
        candidates_tl.slice(s![.., 0]).mapv(|v| v.max(bbox_tl[0])),
        candidates_tl.slice(s![.., 1]).mapv(|v| v.max(bbox_tl[1]))
    );
    let br = stack!(
        Axis(1),
        candidates_br.slice(s![.., 0]).mapv(|v| v.min(bbox_br[0])),
        candidates_br.slice(s![.., 1]).mapv(|v| v.min(bbox_br[1]))
    );
    let wh = (br - tl).mapv(|v| v.clamp(0.0, f32::MAX));

    let area_intersection = wh.map_axis(Axis(1), |v| v[0] * v[1]);
    let area_bbox = bbox.map_axis(Axis(0), |v| v[2] * v[3]);
    let area_candidates = candidates.map_axis(Axis(1), |v| v[2] * v[3]);

    &area_intersection / (&area_bbox + &area_candidates - &area_intersection)
}

pub fn iou_cost(
    tracks: Vec<Track>,
    detections: Vec<Detection>,
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> Array2<f32> {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(|| (0..detections.len()).collect());

    let mut cost_matrix = Array2::<f32>::zeros((0, detection_indices.len()));
    track_indices.iter().for_each(|track_idx| {
        let track = tracks.get(*track_idx).unwrap();

        if track.time_since_update > 1 {
            cost_matrix
                .push_row(Array1::from_elem(detection_indices.len(), f32::MAX).view())
                .unwrap();
        } else {
            let bbox = track.to_tlwh();
            let mut candidates = Array2::<f32>::zeros((0, 4));
            detection_indices.iter().for_each(|detection_idx| {
                candidates
                    .push(Axis(0), detections.get(*detection_idx).unwrap().tlwh.view())
                    .unwrap()
            });

            cost_matrix
                .push_row((1.0 - iou(&bbox, &candidates)).view())
                .unwrap();
        }
    });

    cost_matrix
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::*;

    #[test]
    fn iou() {
        let iou = iou_matching::iou(
            &arr1::<f32>(&[4.0, 5.0, 6.0, 7.0]),
            &arr2::<f32, _>(&[
                [3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0],
                [6.0, 7.0, 8.0, 9.0],
            ]),
        );
        assert_eq!(iou, arr1::<f32>(&[0.3846154, 0.0, 0.21276596]));
    }

    #[test]
    fn iou_cost() {
        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[4.0, 5.0, 6.0, 7.0]));
        let track0 = Track::new(mean, covariance, 0, 0, 0, None);

        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.clone().initiate(&arr1::<f32>(&[2.0, 3.0, 4.0, 5.0]));
        let track1 = Track::new(mean, covariance, 0, 0, 0, None);

        let detection0 = Detection::new(arr1::<f32>(&[3.0, 4.0, 5.0, 6.0]), 1.0, arr1::<f32>(&[]));
        let detection1 = Detection::new(arr1::<f32>(&[1.0, 2.0, 3.0, 4.0]), 1.0, arr1::<f32>(&[]));
        let detection2 = Detection::new(arr1::<f32>(&[6.0, 7.0, 8.0, 9.0]), 1.0, arr1::<f32>(&[]));

        let cost_matrix = iou_matching::iou_cost(
            vec![track0, track1],
            vec![detection0, detection1, detection2],
            None,
            None,
        );

        assert_eq!(
            cost_matrix,
            arr2::<f32, _>(&[
                [0.92537314, 0.9591837, 0.9661017],
                [0.93877554, 0.8965517, 1.0]
            ])
        );
    }
}
