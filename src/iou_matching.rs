use crate::*;

use ndarray::*;

/// Compute intersection over union.
///
/// # Parameters
///
/// - `bbox`: A bounding box in format `(top left x, top left y, width, height)`.
/// - `candidates`: A matrix of candidate bounding boxes (one per row) in the same format as `bbox`.
///
/// # Returns
///
/// The intersection over union in [0.0, 1.0] between the `bbox` and each candidate. A higher score means a larger fraction of the `bbox` is occluded by the candidate.
fn iou(bbox: &Array1<f32>, candidates: &Array2<f32>) -> Array1<f32> {
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

/// Intersection over union distance metric.
///
/// # Parameters
///
/// - `tracks`: A list of tracks.
/// - `detections`: A list of detections.
/// - `track_indices`: A list of indices to tracks that should be matched. Defaults to all `tracks`.
/// - `detection_indices`: A list of indices to detections that should be matched. Defaults to all `detections`.
///
/// # Returns
///
/// A cost matrix of shape track_indices.len(), detection_indices.len() where entry (i, j) is:
/// `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
pub fn iou_cost(
    tracks: &[Track],
    detections: &[Detection],
    track_indices: Option<Vec<usize>>,
    detection_indices: Option<Vec<usize>>,
) -> Array2<f32> {
    let track_indices = track_indices.unwrap_or_else(|| (0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(|| (0..detections.len()).collect());

    let mut cost_matrix = Array2::<f32>::zeros((0, detection_indices.len()));
    track_indices.iter().for_each(|track_idx| {
        let track = tracks.get(*track_idx).unwrap();

        if *track.time_since_update() > 1 {
            cost_matrix
                .push_row(Array1::from_elem(detection_indices.len(), f32::MAX).view())
                .unwrap();
        } else {
            let bbox = track.bbox().to_tlwh();
            let mut candidates = Array2::<f32>::zeros((0, 4));
            detection_indices.iter().for_each(|detection_idx| {
                candidates
                    .push(
                        Axis(0),
                        detections
                            .get(*detection_idx)
                            .unwrap()
                            .bbox()
                            .to_tlwh()
                            .view(),
                    )
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
            &arr1::<f32>(&[0.0, 0.0, 5.0, 5.0]),
            &arr2::<f32, _>(&[
                [0.0, 0.0, 5.0, 5.0],
                [1.0, 1.0, 6.0, 6.0],
                [2.0, 2.0, 7.0, 7.0],
                [3.0, 3.0, 8.0, 8.0],
                [4.0, 4.0, 9.0, 9.0],
                [5.0, 5.0, 10.0, 10.0],
            ]),
        );
        assert_eq!(
            iou,
            arr1::<f32>(&[1.0, 0.35555556, 0.13846155, 0.047058824, 0.00952381, 0.0])
        );
    }

    #[test]
    fn iou_cost() {
        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.clone().initiate(&BoundingBox::new(0.0, 0.0, 5.0, 5.0));
        let t0 = Track::new(mean, covariance, 0, 1.0, None, 0, 30, None);

        let kf = KalmanFilter::new();
        let (mean, covariance) = kf.clone().initiate(&BoundingBox::new(5.0, 5.0, 5.0, 5.0));
        let t1 = Track::new(mean, covariance, 1, 1.0, None, 0, 30, None);

        let d0 = Detection::new(BoundingBox::new(0.0, 0.0, 5.0, 5.0), 1.0, None, None);
        let d1 = Detection::new(BoundingBox::new(1.0, 1.0, 5.0, 5.0), 1.0, None, None);
        let d2 = Detection::new(BoundingBox::new(2.0, 2.0, 5.0, 5.0), 1.0, None, None);
        let d3 = Detection::new(BoundingBox::new(3.0, 3.0, 5.0, 5.0), 1.0, None, None);
        let d4 = Detection::new(BoundingBox::new(4.0, 4.0, 5.0, 5.0), 1.0, None, None);
        let d5 = Detection::new(BoundingBox::new(5.0, 5.0, 5.0, 5.0), 1.0, None, None);

        let cost_matrix =
            iou_matching::iou_cost(&vec![t0, t1], &vec![d0, d1, d2, d3, d4, d5], None, None);

        assert_eq!(
            cost_matrix,
            arr2::<f32, _>(&[
                [0.0, 0.5294118, 0.7804878, 0.9130435, 0.97959185, 1.0],
                [1.0, 0.97959185, 0.9130435, 0.7804878, 0.5294118, 0.0]
            ])
        );
    }
}
