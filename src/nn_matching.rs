use crate::{linear_assignment, Detection, DistanceMetricFn, KalmanFilter, Track};
use ndarray::*;
use ndarray_linalg::*;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

const FEATURE_LENGTH: usize = 128;

pub enum Metric {
    Cosine,
    Euclidean,
}

/// Compute pair-wise cosine distance between points in `a` and `b`.
///
/// # Parameters
///
/// - `x`: A matrix of N non-normalized row-vectors (sample points).
/// - `y`: A matrix of M non-normalized row-vectors (query points).
///
/// # Returns
///
/// A vector of length M that contains for each entry in `y` the smallest cosine distance to a sample in `x`.
fn cosine_distance(x: &Array2<f32>, y: &Array2<f32>) -> Array1<f32> {
    let (x_norm, _) = normalize(x.clone(), NormalizeAxis::Row);
    let (y_norm, _) = normalize(y.clone(), NormalizeAxis::Row);

    let distances = 1.0 - x_norm.dot(&y_norm.t());

    distances.fold_axis(Axis(0), f32::MAX, |&accumulator, &value| {
        accumulator.min(value)
    })
}

/// Compute pair-wise sqauared distance between points in `a` and `b`.
///
/// # Parameters
///
/// - `x`: A matrix of N row-vectors (sample points).
/// - `y`: A matrix of M row-vectors (query points).
///
/// # Returns
///
/// A vector of length M that contains for each entry in `y` the smallest Euclidean distance to a sample in `x`.
fn euclidean_distance(x: &Array2<f32>, y: &Array2<f32>) -> Array1<f32> {
    let x2 = x.mapv(|v| v.powi(2)).sum_axis(Axis(1)).insert_axis(Axis(0));
    let y2 = y.mapv(|v| v.powi(2)).sum_axis(Axis(1)).insert_axis(Axis(0));

    let res = -2.0 * x.dot(&y.t()) + x2.t() + y2;
    let distances = res.mapv(|v| v.clamp(0.0, f32::MAX));

    distances
        .fold_axis(Axis(0), f32::MAX, |&accumulator, &value| {
            accumulator.min(value)
        })
        .mapv(|v| v.clamp(0.0, f32::MAX))
}

/// A nearest neighbor distance metric that, for each target, returns the closest distance to any sample that has been observed so far.
#[derive(Clone)]
pub struct NearestNeighborDistanceMetric {
    metric: fn(&Array2<f32>, &Array2<f32>) -> Array1<f32>,
    /// The matching threshold. Samples with larger distance are considered an invalid match.
    matching_threshold: f32,
    /// If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
    budget: Option<usize>,
    /// The length of the feature vectors.
    feature_length: usize,
    /// A HashMap that maps from target identities to the list of samples that have been observed so far.
    samples: HashMap<usize, Array2<f32>>,
}

impl Default for NearestNeighborDistanceMetric {
    fn default() -> Self {
        Self::new(None, None, None, None)
    }
}

impl fmt::Debug for NearestNeighborDistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NearestNeighborDistanceMetric")
            .field("matching_threshold", &self.matching_threshold)
            .field("budget", &self.budget)
            .field("feature_length", &self.feature_length)
            .field("samples", &self.samples)
            .finish()
    }
}

impl NearestNeighborDistanceMetric {
    /// Returns a new NearestNeighborDistanceMetric
    ///
    /// # Parameters
    ///
    /// - `metric`: Either `Metric::Euclidean` or `Metric::Cosine`.
    /// - `matching_threshold`: The matching threshold. Samples with larger distance are considered an invalid match. Default `0.2`.
    /// - `feature_length`: The feature vector length. Default `128`.
    /// - `budget`: If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
    pub fn new(
        metric: Option<Metric>,
        matching_threshold: Option<f32>,
        feature_length: Option<usize>,
        budget: Option<usize>,
    ) -> NearestNeighborDistanceMetric {
        NearestNeighborDistanceMetric {
            metric: metric
                .map(|metric| match metric {
                    Metric::Cosine => cosine_distance,
                    Metric::Euclidean => euclidean_distance,
                })
                .unwrap_or(cosine_distance),
            matching_threshold: matching_threshold.unwrap_or(0.2),
            feature_length: feature_length.unwrap_or(FEATURE_LENGTH),
            budget,
            samples: HashMap::new(),
        }
    }

    /// Return the matching threshold
    pub fn matching_threshold(&self) -> f32 {
        self.matching_threshold
    }

    /// Return the feature length
    pub fn feature_length(&self) -> usize {
        self.feature_length
    }

    /// Return the stored feature vectors for a given track identifier
    pub fn track_features(&self, track_id: usize) -> Option<&Array2<f32>> {
        self.samples.get(&track_id)
    }

    /// Update the distance metric with new data.
    ///
    /// # Parameters
    ///
    /// - `features`: An NxM matrix of N features of dimensionality M.
    /// - `targets`: An integer array of associated target identities.
    /// - `active_targets`: A list of targets that are currently present in the scene.
    pub fn partial_fit(
        &mut self,
        features: &Array2<f32>,
        targets: &[usize],
        active_targets: &[usize],
    ) {
        targets
            .iter()
            .zip(features.rows().into_iter())
            .for_each(|(target, feature)| match self.samples.get_mut(target) {
                Some(target_features) => {
                    target_features.push_row(feature).unwrap();

                    // if budget is set truncate num rows from bottom
                    if let Some(budget) = self.budget {
                        if target_features.nrows() > budget {
                            target_features.slice_collapse(s![-(budget as i32).., ..])
                        }
                    }
                }
                None => {
                    let mut target_features = Array2::<f32>::zeros((0, feature.len()));
                    target_features.push_row(feature).unwrap();
                    self.samples.insert(target.to_owned(), target_features);
                }
            });

        self.samples.retain(|k, _| active_targets.contains(k));
    }

    /// Compute distance between features and targets.
    ///
    /// # Parameters
    ///
    /// - `features`: An NxM matrix of N features of dimensionality M.
    /// - `targets`: A list of targets to match the given `features` against.
    ///
    /// # Returns
    ///
    /// A cost matrix of shape len(targets), len(features), where element (i, j) contains the closest squared distance between `targets[i]` and `features[j]`.
    pub fn distance(
        &self,
        features: &Array2<f32>,
        detections: &[usize],
        targets: &[usize],
    ) -> Array2<f32> {
        let mut cost_matrix = Array2::<f32>::zeros((0, detections.len()));
        targets.iter().for_each(|target| {
            match self.samples.get(target) {
                Some(samples) => cost_matrix
                    .push_row((self.metric)(samples, features).view())
                    .unwrap(),
                None => cost_matrix
                    .push_row(Array1::<f32>::from_elem(detections.len(), 1.0).view())
                    .unwrap(),
            };
        });
        cost_matrix
    }

    /// Create the distance metric function required by the matching cascade
    ///
    /// # Parameters
    ///
    /// - `kf`: The Kalman filter.
    ///
    /// # Returns
    ///
    /// A function that calculates the distance between the incoming detection features and track existing feature vector
    pub fn distance_metric(&self, kf: KalmanFilter) -> DistanceMetricFn {
        let nn_metric = self.clone();
        Rc::new(
            move |tracks: &[Track],
                  detections: &[Detection],
                  track_indices: Option<Vec<usize>>,
                  detection_indices: Option<Vec<usize>>|
                  -> Array2<f32> {
                let detection_indices = detection_indices.unwrap();
                let track_indices = track_indices.unwrap();

                let mut features = Array2::<f32>::zeros((0, nn_metric.feature_length));
                detection_indices.iter().for_each(|i| {
                    if let Some(feature) = detections.get(*i).unwrap().feature() {
                        features.push_row(feature.view()).unwrap()
                    }
                });
                let targets = track_indices
                    .iter()
                    .map(|i| tracks.get(*i).unwrap().track_id())
                    .collect::<Vec<_>>();

                let cost_matrix = nn_metric.distance(&features, &detection_indices, &targets);

                linear_assignment::gate_cost_matrix(
                    kf.clone(),
                    cost_matrix,
                    tracks,
                    detections,
                    track_indices,
                    detection_indices,
                    None,
                    None,
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_matching::{self, *};

    use rand::prelude::*;
    use rand_distr::Normal;
    use rand_pcg::{Lcg64Xsh32, Pcg32};

    /**
    Returns a 1 dimensional array of length n with a normal distribution
    */
    fn normal_array(rng: &mut Lcg64Xsh32, n: usize) -> Array1<f32> {
        let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
        Array1::from_iter(0..n).map(|_| normal.sample(rng))
    }

    #[test]
    fn partial_fit() {
        let mut metric = NearestNeighborDistanceMetric::new(None, None, None, None);
        metric.partial_fit(&array![[]], &[], &[]);

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(0.0, 128.0, 1.0)
            ],
            &[0, 1],
            &[0, 1],
        );
        assert_eq!(
            metric.samples.get(&0).unwrap(),
            stack![Axis(0), Array::range(0.0, 128.0, 1.0)]
        );
        assert_eq!(
            metric.samples.get(&1).unwrap(),
            stack![Axis(0), Array::range(0.0, 128.0, 1.0)]
        );

        metric.partial_fit(
            &stack![Axis(0), Array::range(1.0, 129.0, 1.0)],
            &[0],
            &[0, 1],
        );
        assert_eq!(
            metric.samples.get(&0).unwrap(),
            stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ]
        );
        assert_eq!(
            metric.samples.get(&1).unwrap(),
            stack![Axis(0), Array::range(0.0, 128.0, 1.0)]
        );

        metric.partial_fit(&stack![Axis(0), Array::range(1.0, 129.0, 1.0)], &[1], &[1]);
        assert_eq!(metric.samples.get(&0), None);
        assert_eq!(
            metric.samples.get(&1).unwrap(),
            stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ]
        );
    }

    #[test]
    fn cosine_distance() {
        let mut rng = Pcg32::seed_from_u64(0);
        for _ in 0..1 {
            let x = &stack![Axis(0), normal_array(&mut rng, 128)];
            let distances = nn_matching::cosine_distance(x, x);
            assert!(distances[0].abs() < 1e-6);
        }
    }

    #[test]
    fn euclidean_distance() {
        let mut metric =
            NearestNeighborDistanceMetric::new(Some(Metric::Euclidean), None, None, None);

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ],
            &[0, 1],
            &[0, 1],
        );

        let distances = metric.distance(
            &stack![
                Axis(0),
                Array::range(0.1, 128.1, 1.0),
                Array::range(1.1, 129.1, 1.0)
            ],
            &[0, 1],
            &[0, 1],
        );

        assert_eq!(
            distances,
            array![[1.25f32, 154.875f32], [103.625f32, 1.25f32]],
        );
    }
}
