use crate::{linear_assignment, Detection, DistanceMetricFn, KalmanFilter, Track};
use anyhow::{Ok, Result};
use ndarray::*;
use ndarray_linalg::*;
use std::{collections::HashMap, fmt, rc::Rc};

#[derive(Clone)]
pub enum Metric {
    Cosine,
    Euclidean,
}

/// Compute pair-wise cosine distance between points in `a` and `b`.
///
/// # Parameters
///
/// * `x`: A matrix of N non-normalized row-vectors (sample points).
/// * `y`: A matrix of M non-normalized row-vectors (query points).
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
/// * `x`: A matrix of N row-vectors (sample points).
/// * `y`: A matrix of M row-vectors (query points).
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
    /// Either Cosine or Euclidean distance.
    metric: Metric,
    /// The matching threshold. Samples with larger distance are considered an invalid match.
    matching_threshold: f32,
    /// If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
    budget: Option<usize>,
    /// A HashMap that maps from target identities to the list of samples that have been observed so far.
    samples: HashMap<usize, Array2<f32>>,
}

impl Default for NearestNeighborDistanceMetric {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

impl fmt::Debug for NearestNeighborDistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NearestNeighborDistanceMetric")
            .field("matching_threshold", &self.matching_threshold)
            .field("budget", &self.budget)
            .field("samples", &self.samples)
            .finish()
    }
}

impl NearestNeighborDistanceMetric {
    /// Returns a new NearestNeighborDistanceMetric
    ///
    /// # Parameters
    ///
    /// * `metric`: Either `Metric::Euclidean` or `Metric::Cosine`.
    /// * `matching_threshold`: The matching threshold. Samples with larger distance are considered an invalid match. Default `0.2`.
    /// * `feature_length`: The feature vector length. Default `128`.
    /// * `budget`: If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
    pub fn new(
        metric: Option<Metric>,
        matching_threshold: Option<f32>,
        budget: Option<usize>,
    ) -> NearestNeighborDistanceMetric {
        NearestNeighborDistanceMetric {
            metric: metric.unwrap_or(Metric::Cosine),
            matching_threshold: matching_threshold.unwrap_or(0.2),
            budget,
            samples: HashMap::new(),
        }
    }

    /// Set metric
    pub fn with_metric(&mut self, metric: Metric) -> &mut Self {
        self.metric = metric;
        self
    }

    /// Set matching_threshold
    pub fn with_matching_threshold(&mut self, matching_threshold: f32) -> &mut Self {
        self.matching_threshold = matching_threshold;
        self
    }

    /// Set budget
    pub fn with_budget(&mut self, budget: usize) -> &mut Self {
        self.budget = Some(budget);
        self
    }

    /// Return the matching threshold
    pub fn matching_threshold(&self) -> f32 {
        self.matching_threshold
    }

    /// Return the stored feature vectors for a given track identifier
    pub fn track_features(&self, track_id: usize) -> Option<&Array2<f32>> {
        self.samples.get(&track_id)
    }

    /// Update the distance metric with new data.
    ///
    /// # Parameters
    ///
    /// * `features`: An NxM matrix of N features of dimensionality M.
    /// * `targets`: An integer array of associated target identities.
    /// * `active_targets`: A list of targets that are currently present in the scene.
    pub fn partial_fit(
        &mut self,
        features: &Array2<f32>,
        targets: &[usize],
        active_targets: &[usize],
    ) -> Result<()> {
        targets
            .iter()
            .zip(features.rows().into_iter())
            .try_for_each(|(target, feature)| {
                match self.samples.get_mut(target) {
                    Some(target_features) => {
                        target_features.push_row(feature)?;

                        // if budget is set truncate num rows from bottom
                        if let Some(budget) = self.budget {
                            if target_features.nrows() > budget {
                                target_features.slice_collapse(s![-(budget as i32).., ..])
                            }
                        }
                    }
                    None => {
                        let mut target_features = Array2::<f32>::zeros((0, feature.len()));
                        target_features.push_row(feature)?;
                        self.samples.insert(target.to_owned(), target_features);
                    }
                };

                Ok(())
            })?;

        self.samples.retain(|k, _| active_targets.contains(k));

        Ok(())
    }

    /// Compute distance between features and targets.
    ///
    /// # Parameters
    ///
    /// * `features`: An NxM matrix of N features of dimensionality M.
    /// * `targets`: A list of targets to match the given `features` against.
    ///
    /// # Returns
    ///
    /// A cost matrix of shape len(targets), len(features), where element (i, j) contains the closest squared distance between `targets[i]` and `features[j]`.
    pub fn distance(
        &self,
        features: &Array2<f32>,
        detections: &[usize],
        targets: &[usize],
    ) -> Result<Array2<f32>> {
        let mut cost_matrix = Array2::<f32>::zeros((0, features.nrows()));
        let metric_fn = match self.metric {
            Metric::Cosine => cosine_distance,
            Metric::Euclidean => euclidean_distance,
        };

        targets.iter().try_for_each(|target| {
            match self.samples.get(target) {
                Some(samples) => cost_matrix.push_row(metric_fn(samples, features).view())?,
                None => {
                    cost_matrix.push_row(Array1::<f32>::from_elem(detections.len(), 1.0).view())?
                }
            };

            Ok(())
        })?;

        Ok(cost_matrix)
    }

    /// Create the distance metric function required by the matching cascade
    ///
    /// # Parameters
    ///
    /// * `kf`: The Kalman filter.
    ///
    /// # Returns
    ///
    /// A function that calculates the distance between the incoming detection features and track existing feature vector
    pub fn distance_metric(&self, kf: &KalmanFilter) -> DistanceMetricFn {
        let nn_metric = self.clone();
        let kf = kf.clone();

        Rc::new(
            move |tracks: &[Track],
                  detections: &[Detection],
                  track_indices: &[usize],
                  detection_indices: &[usize]|
                  -> Result<Array2<f32>> {
                let features = stack(
                    Axis(0),
                    &detection_indices
                        .iter()
                        .filter_map(|i| {
                            let detection = detections.get(*i).unwrap();
                            detection.feature().as_ref().map(|feature| feature.view())
                        })
                        .collect::<Vec<_>>(),
                )?;

                let targets = track_indices
                    .iter()
                    .map(|i| tracks.get(*i).unwrap().track_id())
                    .collect::<Vec<_>>();

                let cost_matrix = nn_metric.distance(&features, detection_indices, &targets)?;

                linear_assignment::gate_cost_matrix(
                    &kf,
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
    use anyhow::Result;
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
    fn partial_fit() -> Result<()> {
        let mut metric = NearestNeighborDistanceMetric::new(None, None, None);
        metric.partial_fit(&array![[]], &[], &[])?;

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(0.0, 128.0, 1.0)
            ],
            &[0, 1],
            &[0, 1],
        )?;
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
        )?;
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

        metric.partial_fit(&stack![Axis(0), Array::range(1.0, 129.0, 1.0)], &[1], &[1])?;
        assert_eq!(metric.samples.get(&0), None);
        assert_eq!(
            metric.samples.get(&1).unwrap(),
            stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ]
        );

        Ok(())
    }

    #[test]
    fn cosine_distance() {
        let mut rng = Pcg32::seed_from_u64(0);
        let x = &stack![Axis(0), normal_array(&mut rng, 128)];
        let distances = nn_matching::cosine_distance(x, x);
        assert!(distances[0].abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance() -> Result<()> {
        let mut metric = NearestNeighborDistanceMetric::new(Some(Metric::Euclidean), None, None);

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ],
            &[0, 1],
            &[0, 1],
        )?;

        let distances = metric.distance(
            &stack![
                Axis(0),
                Array::range(0.1, 128.1, 1.0),
                Array::range(1.1, 129.1, 1.0)
            ],
            &[0, 1],
            &[0, 1],
        )?;

        assert_eq!(
            distances,
            array![[1.25f32, 154.875f32], [103.625f32, 1.25f32]],
        );

        Ok(())
    }
}
