use ndarray::*;
use ndarray_linalg::*;
use std::collections::HashMap;

pub enum Metric {
    Cosine,
    Euclidean,
}

/**
Compute pair-wise cosine distance between points in `a` and `b`.

Parameters
----------
x : ndarray
    A matrix of N row-vectors (sample points).
y : ndarray
    A matrix of M row-vectors (query points).

Returns
-------
ndarray
    A vector of length M that contains for each entry in `y` the
    smallest cosine distance to a sample in `x`.
*/
fn cosine_distance(x: Array2<f32>, y: Array2<f32>) -> Array1<f32> {
    let (x_norm, _) = norm::normalize(x, NormalizeAxis::Row);
    let (y_norm, _) = norm::normalize(y, NormalizeAxis::Row);

    let distances = 1.0 - x_norm.dot(&y_norm.t());

    distances.fold_axis(Axis(0), f32::MAX, |&accumulator, &value| {
        accumulator.min(value)
    })
}

/**
Compute pair-wise sqauared distance between points in `a` and `b`.

Parameters
----------
x : ndarray
    A matrix of N row-vectors (sample points).
y : ndarray
    A matrix of M row-vectors (query points).

Returns
-------
ndarray
    A vector of length M that contains for each entry in `y` the
    smallest Euclidean distance to a sample in `x`.
*/
fn euclidean_distance(x: Array2<f32>, y: Array2<f32>) -> Array1<f32> {
    let x2 = x.mapv(|v| v.powi(2)).sum_axis(Axis(1));
    let y2 = y.mapv(|v| v.powi(2)).sum_axis(Axis(1));

    let res = -2.0 * x.dot(&y.t()) + stack![Axis(0), x2].t() + stack![Axis(0), y2];
    let distances = res.mapv(|v| v.clamp(0.0, f32::MAX));

    distances
        .fold_axis(Axis(0), f32::MAX, |&accumulator, &value| {
            accumulator.min(value)
        })
        .mapv(|v| v.clamp(0.0, f32::MAX))
}

/**
A nearest neighbor distance metric that, for each target, returns
the closest distance to any sample that has been observed so far.

Parameters
----------
metric : str
    Either "euclidean" or "cosine".
matching_threshold: float
    The matching threshold. Samples with larger distance are considered an
    invalid match.
budget : Optional[int]
    If not None, fix samples per class to at most this number. Removes
    the oldest samples when the budget is reached.

Attributes
----------
samples : Dict[int -> List[ndarray]]
    A dictionary that maps from target identities to the list of samples
    that have been observed so far.
*/
#[derive(Debug, Clone)]
pub struct NearestNeighborDistanceMetric {
    metric: fn(Array2<f32>, Array2<f32>) -> Array1<f32>,
    matching_threshold: f32,
    budget: Option<i32>,
    samples: HashMap<usize, Array2<f32>>,
}

impl NearestNeighborDistanceMetric {
    pub fn new(
        metric: Metric,
        matching_threshold: f32,
        budget: Option<i32>,
    ) -> NearestNeighborDistanceMetric {
        let metric_impl = match metric {
            Metric::Cosine => cosine_distance,
            Metric::Euclidean => euclidean_distance,
        };

        NearestNeighborDistanceMetric {
            metric: metric_impl,
            matching_threshold,
            budget,
            samples: HashMap::new(),
        }
    }

    /**
    Return the matching threshold
    */
    pub fn matching_threshold(&self) -> f32 {
        self.matching_threshold
    }

    /**
    Return the stored feature vectors for a given track identifier
    */
    pub fn features(&self, track_id: usize) -> Option<&Array2<f32>> {
        self.samples.get(&track_id)
    }

    /**
    Update the distance metric with new data.

    Parameters
    ----------
    features : ndarray
        An NxM matrix of N features of dimensionality M.
    targets : ndarray
        An integer array of associated target identities.
    active_targets : List[int]
        A list of targets that are currently present in the scene.
    */
    pub fn partial_fit(
        &mut self,
        features: &Array2<f32>,
        targets: &Array1<usize>,
        active_targets: &[usize],
    ) {
        if targets.len() != 0 {
            Zip::from(features.rows())
                .and(targets)
                .for_each(|feature, target| match self.samples.get_mut(target) {
                    Some(target_features) => {
                        target_features.push_row(feature).unwrap();
                        if let Some(budget) = &self.budget {
                            target_features.slice(s![-budget.., ..]);
                        }
                    }
                    None => {
                        let mut target_features = Array2::<f32>::zeros((0, 128));
                        target_features.push_row(feature).unwrap();
                        self.samples.insert(target.to_owned(), target_features);
                    }
                });
        }

        self.samples.retain(|k, _| active_targets.contains(k));
    }

    /**
    Compute distance between features and targets.

    Parameters
    ----------
    features : ndarray
        An NxM matrix of N features of dimensionality M.
    targets : List[int]
        A list of targets to match the given `features` against.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape len(targets), len(features), where
        element (i, j) contains the closest squared distance between
        `targets[i]` and `features[j]`.
    */
    pub fn distance(&self, features: &Array2<f32>, targets: &[usize]) -> Array2<f32> {
        let mut cost_matrix = Array2::<f32>::zeros((0, features.nrows()));
        targets.iter().for_each(|target| {
            cost_matrix
                .push_row(
                    (self.metric)(self.samples.get(target).unwrap().clone(), features.clone())
                        .view(),
                )
                .unwrap();
        });

        cost_matrix
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_matching::*;

    #[test]
    fn partial_fit() {
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Cosine, 0.5, None);
        metric.partial_fit(&array![[]], &array![], &vec![]);

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(0.0, 128.0, 1.0)
            ],
            &array![0, 1],
            &vec![0, 1],
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
            &array![0],
            &vec![0, 1],
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

        metric.partial_fit(
            &stack![Axis(0), Array::range(1.0, 129.0, 1.0)],
            &array![1],
            &vec![1],
        );
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
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Cosine, 0.5, None);

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ],
            &array![0, 1],
            &vec![0, 1],
        );

        let distances = metric.distance(
            &stack![
                Axis(0),
                Array::range(0.1, 128.1, 1.0),
                Array::range(1.1, 129.1, 1.0)
            ],
            &vec![0, 1],
        );

        assert_eq!(
            distances,
            array![
                [0.00000023841858f32, 0.000027418137f32],
                [0.000018537045f32, 0.00000023841858f32]
            ],
        );
    }

    #[test]
    fn euclidean_distance() {
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Euclidean, 0.5, None);

        metric.partial_fit(
            &stack![
                Axis(0),
                Array::range(0.0, 128.0, 1.0),
                Array::range(1.0, 129.0, 1.0)
            ],
            &array![0, 1],
            &vec![0, 1],
        );

        let distances = metric.distance(
            &stack![
                Axis(0),
                Array::range(0.1, 128.1, 1.0),
                Array::range(1.1, 129.1, 1.0)
            ],
            &vec![0, 1],
        );

        assert_eq!(
            distances,
            array![[1.25f32, 154.875f32], [103.625f32, 1.25f32]],
        );
    }
}
