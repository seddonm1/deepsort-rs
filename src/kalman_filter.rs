use crate::BoundingBox;
use ndarray::*;
use ndarray_linalg::*;
use std::collections::HashMap;

lazy_static! {
    /**
    Table for the 0.95 quantile of the chi-square distribution with N degrees of freedom (contains values for N=1, ..., 9).
    Taken from MATLAB/Octave's chi2inv function and used as Mahalanobis gating threshold.
    */
    pub static ref CHI2INV95: HashMap<usize, f32> = HashMap::from([
        (1, 3.8415),
        (2, 5.9915),
        (3, 7.8147),
        (4, 9.4877),
        (5, 11.070),
        (6, 12.592),
        (7, 14.067),
        (8, 15.507),
        (9, 16.919),
    ]);
}

/**
A simple Kalman filter for tracking bounding boxes in image space.

The 8-dimensional state space:
    x, y, a, h, vx, vy, va, vh
contains the bounding box center position (x, y), aspect ratio a, height h, and their respective velocities.

Object motion follows a constant velocity model. The bounding box location (x, y, a, h) is taken as direct observation of the state space (linear observation model).
*/
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    motion_mat: Array2<f32>,
    update_mat: Array2<f32>,
    std_weight_position: f32,
    std_weight_velocity: f32,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    /// Returns a new KalmanFilter
    pub fn new() -> KalmanFilter {
        let ndim = 4;

        // Create Kalman filter model matrices and set initial values
        let mut motion_mat = Array2::from_diag(&Array1::<f32>::ones(2 * ndim));
        for i in 0..ndim {
            motion_mat[[i, ndim + i]] = 1.0;
        }

        let update_mat = concatenate!(
            Axis(1),
            Array2::from_diag(&Array1::<f32>::ones(ndim)),
            Array2::zeros((4, 4))
        );

        // Motion and observation uncertainty are chosen relative to the current
        // state estimate. These weights control the amount of uncertainty in
        // the model. This is a bit hacky.
        let std_weight_position = 1.0 / 20.0;
        let std_weight_velocity = 1.0 / 160.0;

        KalmanFilter {
            motion_mat,
            update_mat,
            std_weight_position,
            std_weight_velocity,
        }
    }

    /// Create track from unassociated measurement.
    ///
    /// # Arguments
    ///
    /// - `bbox`: Bounding box object of the new measurement.
    ///
    /// # Returns
    ///
    /// A tuple with the following two entries of the new track:
    /// - The mean vector (8 dimensional).
    /// - The covariance matrix (8x8 dimensional).
    ///
    /// Unobserved velocities are initialized to 0 mean.
    pub fn initiate(&self, bbox: &BoundingBox) -> (Array1<f32>, Array2<f32>) {
        let mean_pos = bbox.to_xyah();
        let mean_vel = Array1::<f32>::zeros(mean_pos.raw_dim());
        let mean = concatenate![Axis(0), mean_pos, mean_vel];

        let std = arr1::<f32>(&[
            2.0 * self.std_weight_position * mean_pos[3],
            2.0 * self.std_weight_position * mean_pos[3],
            1e-2,
            2.0 * self.std_weight_position * mean_pos[3],
            10.0 * self.std_weight_velocity * mean_pos[3],
            10.0 * self.std_weight_velocity * mean_pos[3],
            1e-5,
            10.0 * self.std_weight_velocity * mean_pos[3],
        ]);

        let covariance = Array2::from_diag(&std.mapv(|v| v.powi(2)).diag());

        (mean, covariance)
    }

    /// Run Kalman filter prediction step.
    ///
    /// # Arguments
    ///
    /// - `mean`: The 8 dimensional mean vector of the object state at the previous time step.
    /// - `covariance`: The 8x8 dimensional covariance matrix of the object state at the previous time step.
    ///
    /// # Returns
    ///
    /// A tuple with the following two entries of the predicted state:
    /// - The mean vector (8 dimensional).
    /// - The covariance matrix (8x8 dimensional).
    ///
    /// Unobserved velocities are initialized to 0 mean.
    pub fn predict(
        &self,
        mean: &Array1<f32>,
        covariance: &Array2<f32>,
    ) -> (Array1<f32>, Array2<f32>) {
        let std_pos = arr1::<f32>(&[
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3],
        ]);

        let std_vel = arr1::<f32>(&[
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3],
        ]);

        let motion_cov = Array2::from_diag(
            &concatenate![Axis(0), std_pos, std_vel]
                .mapv(|v| v.powi(2))
                .diag(),
        );

        let mean = self.motion_mat.dot(mean);
        let covariance = self.motion_mat.dot(covariance).dot(&self.motion_mat.t()) + motion_cov;

        (mean, covariance)
    }

    /// Project state distribution to measurement space.
    ///
    /// # Arguments
    ///
    /// - `mean`: The state's mean vector (8 dimensional array).
    /// - `covariance`: The state's covariance matrix (8x8 dimensional).
    ///
    /// # Returns
    ///
    /// A tuple with the following two entries of the given state estimate:
    /// - The mean vector (8 dimensional).
    /// - The covariance matrix (8x8 dimensional).
    pub fn project(
        &self,
        mean: &Array1<f32>,
        covariance: &Array2<f32>,
    ) -> (Array1<f32>, Array2<f32>) {
        let std = arr1::<f32>(&[
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3],
        ]);

        let innovation_cov = Array2::from_diag(&std.mapv(|v| v.powi(2)).diag());

        let mean = self.update_mat.dot(mean);
        let covariance = self.update_mat.dot(covariance).dot(&self.update_mat.t()) + innovation_cov;

        (mean, covariance)
    }

    /// Run Kalman filter correction step.
    ///
    /// # Arguments
    ///
    /// - `mean`: The state's mean vector (8 dimensional array).
    /// - `covariance`: The state's covariance matrix (8x8 dimensional).
    /// - `measurement`: The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center position, a the aspect ratio, and h the height of the bounding box.
    ///
    /// # Returns
    ///
    /// A tuple with the following two entries of the measurement-corrected state distribution:
    /// - The mean vector (8 dimensional).
    /// - The covariance matrix (8x8 dimensional).
    pub fn update(
        &self,
        mean: &Array1<f32>,
        covariance: &Array2<f32>,
        measurement: &Array1<f32>,
    ) -> (Array1<f32>, Array2<f32>) {
        let (projected_mean, projected_cov) = &self.project(mean, covariance);

        let cholesky_factor = projected_cov.factorizec(UPLO::Lower).unwrap();

        let covariance_dot = covariance.dot(&self.update_mat.t());
        let mut kalman_gain = Array2::<f32>::zeros((0, 4));
        for i in 0..covariance_dot.nrows() {
            kalman_gain
                .push_row(
                    cholesky_factor
                        .solvec(&covariance_dot.row(i))
                        .unwrap()
                        .view(),
                )
                .unwrap();
        }

        let innovation = measurement - projected_mean;

        let new_mean = mean + innovation.dot(&kalman_gain.t());
        let new_covariance = covariance - kalman_gain.dot(projected_cov).dot(&kalman_gain.t());

        (new_mean, new_covariance)
    }

    /// Compute gating distance between state distribution and measurements.
    ///
    /// # Parameters
    ///
    /// - `mean`: Mean vector over the state distribution (8 dimensional).
    /// - `covariance`: Covariance of the state distribution (8x8 dimensional).
    /// - `measurements`: An Nx4 dimensional matrix of N measurements, each in format (x, y, a, h) where (x, y) is the bounding box center position, a the aspect ratio, and h the height.
    ///
    /// # Returns
    ///
    /// An array of length N, where the i-th element contains the squared Mahalanobis distance between (mean, covariance) and `measurements[i]`.
    pub fn gating_distance(
        &self,
        mean: &Array1<f32>,
        covariance: &Array2<f32>,
        measurements: &Array2<f32>,
    ) -> Array1<f32> {
        let (mean, covariance) = &self.project(mean, covariance);

        let cholesky_factor = covariance.cholesky(UPLO::Lower).unwrap();
        let d = measurements - mean;
        let z = cholesky_factor
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &d.reversed_axes())
            .unwrap();

        (&z * &z).sum_axis(Axis(0))
    }
}

#[cfg(test)]
mod tests {
    use crate::{BoundingBox, KalmanFilter};
    use assert_approx_eq::assert_approx_eq;
    use ndarray::*;

    #[test]
    fn new() {
        let kf = KalmanFilter::new();

        assert_eq!(
            kf.motion_mat,
            arr2::<f32, _>(&[
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ]),
        );
        assert_eq!(
            kf.update_mat,
            arr2::<f32, _>(&[
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            ]),
        );
    }

    #[test]
    fn initiate() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 1.0, 2.0, 3.0));

        assert_eq!(
            mean,
            arr1::<f32>(&[1.0, 2.5, 0.6666667, 3.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            covariance,
            arr2::<f32, _>(&[
                [0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.03515625, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.03515625, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000000099999994, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03515625]
            ]),
        );
    }

    #[test]
    fn predict() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 1.0, 2.0, 3.0));
        let (mean, covariance) = kf.predict(&mean, &covariance);

        assert_eq!(
            mean,
            arr1::<f32>(&[1.0, 2.5, 0.6666667, 3.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            covariance,
            arr2::<f32, _>(&[
                [0.14765626, 0.0, 0.0, 0.0, 0.03515625, 0.0, 0.0, 0.0],
                [0.0, 0.14765626, 0.0, 0.0, 0.0, 0.03515625, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0002000001,
                    0.0,
                    0.0,
                    0.0,
                    0.000000000099999994,
                    0.0
                ],
                [0.0, 0.0, 0.0, 0.14765626, 0.0, 0.0, 0.0, 0.03515625],
                [0.03515625, 0.0, 0.0, 0.0, 0.035507813, 0.0, 0.0, 0.0],
                [0.0, 0.03515625, 0.0, 0.0, 0.0, 0.035507813, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.000000000099999994,
                    0.0,
                    0.0,
                    0.0,
                    0.00000000019999999,
                    0.0
                ],
                [0.0, 0.0, 0.0, 0.03515625, 0.0, 0.0, 0.0, 0.035507813]
            ]),
        );
    }

    #[test]
    fn project() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 1.0, 2.0, 3.0));
        let (mean, covariance) = kf.project(&mean, &covariance);

        assert_eq!(mean, arr1::<f32>(&[1.0, 2.5, 0.6666667, 3.0]));
        assert_eq!(
            covariance,
            arr2::<f32, _>(&[
                [0.112500004, 0.0, 0.0, 0.0],
                [0.0, 0.112500004, 0.0, 0.0],
                [0.0, 0.0, 0.010100001, 0.0],
                [0.0, 0.0, 0.0, 0.112500004]
            ]),
        );
    }

    #[test]
    fn update() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(0.0, 1.0, 2.0, 3.0));
        let (mean, covariance) = kf.predict(&mean, &covariance);
        let (mean, covariance) = kf.update(&mean, &covariance, &arr1::<f32>(&[1.0, 2.0, 3.0, 4.0]));

        assert_eq!(
            mean,
            arr1::<f32>(&[
                1.0,
                2.0661156,
                0.7124183,
                3.8677685,
                0.0,
                -0.10330577,
                0.000000022875815,
                0.20661154
            ]),
        );
        assert_eq!(
            covariance,
            arr2::<f32, _>(&[
                [0.019524798, 0.0, 0.0, 0.0, 0.004648762, 0.0, 0.0, 0.0],
                [0.0, 0.019524798, 0.0, 0.0, 0.0, 0.004648762, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.00019607853,
                    0.0,
                    0.0,
                    0.0,
                    0.00000000009803921,
                    0.0
                ],
                [0.0, 0.0, 0.0, 0.019524798, 0.0, 0.0, 0.0, 0.004648762],
                [0.00464876, 0.0, 0.0, 0.0, 0.028244127, 0.0, 0.0, 0.0],
                [0.0, 0.00464876, 0.0, 0.0, 0.0, 0.028244127, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.00000000009803921,
                    0.0,
                    0.0,
                    0.0,
                    0.00000000019999999,
                    0.0
                ],
                [0.0, 0.0, 0.0, 0.00464876, 0.0, 0.0, 0.0, 0.028244127]
            ]),
        );
    }

    #[test]
    fn gating_distance() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(&BoundingBox::new(1.0, 2.0, 3.0, 4.0));
        let (mean, covariance) = kf.predict(&mean, &covariance);
        let squared_maha = kf.gating_distance(
            &mean,
            &covariance,
            &stack![
                Axis(0),
                BoundingBox::new(2.0, 3.0, 4.0, 5.0).to_xyah(),
                BoundingBox::new(3.0, 4.0, 5.0, 6.0).to_xyah()
            ],
        );

        assert_approx_eq!(squared_maha[0], 18.426916, 1e-4);
        assert_approx_eq!(squared_maha[1], 73.4081, 1e-4);
    }
}
