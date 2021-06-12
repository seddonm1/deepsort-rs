use ndarray::*;
use ndarray_linalg::*;

/**
A simple Kalman filter for tracking bounding boxes in image space.

The 8-dimensional state space
    x, y, a, h, vx, vy, va, vh
contains the bounding box center position (x, y), aspect ratio a, height h,
and their respective velocities.

Object motion follows a constant velocity model. The bounding box location
(x, y, a, h) is taken as direct observation of the state space (linear
observation model).
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
    pub fn new() -> KalmanFilter {
        let ndim = 4;

        // Create Kalman filter model matrices and set initial values
        let mut motion_mat = Array2::from_diag(&Array1::<f32>::ones(2 * ndim));
        for i in 0..ndim {
            motion_mat[[i, ndim + i]] = 1.0f32;
        }

        let update_mat = concatenate!(
            Axis(1),
            Array2::from_diag(&Array1::<f32>::ones(ndim)),
            Array2::zeros((4, 4))
        );

        // Motion and observation uncertainty are chosen relative to the current
        // state estimate. These weights control the amount of uncertainty in
        // the model. This is a bit hacky.
        let std_weight_position = 1.0f32 / 20.0f32;
        let std_weight_velocity = 1.0f32 / 160.0f32;

        KalmanFilter {
            motion_mat,
            update_mat,
            std_weight_position,
            std_weight_velocity,
        }
    }

    /**
    Create track from unassociated measurement.

    Parameters
    ----------
    measurement : ndarray
        Bounding box coordinates (x, y, a, h) with center position (x, y),
        aspect ratio a, and height h.

    Returns
    -------
    (ndarray, ndarray)
        Returns the mean vector (8 dimensional) and covariance matrix (8x8
        dimensional) of the new track. Unobserved velocities are initialized
        to 0 mean.
    */
    pub fn initiate(self, measurement: Array1<f32>) -> (Array1<f32>, Array2<f32>) {
        let mean_pos = measurement.clone();
        let mean_vel = Array1::<f32>::zeros(mean_pos.raw_dim());
        let mean = concatenate![Axis(0), mean_pos, mean_vel];

        let std = array![
            2.0 * self.std_weight_position * measurement[3],
            2.0 * self.std_weight_position * measurement[3],
            1e-2,
            2.0 * self.std_weight_position * measurement[3],
            10.0 * self.std_weight_velocity * measurement[3],
            10.0 * self.std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self.std_weight_velocity * measurement[3],
        ];

        let covariance = Array2::from_diag(&std.mapv(|v| v.powi(2)).diag());

        (mean, covariance)
    }

    /**
    Run Kalman filter prediction step.

    Parameters
    ----------
    mean : ndarray
        The 8 dimensional mean vector of the object state at the previous
        time step.
    covariance : ndarray
        The 8x8 dimensional covariance matrix of the object state at the
        previous time step.
    Returns
    -------
    (ndarray, ndarray)
        Returns the mean vector and covariance matrix of the predicted
        state. Unobserved velocities are initialized to 0 mean.
    */
    pub fn predict(self, mean: Array1<f32>, covariance: Array2<f32>) -> (Array1<f32>, Array2<f32>) {
        let std_pos = array![
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3]
        ];

        let std_vel = array![
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3]
        ];

        let motion_cov = Array2::from_diag(
            &concatenate![Axis(0), std_pos, std_vel]
                .mapv(|v| v.powi(2))
                .diag(),
        );

        let mean = self.motion_mat.dot(&mean);
        let covariance = self.motion_mat.dot(&covariance).dot(&self.motion_mat.t()) + motion_cov;

        (mean, covariance)
    }

    /**
    Project state distribution to measurement space.

    Parameters
    ----------
    mean : ndarray
        The state's mean vector (8 dimensional array).
    covariance : ndarray
        The state's covariance matrix (8x8 dimensional).
    Returns
    -------
    (ndarray, ndarray)
        Returns the projected mean and covariance matrix of the given state
        estimate.
    */
    pub fn project(self, mean: Array1<f32>, covariance: Array2<f32>) -> (Array1<f32>, Array2<f32>) {
        let std = array![
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3]
        ];

        let innovation_cov = Array2::from_diag(&std.mapv(|v| v.powi(2)).diag());

        let mean = self.update_mat.dot(&mean);
        let covariance =
            self.update_mat.dot(&covariance).dot(&self.update_mat.t()) + innovation_cov;

        (mean, covariance)
    }

    /**
    Run Kalman filter correction step.
    Parameters
    ----------
    mean : ndarray
        The predicted state's mean vector (8 dimensional).
    covariance : ndarray
        The state's covariance matrix (8x8 dimensional).
    measurement : ndarray
        The 4 dimensional measurement vector (x, y, a, h), where (x, y)
        is the center position, a the aspect ratio, and h the height of the
        bounding box.
    Returns
    -------
    (ndarray, ndarray)
        Returns the measurement-corrected state distribution.
    */
    pub fn update(
        self,
        mean: Array1<f32>,
        covariance: Array2<f32>,
        measurement: Array1<f32>,
    ) -> (Array1<f32>, Array2<f32>) {
        let (projected_mean, projected_cov) =
            &self.clone().project(mean.clone(), covariance.clone());

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

        let innovation = &measurement - projected_mean;

        let new_mean = &mean + innovation.dot(&kalman_gain.t());
        let new_covariance = covariance - kalman_gain.dot(projected_cov).dot(&kalman_gain.t());

        (new_mean, new_covariance)
    }

    /**
    Compute gating distance between state distribution and measurements.
    A suitable distance threshold can be obtained from `chi2inv95`. If
    `only_position` is False, the chi-square distribution has 4 degrees of
    freedom, otherwise 2.

    Parameters
    ----------
    mean : ndarray
        Mean vector over the state distribution (8 dimensional).
    covariance : ndarray
        Covariance of the state distribution (8x8 dimensional).
    measurements : ndarray
        An Nx4 dimensional matrix of N measurements, each in
        format (x, y, a, h) where (x, y) is the bounding box center
        position, a the aspect ratio, and h the height.
    only_position : Optional[bool]
        If True, distance computation is done with respect to the bounding
        box center position only.

    Returns
    -------
    ndarray
        Returns an array of length N, where the i-th element contains the
        squared Mahalanobis distance between (mean, covariance) and
        `measurements[i]`.
    */
    pub fn gating_distance(
        self,
        mean: Array1<f32>,
        covariance: Array2<f32>,
        measurements: Array2<f32>,
    ) -> Array1<f32> {
        let (mean, covariance) = &self.project(mean, covariance);

        let cholesky_factor = covariance.cholesky(UPLO::Lower).unwrap();
        let d = &measurements - mean;
        let z = cholesky_factor
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &d.reversed_axes())
            .unwrap();

        (&z * &z).sum_axis(Axis(0))
    }
}

#[cfg(test)]
mod tests {
    use crate::KalmanFilter;
    use ndarray::*;

    #[test]
    fn test_new() {
        let kf = KalmanFilter::new();

        itertools::assert_equal(
            kf.motion_mat,
            array![
                [1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32]
            ],
        );
        itertools::assert_equal(
            kf.update_mat,
            array![
                [1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]
            ],
        );
    }

    #[test]
    fn test_initiate() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.initiate(array![1.0, 2.0, 3.0, 4.0]);

        itertools::assert_equal(
            mean,
            array![1.0f32, 2.0f32, 3.0f32, 4.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
        );
        itertools::assert_equal(
            covariance,
            array![
                [
                    0.16000001f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.16000001f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [0.0f32, 0.0f32, 0.0001f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.16000001f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0625f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0625f32, 0.0f32, 0.0f32],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.000000000099999994f32,
                    0.0f32
                ],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0625f32]
            ],
        );
    }

    #[test]
    fn test_predict() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(array![1.0, 2.0, 3.0, 4.0]);
        let (mean, covariance) = kf.clone().predict(mean.clone(), covariance.clone());

        itertools::assert_equal(
            mean,
            array![1.0f32, 2.0f32, 3.0f32, 4.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
        );
        itertools::assert_equal(
            covariance,
            array![
                [
                    0.26250002f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.26250002f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0002000001f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.000000000099999994f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.26250002f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32
                ],
                [
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.063125f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.063125f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.000000000099999994f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00000000019999999f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.0625f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.063125f32
                ]
            ],
        );
    }

    #[test]
    fn test_project() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(array![1.0, 2.0, 3.0, 4.0]);
        let (mean, covariance) = kf.clone().project(mean.clone(), covariance.clone());

        itertools::assert_equal(mean, array![1.0f32, 2.0f32, 3.0f32, 4.0f32]);
        itertools::assert_equal(
            covariance,
            array![
                [0.20000002f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.20000002, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.010100001f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.20000002f32]
            ],
        );
    }

    #[test]
    fn test_update() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(array![1.0, 2.0, 3.0, 4.0]);
        let (mean, covariance) = kf.clone().predict(mean.clone(), covariance.clone());
        let (mean, covariance) =
            kf.clone()
                .update(mean.clone(), covariance.clone(), array![2.0, 3.0, 4.0, 5.0]);

        itertools::assert_equal(
            mean,
            array![
                1.8677686f32,
                2.8677688f32,
                3.0196078f32,
                4.867769f32,
                0.20661156f32,
                0.20661156f32,
                0.000000009803921f32,
                0.20661156f32
            ],
        );
        itertools::assert_equal(
            covariance,
            array![
                [
                    0.034710735f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.008264463f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.034710735f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.008264463f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.00019607853f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00000000009803921f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.034710735f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.008264463f32
                ],
                [
                    0.00826446f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.050211776f32,
                    0.0f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.00826446f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.050211776f32,
                    0.0f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.00000000009803921f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00000000019999999f32,
                    0.0f32
                ],
                [
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.00826446f32,
                    0.0f32,
                    0.0f32,
                    0.0f32,
                    0.050211776f32
                ]
            ],
        );
    }

    #[test]
    fn test_gating_distance() {
        let kf = KalmanFilter::new();

        let (mean, covariance) = kf.clone().initiate(array![1.0, 2.0, 3.0, 4.0]);
        let (mean, covariance) = kf.clone().predict(mean.clone(), covariance.clone());
        let squared_maha = kf.gating_distance(
            mean,
            covariance,
            array![[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
        );

        itertools::assert_equal(squared_maha, array![107.95658, 431.82632,]);
    }
}
