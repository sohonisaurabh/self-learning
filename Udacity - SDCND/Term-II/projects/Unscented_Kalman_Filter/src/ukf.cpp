#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  //State vector dimensions
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.

  1. Initialization structure similar to EKF
  2. Extract deltaT
  3. Call Prediction() method
  4. Go into control stucture for Laser and Radar for the update step
  5. 
  */
 
  //1. Initialization structure similar to EKF
  // Filter is not initialized. Treat the first measurement here.
  if (!is_initialized_) {
    x_ << 1, 1, 0, 0, 0;

    //Caching components of first measurement
    float first_measurement_comp1 = meas_package.raw_measurements_[0];
    float first_measurement_comp2 = meas_package.raw_measurements_[1];

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      RADAR measures rho (radial distance) and phi (angle w.r.t direction axis of car),
      Hence, Convert radar from polar to cartesian coordinates to get px and py. phi is
      nothing but the yaw angle. There is no information on velocity and rate of change of
      yaw angle.
      */
      x_[0] = first_measurement_comp1*cos(first_measurement_comp2);
      x_[1] = first_measurement_comp1*sin(first_measurement_comp2);
      x_[2] = first_measurement_comp2;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      LASER measures px and py. Hence, first two values go directly. However, there is no
      information on velocity, yaw angle and rate of change of yaw angle.
      */
      x_[0] = first_measurement_comp1;
      x_[1] = first_measurement_comp2;
    }
    /**
     *Initializing State uncertainity matrix. Based on R_laser and R_radar, it is found that
     *uncertainty in measurement of position px and py is less (certain upto 0.1 units). While,
     *there is no information on velocity, uncertainty in vx and vy is high.
     */
    P_ << 10, 0, 0, 0, 0,
        0, 10, 0, 0, 0,
        0, 0, 1000, 0, 0,
        0, 0, 0, 10, 0,
        0, 0, 0, 0, 1000;

    time_us_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.

  1. The state transition function F in the case of CTRV model is non-linear, hence
      the prediction step will have to go through UKF sigma points approximation.
  2. Generate sigma points for step Xt (Use augmented sigma points to consider process noise)
  3. Calculate sigma points for step Xt+deltaT by passing them through F.
  4. Predict the mean and covariance of step Xt+deltaT
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.

  1. This process is Linear, hence use regular Kalman Filter equations.
  2. This is same as the one used in EKF
  3. Calculate NIS for Laser
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.

  1. This process is non-linear, so use UKF here.
  2. Make use of sigma points predicted in Predict step (Xsig_pred). Transform them to measurement 
  space to get transformed sigma points(Zsig).
  3. Find mean and covariance to get vector z.
  4. Use Xsig_pred, Zsig and z to find Kalman gain.
  5. Update x and P accordingly.
  */
}
