#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;
  /**
  TODO COMPLETED
  */

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    cout << "EKF: " << endl;
    //Declaring state vector efk_.x_ and assigining to 1, 1, 0, 0
    //Velocity(vx and vy) is assumed to be zero at the beginning.
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 0, 0;

    //Caching x and y components of first measurement
    float first_measurement_x = measurement_pack.raw_measurements_[0];
    float first_measurement_y = measurement_pack.raw_measurements_[1];

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      RADAR measures rho (radial distance) and phi (angle w.r.t direction axis of car),
      while the state vector contains px and py.
      Hence, Convert radar from polar to cartesian coordinates and initialize state.
      */
      ekf_.x_[0] = first_measurement_x*cos(first_measurement_y);
      ekf_.x_[1] = first_measurement_x*sin(first_measurement_y);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      No need of conversion here as LASER measures px and py.
      */
      ekf_.x_[0] = first_measurement_x;
      ekf_.x_[1] = first_measurement_y;
    }

    /**
     *Initializing State uncertainity matrix. Based on R_laser and R_radar, it is found that
     *uncertainty in measurement of position px and py is less (certain upto 0.1 units). While,
     *there is no information on velocity, uncertainty in vx and vy is high.
     */
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 10, 0, 0, 0,
        0, 10, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

    previous_timestamp_ = measurement_pack.timestamp_;

    /**
     *Declaration of F matrix once. F matrix is updated with deltaT for every measurement received.
     *Initializing State transition matrix*/
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  /*Taking into account the timestamp*/
  float deltaT = measurement_pack.timestamp_ - previous_timestamp_;
  //Converting time to seconds.
  deltaT = deltaT/pow(10.0, 6);

  //Setting previous timestamp to current timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  /*Initializing Process covariance matrix*/
  float noise_ax = 9;
  float noise_ay = 9;
  float time_r2 = pow(deltaT, 2);
  float time_r3 = pow(deltaT, 3);
  float time_r4 = pow(deltaT, 4);

  //Update F matrix to take into account deltaT for latest measurement received.
  ekf_.F_.row(0)[2] = deltaT;
  ekf_.F_.row(1)[3] = deltaT;

  //Declare and fill state covariance matrix representing stocastic part of motion.
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << time_r4*noise_ax/4, 0, time_r3*noise_ax/2, 0,
      0, time_r4*noise_ay/4, 0, time_r3*noise_ay/2,
      time_r3*noise_ax/2, 0, time_r2*noise_ax, 0,
      0, time_r3*noise_ay/2, 0, time_r2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates. Proceed with Extended Kalman filter.
    ekf_.R_ = R_radar_;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates. Proceed with normal Kalman filter.
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
