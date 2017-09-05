#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

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
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.5;

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

  //Augmented state vector dimensions
  n_aug_ = 7;

  //Lambda
  lambda_ = 3 - n_aug_;

  //Declare X_sig_pred
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //Declare the weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  Tools tools;
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
      //x_[2] = first_measurement_comp2;
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
    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 10, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 10;

    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  //2. Extract deltaT
  /*****************************************************************************
   *  Calculate deltaT
   ****************************************************************************/
  /*Taking into account the timestamp*/
  float deltaT = meas_package.timestamp_ - time_us_;
  //Converting time to seconds.
  deltaT = deltaT/pow(10.0, 6);
  // std::cout<<"Delta T is: "<<deltaT<<std::endl;

  //Setting previous timestamp to current timestamp
  time_us_ = meas_package.timestamp_;

  //3. Call Prediction() method
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  Prediction(deltaT);

  //4. Go into control stucture for Laser and Radar for the update step
   /*****************************************************************************
   *  Update
   ****************************************************************************/
   if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
    std::cout<<"Radar update successful!"<<std::endl;
  } else {
    UpdateLidar(meas_package);
    std::cout<<"Lidar update successful!"<<std::endl;
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // std::cout<<"Delta T inside pred is: "<<delta_t<<std::endl;
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

  /*1. The state transition function F in the case of CTRV model is non-linear, hence
      the prediction step will have to go through UKF sigma points approximation.*/
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug[n_x_] = 0;
  x_aug[n_x_ + 1] = 0;
  // std::cout<<"X_aug is: "<<x_aug<<std::endl;

  //create augmented covariance matrix
  P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
  P_aug.block<2,2>(n_x_, n_x_) << std_a_*std_a_, 0,
                        0, std_yawdd_*std_yawdd_;
  // std::cout<<"P_aug is: "<<P_aug<<std::endl;

  //create square root matrix
  MatrixXd P_aug_squareroot = P_aug.llt().matrixL();
  // std::cout<<"Square root matrix is: "<<P_aug_squareroot<<std::endl;

  double sqrt_lambda = sqrt(lambda_ + n_aug_);

  //2. Generate sigma points for step Xt (Use augmented sigma points to consider process noise)
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < P_aug_squareroot.cols(); i++) {
      Xsig_aug.col(i + 1) = x_aug + (sqrt_lambda*P_aug_squareroot.col(i));
      Xsig_aug.col(n_aug_ + i + 1) = x_aug - (sqrt_lambda*P_aug_squareroot.col(i));
  }
  // std::cout<<"Xsig_aug is: "<<Xsig_aug<<std::endl;

  //3. Calculate sigma points for step Xt+deltaT by passing them through F.
  //create matrix with predicted sigma points as columns
  Xsig_pred_.fill(0.0);

  //predict sigma points
  VectorXd current_sigma = VectorXd(n_aug_);
  current_sigma.fill(0.0);
  VectorXd current_pred_det = VectorXd(n_x_);
  current_pred_det.fill(0.0);
  VectorXd current_pred_stoc = VectorXd(n_x_);
  current_pred_stoc.fill(0.0);
  float v, psi, psi_dot, v_a, v_psi_dot2;
  double delta_t2 = pow(delta_t, 2);

  for (int i = 0; i < Xsig_aug.cols(); i++) {
      current_sigma = Xsig_aug.col(i);
      v = current_sigma[2];
      psi = current_sigma[3];
      psi_dot = current_sigma[4];
      v_a = current_sigma[5];
      v_psi_dot2 = current_sigma[6];
      current_pred_stoc << 0.5*delta_t2*cos(psi)*v_a,
                            0.5*delta_t2*sin(psi)*v_a,
                            delta_t*v_a,
                            0.5*delta_t2*v_psi_dot2,
                            delta_t*v_psi_dot2;
      // std::cout << "v_psi_dot2 = " <<v_psi_dot2 << std::endl;
      //avoid division by zero
      if (fabs(psi_dot) < 0.0001) {
        // std::cout<<"Entered fabs condition";
          current_pred_det << v*cos(psi)*delta_t,
                        v*sin(psi)*delta_t,
                        0,
                        0,
                        0;
      } else {
        // std::cout<<"Entered else condition";
          current_pred_det << (v/psi_dot)*(sin(psi + psi_dot*delta_t) - sin(psi)),
                            (v/psi_dot)*(-cos(psi + psi_dot*delta_t) + cos(psi)),
                            0,
                            psi_dot*delta_t,
                            0;
      }
      //write predicted sigma points into right column
      Xsig_pred_.col(i) = current_sigma.topRows(n_x_) + current_pred_det + current_pred_stoc;
  }
  // std::cout << "Xsig_pred_ = " <<Xsig_pred_ << std::endl;

  //4. Predict the mean and covariance of step Xt+deltaT

  //Calculation of weights
  int n_sigma = Xsig_pred_.cols();
//   std::cout<<"NSigma is: "<<n_sigma<<std::endl;
  float lamb_n_sig = lambda_ + n_aug_;
  for (int i = 1; i <= n_sigma; i++) {
      //set weights
      if (i == 1) {
          weights_[i - 1] = lambda_/(lamb_n_sig);
      } else {
          weights_[i - 1] = 1/(2*(lamb_n_sig));
      }
  }
  // std::cout<<"Weights are: "<<weights_<<std::endl;
  //Reset x_ and P_ to zero as we are using recursive summation
  x_.fill(0.0);
  P_.fill(0.0);
  //Calculation of predicted state x_
  for (int i = 0; i < n_sigma; i++) {
      //predict state mean
      x_ += weights_[i] * Xsig_pred_.col(i);
    //   std::cout<<"X is: "<<x<<std::endl;
  }

  //Calculation of state covariance matrix P_ with angle normalization
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
  // std::cout<<"x_ after pred: "<<x_<<std::endl;
  // std::cout<<"P_ after pred: "<<P_<<std::endl;
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
  2. Calculate NIS for Laser
  */

  //1. This process is Linear, hence use regular Kalman Filter equations.
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  //measurement covariance matrix - laser
  MatrixXd R_laser = MatrixXd(n_z, n_z);
  R_laser.fill(0.0);
  R_laser << std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  //Measurement function for laser
  MatrixXd H_laser = MatrixXd(n_z, n_x_);
  H_laser.fill(0.0);
  H_laser.row(0)[0] = 1;
  H_laser.row(1)[1] = 1;
  // std::cout << "H laser is: " << std::endl << H_laser << std::endl;

  VectorXd z = meas_package.raw_measurements_;
  // std::cout << "Z laser is: " << std::endl << z << std::endl;
  VectorXd y = z - H_laser * x_;
  // std::cout << "y laser is: " << std::endl << y << std::endl;
  MatrixXd H_laser_t = H_laser.transpose();
  MatrixXd S = (H_laser * P_ * H_laser_t) + R_laser;
  // std::cout << "S laser is: " << std::endl << S << std::endl;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * H_laser_t * Si;
  // std::cout << "K laser is: " << std::endl << K << std::endl;
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
  P_ = (I - (K * H_laser)) * P_;

  // 2. Calculate NIS for Laser
  
  nis_ = tools.CalculateNIS(y, Si);

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
  2. Make use of sigma points predicted in Predict step (Xsig_pred_). Transform them to measurement 
  space to get transformed sigma points(Zsig).
  3. Find mean and covariance to get vector z.
  4. Use Xsig_pred_, Zsig and z to find Kalman gain.
  5. Update x and P accordingly.
  6. Calculate NIS
  */

  //1. This process is non-linear, so use UKF here.
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  MatrixXd R_radar = MatrixXd(n_z, n_z);
  R_radar.fill(0.0);
  R_radar << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  // std::cout << "z_pred is: " << std::endl << z_pred << std::endl;
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  // std::cout << "S is: " << std::endl << S << std::endl;

  /*2. Make use of sigma points predicted in Predict step (Xsig_pred_). Transform them to measurement 
    space to get transformed sigma points(Zsig).*/
  VectorXd Xsig_col = VectorXd(Xsig_pred_.rows());
  Xsig_col.fill(0.0);
  // std::cout << "Xsig_col is: " << std::endl << Xsig_col << std::endl;
  VectorXd Zsig_col = VectorXd(n_z);
  Zsig_col.fill(0.0);
  // std::cout << "Zsig_col is: " << std::endl << Zsig_col << std::endl;
  float px, py, v, psi, sqrt_px2_py2;

  //transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
      Xsig_col = Xsig_pred_.col(i);
      px = Xsig_col[0];
      py = Xsig_col[1];
      v = Xsig_col[2];
      psi = Xsig_col[3];
      sqrt_px2_py2 = sqrt(pow(px, 2) + pow(py, 2));
      if (fabs(sqrt_px2_py2) < 0.0001) {
        sqrt_px2_py2 = 0.0001;
      }
      Zsig_col << sqrt_px2_py2,
                atan2(py, px),
                ((px*cos(psi)*v) + (py*sin(psi)*v))/sqrt_px2_py2;
      Zsig.col(i) = Zsig_col;
  }
    std::cout << "Zsig: " << std::endl << Zsig << std::endl;

  //3. Find mean and covariance to get vector z.
  for (int i = 0; i < Zsig.cols(); i++) {
      z_pred += weights_[i] * Zsig.col(i);
  }
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_radar;
  std::cout << "S: " << std::endl << S << std::endl;

  //4. Use Xsig_pred_, Zsig and z to find Kalman gain.
  //create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;
  std::cout << "z is: " << std::endl << z << std::endl;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //5. Update x and P accordingly.
  //update state mean and covariance matrix
  VectorXd y = z - z_pred;
  x_ += K * (y);
  P_ -= K * S * K.transpose();

  //6. Calculate NIS for Laser
  
  nis_ = tools.CalculateNIS(y, Si);
}

/**
 * Returns the NIS for current state of system after measurement
 */
float getNIS() {
  return nis_;
}
