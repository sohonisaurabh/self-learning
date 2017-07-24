#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //1. Estimation vector size should not be zero
  //2. Estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  for (unsigned int i = 0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
  /**
  TODO COMPLETED
  */
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  //state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float px2_py2 = pow(px, 2) + pow(py, 2);
  //check division by zero
  if(px2_py2 < 0.0001){
      cout << "CalculateJacobian () - Error - Division by Zero" << endl;
      px2_py2 = 0.0001;
  }
  float sqrt_px2_py2 = sqrt(px2_py2);
  float px2_py2_3 = (px2_py2*sqrt_px2_py2);


  //compute the Jacobian matrix
  Hj << (px/sqrt_px2_py2), (py/sqrt_px2_py2), 0, 0,
        -(py/px2_py2), (px/px2_py2), 0, 0,
        py*(vx*py - vy*px)/px2_py2_3, px*(px*vy - py*vx)/px2_py2_3, px/sqrt_px2_py2, py/sqrt_px2_py2;

  return Hj;
  /**
  TODO COMPLETED
  */
}
