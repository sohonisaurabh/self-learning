#include <iostream>
#include "ukf.h"

UKF::UKF() {
  //TODO Auto-generated constructor stub
  Init();
}

UKF::~UKF() {
  //TODO Auto-generated destructor stub
}

void UKF::Init() {

}


/*******************************************************************************
* Programming assignment functions: 
*******************************************************************************/

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

  //set state dimension
  int n_x = 5;

  //set augmented dimension
  int n_aug = 7;

  //create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
     Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  double delta_t = 0.1; //time diff in sec
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //predict sigma points
  VectorXd current_sigma;
  VectorXd current_pred_det = VectorXd(n_x);
  VectorXd current_pred_stoc = VectorXd(n_x);
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
      //avoid division by zero
      if (fabs(psi_dot) < 0.0001) {
          current_pred_det << v*cos(psi)*delta_t,
                        v*sin(psi)*delta_t,
                        0,
                        0,
                        0;
      } else {
          current_pred_det << (v/psi_dot)*(sin(psi + psi_dot*delta_t) - sin(psi)),
                            (v/psi_dot)*(-cos(psi + psi_dot*delta_t) + cos(psi)),
                            0,
                            psi_dot*delta_t,
                            0;
            
      }
      //write predicted sigma points into right column
      Xsig_pred.col(i) = current_sigma.topRows(n_x) + current_pred_det + current_pred_stoc;
  }
  

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;

}
