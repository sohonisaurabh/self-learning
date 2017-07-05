#include <iostream>
#include "Eigen/Dense"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {

	/*
	 * Compute the Jacobian Matrix
	 */

	//predicted state  example
	//px = 1, py = 2, vx = 0.2, vy = 0.4
	VectorXd x_predicted(4);
	x_predicted << 1, 2, 0.2, 0.4;

	MatrixXd Hj = CalculateJacobian(x_predicted);

	cout << "Hj:" << endl << Hj << endl;

	return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	float px2 = px*px;
	float py2 = py*py;
	float pxpysqrt = sqrt(px2 + py2);

	//TODO: YOUR CODE HERE 

	//check division by zero
	if (px == 0 && py == 0) {
	    cout << "Error, division by zero is not possible";
	    Hj << 0,0,0,0,
	           0,0,0,0,
	           0,0,0,0;
	} else {
	    Hj << (px/pxpysqrt), (py/pxpysqrt), 0, 0,
	           (-1*py/(px2 + py2)), (px/(px2 + py2)), 0, 0,
	           (py*(vx*py - vy*px)/pow(pxpysqrt, 3)), (px*(vy*px - vx*py)/pow(pxpysqrt, 3)), (px/pxpysqrt), (py/pxpysqrt);
	}
	
	//compute the Jacobian matrix

	return Hj;
}