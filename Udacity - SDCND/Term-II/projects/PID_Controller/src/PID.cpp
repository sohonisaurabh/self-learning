#include <uWS/uWS.h>
#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {
	tolerance = 0.005;
	delta_p.push_back(-0.01);
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	/*Set Kp, Ki and Kd to initial values passed by controller. These are passed from main*/
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;

	/*Set up inital p, i and d error to zero.*/
	p_error = 0;
	i_error = 0;
	d_error = 0;

}

void PID::UpdateError(double cte) {
	d_error = cte - p_error;
	p_error = cte;
	i_error += cte;
}

double PID::TotalError() {
	return ((Kp * p_error) + (Ki * i_error) + (Kd * d_error));
}

void PID::Restart(uWS::WebSocket<uWS::SERVER> ws) {
  std::string reset_msg = "42[\"reset\",{}]";
  ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
}

void PID::Twiddle(double total_error) {
  static double current_best_error = 100000;
  static bool is_twiddle_init = false;
  static bool is_twiddle_reset = false;
  static double last_Kp = 0;
  cout<<"Current best error is: "<< current_best_error<<endl;
  cout<<"Dp is: "<<delta_p[0]<<endl;
  if (!is_twiddle_init) {
  	cout<<"Twiddle init";
  	current_best_error = total_error;
  	is_twiddle_init = true;
  	return;
  }
  if ((fabs(delta_p[0]) > tolerance)) {
  	if (is_twiddle_reset) {
  		cout<<"Twiddle reset!-----------------------------"<<endl;
  		last_Kp = Kp;
  		Kp += delta_p[0];
  		cout<<"Kp magnitude increased!"<<endl;
  		is_twiddle_reset = false;
  	} else {
  		if (total_error < current_best_error) {
  			delta_p[0] *= 1.1;
  			is_twiddle_reset = true;
  			current_best_error = total_error;
  		} else {
  			if (fabs(last_Kp) < fabs(Kp)) {
  				last_Kp = Kp;
  				Kp -= 2.0 * delta_p[0];
  				cout<<"Kp magnitude decreased!"<<endl;
  			} else {
  				last_Kp = Kp;
  				Kp += delta_p[0];
  				delta_p[0] *= 0.9;
  				cout<<"Kp magnitude kept same!"<<endl;
  				is_twiddle_reset = true;
  			}
  		}
  	}
  }
}

