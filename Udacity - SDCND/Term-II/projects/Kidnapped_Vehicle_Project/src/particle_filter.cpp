#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	int i;
	for (i = 0; i < num_particles; i++) {
	  Particle current_particle;
	  current_particle.id = i;
	  current_particle.x = dist_x(gen);
	  current_particle.y = dist_y(gen);
	  current_particle.theta = dist_theta(gen);
	  current_particle.weight = 1;
	  
	  particles.push_back(current_particle);
	  weights.push_back(current_particle.weight);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	
	int i;
	for (i = 0; i < num_particles; i++) {
	  double particle_x = particles[i].x;
	  double particle_y = particles[i].y;
	  double particle_theta = particles[i].theta;
	 
	  double pred_x;
	  double pred_y;
	  double pred_theta;
	  
	  //Instead of a hard check of 0, adding a check for very low value
	  if (fabs(yaw_rate < 0.0001)) {
	    pred_x += particle_x + velocity * cos(particle_theta);
	    pred_y = particle_y + velocity * sin(particle_theta);
	    pred_theta = particle_theta;
	  } else {
	    pred_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
	    pred_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
	    pred_theta = particle_theta + (yaw_rate * delta_t);
	  }
	  
	  normal_distribution<double> dist_x(pred_x, std_pos[0]);
	  normal_distribution<double> dist_y(pred_y, std_pos[1]);
	  normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
	  
	  particles[i].x = dist_x(gen);
	  particles[i].y = dist_y(gen);
	  particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	/*Associate observations in map co-ordinates to predicted landmarks using nearest neighbor algorithm. Here, the number of observations may
    be less than the total number of landmarks as some of the landmarks may be outside the range of vehicle's sensor.*/
  
  int i, j;
  for (i = 0; i < observations.size(); i++) {
    //Maximum distance can be square root of 2 times the range of sensor.
    double lowest_dist = sensor_range * sqrt(2);
    int closest_landmark_id = -1;
    double current_dist;
    LandmarkObs current_obs = observations[i];
    
    for (j = 0; j < predicted.size(); j++) {
      LandmarkObs current_pred = predicted[j];
      current_dist = dist(current_obs.x, current_obs.y, current_pred.x, current_pred.y);
      
      if (current_dist < lowest_dist) {
        lowest_dist = current_dist;
        closest_landmark_id = current_pred.id;
      }
    }
    current_obs.id = closet_landmark_id;
  }  

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  int i, j;
  for (i = 0; i < num_particles; i++) {
    //Cache current particle
    Particle current_particle = particles[i];
    //Vector containing observations transformed to map co-ordinates w.r.t. current particle.
    vector<LandmarkObs> transformed_observations;
    
    //Transform observations from vehicle's co-ordinates to map co-ordinates.
    for (j = 0; j < observations.size(); j++) {
      LandmarkObs transformed_obs;
      transformed_obs.id = j;
      transformed_obs.x = current_particle.x + (cos(current_particle.theta) * observations[j].x) - (sin(current_particle.theta) * observations[j].y);
      transformed_obs.y = current_particle.y + (sin(current_particle.theta) * observations[j].x) + (cos(current_particle.theta) * observations[j].y);
      transformed_observations.push_back(transformed_obs);
    }
    
    /*Filter map landmarks to keep only those which are in the sensor_range of current particle. Push them to predictions vector.*/
    vector<LandmarkObs> predicted_landmarks;
    for (j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((current_particle.x - current_landmark.x_f)) <= sensor_range) && (fabs((current_particle.y - current_landmark.y_f)) <= sensor_range)) {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }
    
    //Associate observations with predicted landmarks
    dataAssociation(predicted_landmarks, transformed_observations, sensor_range);
    
    //Reset the weight of particle to 1.
    current_particle.weight = 1;
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
    int k, l;
    
    /*Calculate the weight of particle based on the multivariate Gaussian probability function*/
    for (k = 0; k < transformed_observations.size(); k++) {
      LandmarkObs current_obs = transformed_observations[k];
      double multi_prob;
      
      for (l = 0; l < predicted_landmarks.size(); l++) {
        LandmarkObs current_pred = predicted_landmarks[l];
        
        if (current_obs.id == current_pred.id) {
          multi_prob = normalizer * exp(-1.0 * ((pow((current_obs.x - current_pred.x), 2)/(2.0 * sigma_x_2)) + (pow((current_obs.y - current_pred.y), 2)/(2.0 * sigma_y_2))));
          current_particle.weight *= multi_prob;
        }
      }
      cout<<"Weight is: "<<current_particle.weight<<endl;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
