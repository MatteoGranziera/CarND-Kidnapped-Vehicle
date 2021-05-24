/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  num_particles = 20; // TODO: increase num of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[3]);

  for(int i=0;i<num_particles;i++){
    Particle particle; 

    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[3]);

  for(int i = 0; i<num_particles; i++){

    // Check if yaw rate is negligible
    if(fabs(yaw_rate) > 0.00001){
      double dt_yaw = particles[i].theta + (delta_t * yaw_rate);

      particles[i].x +=  (velocity / yaw_rate) * (sin(dt_yaw) - sin(particles[i].theta));
      particles[i].y +=  (velocity / yaw_rate) * (cos(particles[i].theta) - cos(dt_yaw ));
      particles[i].theta += yaw_rate * delta_t;
    }
    else{
      particles[i].x += velocity * delta_t * cos( particles[i].theta );
      particles[i].y += velocity * delta_t * sin( particles[i].theta );
    }

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

for(int i=0; i<observations.size(); i++){
  int minId;
  double minDistance = numeric_limits<double>::max();

  for(int j=0; j<predicted.size();j++){

    double distance = dist(predicted[j].x, predicted[j].y,
                            observations[i].x, observations[i].y);

    if(distance < minDistance){
      minDistance = distance;
      minId = predicted[j].id;
    }
  }

  observations[i].id = minId;
}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  weights.clear();

  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  double normalizer = 0.0;

  for(int i=0; i<num_particles; i++){
    vector<LandmarkObs> inRangeLandmarks;

    for(int l=0; l<map_landmarks.landmark_list.size(); l++){
      double distance = dist(particles[i].x,particles[i].y,
                             map_landmarks.landmark_list[l].x_f,map_landmarks.landmark_list[l].y_f);
      // If is in sensor range 
      if(distance <= sensor_range){
        LandmarkObs landmark;

        landmark.id = map_landmarks.landmark_list[l].id_i;
        landmark.x = map_landmarks.landmark_list[l].x_f;
        landmark.y = map_landmarks.landmark_list[l].y_f;

        inRangeLandmarks.push_back(landmark);
      }
    }

    // transform observations coordinates

    vector<LandmarkObs> mappedObsevations;

    for(int l=0; l<observations.size(); l++) {
      LandmarkObs mappedObs;

      mappedObs.id = observations[l].id;
      mappedObs.x = particles[i].x + (cos(particles[i].theta) * observations[l].x) - (sin(particles[i].theta) * observations[l].y);
      mappedObs.y = particles[i].y + (sin(particles[i].theta) * observations[l].x) + (cos(particles[i].theta) * observations[l].y);

      mappedObsevations.push_back(mappedObs);
    }

    dataAssociation(inRangeLandmarks, mappedObsevations);

    particles[i].weight = 1.0;

    for(int l=0; l<mappedObsevations.size(); l++){

      LandmarkObs lmark = findLandmark(inRangeLandmarks, mappedObsevations[l].id);

      double x_diff = (mappedObsevations[l].x - lmark.x);
      double y_diff = (mappedObsevations[l].y - lmark.y);

      double exponent = (pow(x_diff, 2) / (2 * pow(std_x, 2))) + (pow(y_diff, 2) / (2 * (pow(std_y, 2))));
      double weight =  ( 1 / (2 * M_PI * std_x * std_y)) * exp(-exponent);

      if(weight == 0){
        particles[i].weight *= 0.000001;
      }
      else{
        particles[i].weight *= weight;
      }
    }

    weights.push_back(particles[i].weight);
    normalizer+=particles[i].weight;
  }

  // Normalize weights
  for(int i=0; i< weights.size(); i++){
    weights[i]= weights[i]/normalizer; 
  }

}

void ParticleFilter::resample() {
  // Resamplig wheel

  vector<Particle> resampled;

	discrete_distribution<int> particle_index(0, num_particles - 1);
	
	int index = particle_index(gen);
	
	double beta = 0;
	
  // Compute the max weight
  // use * operator to get the value of the pointer
  double mw = *max_element(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
    double rd = ((double) rand() / (RAND_MAX)) + 1 * 2.0 * mw;

		beta += rd;

	  while (beta > weights[index]) {
	    beta -= weights[index];
      

	    index = (index + 1) % num_particles;
	  }

	  resampled.push_back(particles[index]);
	}
	particles = resampled;


}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}