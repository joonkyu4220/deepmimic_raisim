#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {
  public:
    explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
        RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

      world_ = std::make_unique<raisim::World>();

      character_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid.urdf");
      character_->setName("character");
      character_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
      
      /// get robot data
      gcDim_ = character_->getGeneralizedCoordinateDim(); // gcDim_ = 43 = 3 + 4 + 4 + 4 + 4 + 1 + 4 + 1 + 4 + 1 + 4 + 4 + 1 + 4
      // root pos + root orn + chest + neck + right shoulder, elbow + left shoulder, elbow + right hip, knee, ankle + left hip, knee, ankle
      gvDim_ = character_->getDOF(); // gvDim_ = 34 = 3 + 3 + 3 + 3 + 3 + 1 + 3 + 1 + 3 + 1 + 3 + 3 + 1 + 3
      nJoints_ = gvDim_ - 6; // nJoints = 28 = 0 + 0 + 3 + 3 + 3 + 1 + 3 + 1 + 3 + 1 + 3 + 3 + 1 + 3

      gc_init_.setZero(gcDim_);
      gv_init_.setZero(gvDim_);

      gc_.setZero(gcDim_);
      gv_.setZero(gvDim_);

      world_->addGround(0, "steel");
      world_->setERP(1.0); // error reduction parameter
      // friction, restitution, resThreshold
      // TODO
      world_->setMaterialPairProp("default",  "ball", 1.0, 0.8, 0.0001);
      // world_->setMaterialPairProp("default",  "ball", 1.0, 0.0, 0.0001);
      world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
      world_->setMaterialPairProp("ball", "steel", 5.0, 0.85, 0.0001);
      

      ball_ = world_->addArticulatedSystem(resourceDir_ + "/ball3D.urdf");
      ball_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

      ball_gc_init_.setZero(7); ball_gc_init_[1] = 10; ball_gc_init_[2] = 1.5; ball_gc_init_[3] = 1; // put away the ball for now
      ball_gv_init_.setZero(6);
      ball_->setState(ball_gc_init_, ball_gv_init_);

      ball_gc_.setZero(7); ball_gc_[1] = 10; ball_gc_[3] = 1;
      ball_gv_.setZero(6);

      // ball radius: 0.11, hand radius 0.04
      ball_gc_ref_.setZero(7); ball_gc_ref_[1] = 10; ball_gc_ref_[2] = 1.5; ball_gc_ref_[3] = 1; // put away the ball for now
      ball_gv_ref_.setZero(6);

      pTarget_.setZero(gcDim_);
      vTarget_.setZero(gvDim_); 
      
      gc_ref_.setZero(gcDim_);
      gv_ref_.setZero(gvDim_);
      dataLen_ = 39;
      data_gc_ = Eigen::MatrixXd::Zero(dataLen_, gcDim_);
      data_gv_ = Eigen::MatrixXd::Zero(dataLen_, gvDim_);

      phase_speed_ = 1. / (double) dataLen_;

      read_data();

      gc_init_ <<
        0, 0, 1.707, // root pos 0
        0.707, 0.707, 0, 0, // root orn 3
        1, 0, 0, 0, // chest 7
        1, 0, 0, 0, // neck 11
        1, 0, 0, 0, // right shoulder 15
        0, // right elbow 19
        1, 0, 0, 0, // left shoulder 20
        0, // left elbow 24
        1, 0, 0, 0, // right hip 25
        0, // right knee 29
        1, 0, 0, 0, // right ankle 30
        1, 0, 0, 0, // left hip 34
        0, // left knee 38
        1, 0, 0, 0; // left ankle 39
      gc_ref_ <<
        0, 0, 1.707, // root pos
        0.707, 0.707, 0, 0, // root orn
        1, 0, 0, 0, // chest
        1, 0, 0, 0, // neck
        1, 0, 0, 0, // right shoulder
        0, // right elbow
        1, 0, 0, 0, // left shoulder
        0, // left elbow
        1, 0, 0, 0, // right hip
        0, // right knee
        1, 0, 0, 0, // right ankle
        1, 0, 0, 0, // left hip
        0, // left knee
        1, 0, 0, 0; // left ankle
      /// set pd gains
      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_); // 34 = 3 + 3 + 3 + 3 + 3 + 1 + 3 + 1 + 3 + 1 + 3 + 3 + 1 + 3
      jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(250.0);
      jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(25.);
      // jointPgain.setZero(); jointPgain.tail(gvDim_).setConstant(250.0);
      // jointDgain.setZero(); jointDgain.tail(gvDim_).setConstant(25.);
      jointPgain.segment(9, 3).setConstant(50.0); jointDgain.segment(9, 3).setConstant(5.0); // neck
      // NOTE: REDUNDANT
      // jointPgain.segment(12, 8).setConstant(100.0); jointDgain.segment(12, 8).setConstant(10.0); // right shoulder, elbow, left shoulder, elbow
      jointPgain.segment(24, 3).setConstant(150.0); jointDgain.segment(24, 3).setConstant(15.0); // right ankle
      jointPgain.segment(31, 3).setConstant(150.0); jointDgain.segment(31, 3).setConstant(15.0); // left ankle
      jointPgain.segment(12, 8).setConstant(50.0); jointDgain.segment(12, 8).setConstant(5.0); // right shoulder, elbow, left shoulder, elbow
      character_->setPdGains(jointPgain, jointDgain);
      character_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

      obDim_ = 143;
      // (3 * 12) + (3 * 12) + (4 * 8 + 1 * 4) + (3 * 8 + 1 * 4) + 3 + 3 + 1
      // (all in character base frame)
      // joint positions,
      // joint linear velocities, 
      // joint orientations,
      // joint angular velocities,
      // ball pos and vel
      // phase variable
      obDouble_.setZero(obDim_);

      actionDim_ = gcDim_;

      // TODO: how are observation and state different?
      stateDim_ = gcDim_ + 7;
      stateDouble_.setZero(stateDim_ * 2);

      rewards_.initializeFromConfigurationFile(cfg["reward"]);

      if (visualizable_) {
        server_ = std::make_unique<raisim::RaisimServer>(world_.get());
        server_->launchServer();
        // server_->focusOn(character_);
        server_->focusOn(ball_);
      }
    }

  void init() final {}

  void read_data(){
    std::ifstream gcfile(resourceDir_ + "/walk_gc.txt");
    float data;
    int i = 0, j = 0;
    while (gcfile >> data){
      data_gc_.coeffRef(j, i) = data;
      i++;
      if (i >= gcDim_){
        i = 0;
        j++;
      }
    }
    std::ifstream gvfile(resourceDir_ + "/walk_gv.txt");
    i = 0; j = 0;
    while (gvfile >> data){
      data_gv_.coeffRef(j, i) = data;
      i++;
      if (i >= gvDim_){
        i = 0;
        j++;
      }
    }
  }

  void reset() final {
    // std::cout << "RESET" << std::endl;
    sim_step_ = 0;
    total_reward_ = 0;
    
    index_ = rand() % dataLen_; // andom state initialization
    // index_ = 0; // fixed state initialization
    phase_ = index_ * phase_speed_;
    gc_ref_.segment(0, gcDim_) = data_gc_.row(index_);
    
    // right arm higher? 
    gc_ref_[19] = 1.7;

    pTarget_ << gc_ref_;

    // TODO
    // gv_ref_.segment(0, gvDim_) = data_gv_.row(index_);
    character_->setState(gc_ref_, gv_ref_);
    // BALL POS INITIALIZATION
    Vec<3> right_hand_pos;
    size_t right_hand_idx = character_->getFrameIdxByName("right_wrist"); // 9
    character_->getFramePosition(right_hand_idx, right_hand_pos);
    // std::cout << "RIGHT HAND IDX" << std::endl;
    // std::cout << right_hand_idx << std::endl;
    // std::cout << "RIGHT HAND POS" << std::endl;
    // std::cout << right_hand_pos << std::endl;
    ball_gc_init_[0] = right_hand_pos[0];
    ball_gc_init_[1] = right_hand_pos[1];
    ball_gc_init_[2] = right_hand_pos[2] - 0.151; // ball 0.11, hand 0.04
    // ball_gc_init_[1] = right_hand_pos[1] + 5;
    // ball_gc_init_[2] = right_hand_pos[2] + 10; // ball 0.11, hand 0.04

    ball_gv_init_[2] = 0.05;

    ball_->setState(ball_gc_init_, ball_gv_init_);

    from_ground_ = false;
    from_hand_ = false;
    ground_hand_ = false;

    contact_terminal_flag_ = false;

    updateObservation();
  }

  void updateObservation() {
    
    // std::cout << "GC_BEFORE: " << gc_ << std::endl;
    
    character_->getState(gc_, gv_);
    ball_->getState(ball_gc_, ball_gv_);

    // std::cout << "GC_AFTER: " << gc_ << std::endl;

    Mat<3,3> rootRotInv;
    Vec<3> rootPos;
    getRootTransform(rootRotInv, rootPos);

    Vec<3> jointPos_W, jointPos_B, jointVel_W, jointVel_B;
    int obIdx = 0;
    int gcIdx = 7;
    int gvIdx = 6;

    for(size_t bodyIdx = 1; bodyIdx < 13; bodyIdx++){
      character_->getBodyPosition(bodyIdx, jointPos_W);
      character_->getVelocity(bodyIdx, jointVel_W);
      matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
      matvecmul(rootRotInv, jointVel_W, jointVel_B);

      obDouble_.segment(obIdx, 3) = jointPos_B.e();
      obIdx += 3;
      obDouble_.segment(obIdx, 3) = jointVel_B.e();
      obIdx += 3;

      if (bodyIdx == 4 || bodyIdx == 6 || bodyIdx == 8 || bodyIdx == 11) { // revolute jointIdx
        obDouble_.segment(obIdx, 1) = gc_.segment(gcIdx, 1);
        obIdx += 1; gcIdx += 1;
      }
      else {
        obDouble_.segment(obIdx, 4) = gc_.segment(gcIdx, 4);
        obIdx += 4; gcIdx += 4;
      }

      if (bodyIdx == 4 || bodyIdx == 6 || bodyIdx == 8 || bodyIdx == 11) { // revolute jointIdx
        obDouble_.segment(obIdx, 1) = gv_.segment(gvIdx, 1);
        obIdx += 1; gvIdx += 1;
      }
      else {
        obDouble_.segment(obIdx, 3) = gv_.segment(gvIdx, 3);
        obIdx += 3; gvIdx += 3;
      }
      // std::cout << "OBIDX" << obIdx << std::endl;
    }
    
    // relative ball pos vel
    matvecmul(rootRotInv, ball_gc_.head(3) - rootPos.e(), jointPos_B);
    matvecmul(rootRotInv, ball_gv_.head(3), jointVel_B);
    obDouble_.segment(obIdx, 3) = jointPos_B.e();
    obIdx += 3;
    obDouble_.segment(obIdx, 3) = jointVel_B.e();

    obDouble_.tail(1) << phase_;
    // std::cout << "OB" << std::endl;
    // std::cout << obDouble_ << std::endl;
  }

  void getRootTransform(Mat<3,3>& rot, Vec<3>& pos) {
    double yaw = atan2(2 * (gc_[3] * gc_[4] + gc_[5] * gc_[6]), 1 - 2 * (gc_[4] * gc_[4] + gc_[5] * gc_[5]));
    Vec<4> quat;
    quat[0] = cos(yaw / 2); quat[1] = 0; quat[2] = 0; quat[3] = - sin(yaw / 2);
    raisim::quatToRotMat(quat, rot);
    pos[0] = gc_[0]; pos[1] = gc_[1]; pos[2] = gc_[2];
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    gc_ref_.segment(0, gcDim_) = data_gc_.row(index_);
    // gv_ref_.segment(0, gvDim_) = data_gv_.row(index_);
    
    int actionIdx = 3;
    for (int j=0; j<13; j++)
    {
      if (j == 4 || j == 6 || j == 8 || j == 11)
      {
        pTarget_.segment(actionIdx, 1) << pTarget_.segment(actionIdx, 1) + action.cast<double>().segment(actionIdx, 1);
        actionIdx += 1;
      }
      else
      {
        pTarget_.segment(actionIdx, 4) << pTarget_.segment(actionIdx, 4) + action.cast<double>().segment(actionIdx, 4);
        pTarget_.segment(actionIdx, 4) << pTarget_.segment(actionIdx, 4).normalized();
        actionIdx += 4;
      }
    }

    // PD controller of a floating joint?
    // pTarget_[0] = 20000;
    // pTarget_[1] = 0;
    // pTarget_[2] = 0;
    // pTarget_[3] = 1;
    // pTarget_[4] = 0;
    // pTarget_[5] = 0;
    // pTarget_[6] = 0;

    character_->setPdTarget(pTarget_, vTarget_);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
    {
      
      if (server_) server_->lockVisualizationServerMutex();
      // character_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();

      for(auto& contact: ball_->getContacts()){
        if(contact.getPosition()[2] < 0.01)
        {
          // std::cout << "GROUND" << std::endl;
          if (from_ground_) {
            contact_terminal_flag_ = true;
            break;
          }
          from_ground_ = true;
          from_hand_ = false;
        }
        else{
          auto& pair_contact = world_->getObject(contact.getPairObjectIndex())->getContacts()[contact.getPairContactIndexInPairObject()];
          if (character_->getBodyIdx("right_elbow") == pair_contact.getlocalBodyIndex()){
            // std::cout << "RIGHT HAND" << std::endl;
            // if (from_hand_) {
            //   contact_terminal_flag_ = true;
            //   break;
            // }
            if (from_ground_) {
              ground_hand_ = true;
            }
            from_ground_ = false;
            from_hand_ = true;
          }
          else{
            // std::cout << "OTHER BODY PART" << std::endl;
            contact_terminal_flag_ = true;
            break;
          }
        }
      }
    }

    index_ += 1;
    phase_ += phase_speed_;
    sim_step_ += 1;
    if (phase_ >= max_phase_){
      index_ = 0;
      phase_ = 0;
    }

    updateObservation();
    computeReward();

    double current_reward = rewards_.sum();
    total_reward_ += current_reward;
    return current_reward;
  }

  void computeReward() {
    // TODO
    // velocity reward
    // end effector position reward
    // center of mass position reward
    double orn_err = 0, orn_reward = 0; // , vel_err = 0, vel_reward = 0; ee_reward = 0, com_reward = 0
    double ball_dist = 0, ball_dist_reward = 0;
    double dribble_reward = 0;

    Vec<4> quat, quat_ref, quat_err;
    Mat<3,3> mat, mat_ref, mat_err;

    for (size_t jointIdx=0; jointIdx<12; jointIdx++) {
      // no right arm reward
      if (jointIdx == 2 || jointIdx == 3)
      {
        continue;
      }
      if (jointIdx == 3 || jointIdx == 5 || jointIdx == 7 || jointIdx == 10)
      {
        orn_err += std::pow(gc_[joint_start_index[jointIdx]] - gc_ref_[joint_start_index[jointIdx]], 2);
      }
      else {
        quat[0] = gc_[joint_start_index[jointIdx]]; quat[1] = gc_[joint_start_index[jointIdx]+1]; 
        quat[2] = gc_[joint_start_index[jointIdx]+2]; quat[3] = gc_[joint_start_index[jointIdx]+3];
        quat_ref[0] = gc_ref_[joint_start_index[jointIdx]]; quat_ref[1] = gc_ref_[joint_start_index[jointIdx]+1]; 
        quat_ref[2] = gc_ref_[joint_start_index[jointIdx]+2]; quat_ref[3] = gc_ref_[joint_start_index[jointIdx]+3];
        raisim::quatToRotMat(quat, mat);
        raisim::quatToRotMat(quat_ref, mat_ref);
        raisim::mattransposematmul(mat, mat_ref, mat_err);
        raisim::rotMatToQuat(mat_err, quat_err);
        orn_err += std::pow(acos(std::max(std::min(1.0, quat_err[0]), -1.0)) * 2, 2);
      }
    }
    orn_reward = exp(-2 * orn_err);

    ball_dist = (ball_gc_[0] - gc_[0]) * (ball_gc_[0] - gc_[0]) + (ball_gc_[1] - gc_[1]) * (ball_gc_[1] - gc_[1]);
    ball_dist_reward = exp(-ball_dist);

    if (ground_hand_) {
      dribble_reward += 1;
      ground_hand_ = false;
    }

    rewards_.record("orientation", orn_reward);
    rewards_.record("ball distance", ball_dist_reward);
    rewards_.record("dribble", dribble_reward);

    // TODO
    // vel_err += (gv_.segment(6, gvDim_ - 6) - gv_ref_.segment(6, gvDim_ - 6)).squaredNorm();
    // std::cout << "ORN_ERR: " << orn_err << std::endl;
    // std::cout << "VEL_ERR: " << vel_err << std::endl;
    // vel_reward += exp(-0.1 * vel_err);
    // rewards_.record("angular velocity", vel_reward);
  }
  
  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }

  void getState(Eigen::Ref<EigenVec> ob) final {
    stateDouble_ << gc_.tail(gcDim_), ball_gc_.tail(7), gc_ref_.tail(gcDim_), ball_gc_ref_.tail(7);
    ob = stateDouble_.cast<float>();
  }

  bool time_limit_reached() {
    return sim_step_ > max_sim_step_;
  }

  float get_total_reward() {
    return float(total_reward_);
  }

  bool isTerminalState(float& terminalReward) final {
    // low root height
    if (gc_[2] < 0.6) {
      // std::cout << "LOW ROOT FLAG" << std::endl;
      return true;
    }

    if (contact_terminal_flag_) {
      // std::cout << "CONTACT FLAG" << std::endl;
      return true;
    }

    if (ball_gc_[2] > 1.5){
      // std::cout << "BALL HEIGHT FLAG" << std::endl;
      return true;
    }

    if ((ball_gc_[0] - gc_[0]) * (ball_gc_[0] - gc_[0]) + (ball_gc_[1] - gc_[1]) * (ball_gc_[1] - gc_[1]) > 1)
    {
      // BALL TOO FAR
      return true;
    }

    return false;
  }

  private:
    bool visualizable_ = false;
    raisim::ArticulatedSystem* character_;
    raisim::ArticulatedSystem* ball_;

    int gcDim_, gvDim_, nJoints_;
    
    Eigen::VectorXd ball_gc_, ball_gv_, ball_gc_init_, ball_gv_init_, ball_gc_ref_, ball_gv_ref_;
    Eigen::VectorXd gc_, gv_, gc_init_, gv_init_, gc_ref_, gv_ref_;

    int dataLen_;
    int index_ = 0;
    Eigen::MatrixXd data_gc_, data_gv_;

    Eigen::VectorXd pTarget_, vTarget_;

    Eigen::VectorXd obDouble_, stateDouble_;

    float phase_ = 0;
    float phase_speed_ = 0;
    float max_phase_ = 1;
    int sim_step_ = 0;
    int max_sim_step_ = 1000;
    double total_reward_ = 0;

    int joint_start_index[12] = {7, 11, 15, 19, 20, 24, 25, 29, 30, 34, 38, 39};

    bool contact_terminal_flag_ = false;

    bool from_ground_ = false;
    bool from_hand_ = false;
    bool ground_hand_ = false;

};

}