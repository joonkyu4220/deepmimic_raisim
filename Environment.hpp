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
      
      // MODE
      switchMode(cfg["preprocess"]);

      // WORLD
      world_ = std::make_unique<raisim::World>();
      if (visualizable_) {
        server_ = std::make_unique<raisim::RaisimServer>(world_.get());
        server_->launchServer();
      }
      world_->addGround(0, "steel");
      world_->setERP(1.0); // error reduction parameter
      // friction, restitution, resThreshold
      world_->setMaterialPairProp("default",  "ball", 1.0, 0.8, 0.0001); // 0.8 -> 0.0 for soft contact?
      world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
      world_->setMaterialPairProp("ball", "steel", 5.0, 0.85, 0.0001);

      // CHARACTER
      // character_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid.urdf");
      character_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid_dribble.urdf"); // larger, box-shaped hand with curved edges. wrists are spherical joints.
      character_->setName("character");
      if (visualizable_){
        server_->focusOn(character_);
      }
      /*
      root, chest, neck,
      right shoulder, right elbow, right wrist,
      left shoulder, left elbow, left wrist
      right hip, right knee, right ankle
      left hip, left knee, left ankle
      */
      gcDim_ = character_->getGeneralizedCoordinateDim(); // gcDim_ = 51 = 3 + 4 * 11(nSpherical) + 1 * 4(nRevolute)
      gvDim_ = character_->getDOF(); // gvDim_ = 40 = 3 + 3 * 11(nSpherical) + 1 * 4(nRevolute)
      nJoints_ = gvDim_ - 6; // no control over root pos/orn
      gc_.setZero(gcDim_);
      gv_.setZero(gvDim_);
      gc_init_.setZero(gcDim_);
      gv_init_.setZero(gvDim_);
      
      // CONTROLLER
      character_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      pTarget_.setZero(gcDim_);
      vTarget_.setZero(gvDim_); 

      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
      jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(250.0);
      jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(25.);
      jointPgain.segment(9, 3).setConstant(50.0); jointDgain.segment(9, 3).setConstant(5.0); // neck
      jointPgain.segment(30, 3).setConstant(150.0); jointDgain.segment(30, 3).setConstant(15.0); // right ankle
      jointPgain.segment(37, 3).setConstant(150.0); jointDgain.segment(37, 3).setConstant(15.0); // left ankle
      jointPgain.segment(12, 14).setConstant(50.0); jointDgain.segment(12, 14).setConstant(5.0); // right shoulder, elbow, left shoulder, elbow
      
      character_->setPdGains(jointPgain, jointDgain);
      character_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

      // CHARACTER
      posDim_ = 42; // 3 * 14
      comDim_ = 3;
      eeDim_ = 12;
      com_.setZero(3);
      ee_.setZero(12);

      // BALL
      ball_ = world_->addArticulatedSystem(resourceDir_ + "/ball3D.urdf");
      ball_->setName("ball");
      ball_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

      ball_gcDim_ = ball_->getGeneralizedCoordinateDim();
      ball_gvDim_ = ball_->getDOF();

      ball_gc_init_.setZero(ball_gcDim_);
      ball_gv_init_.setZero(ball_gvDim_);

      ball_gc_.setZero(ball_gcDim_);
      ball_gv_.setZero(ball_gvDim_);

      // DATA PREPARATION
      gc_ref_.setZero(gcDim_);
      gv_ref_.setZero(gvDim_);
      ee_ref_.setZero(eeDim_);
      com_ref_.setZero(comDim_);

      loadData();
      if (is_preprocess_) {
        preprocess();
      }
      loadGT();

      // AGENT
      stateDim_ = obDim_; // TODO
      stateDouble_.setZero(stateDim_); // TODO
      obDim_ = 168;
      /*
      joint positions 3 * 14
      joint linear velocities 3 * 14
      joint orientations 4 * 10(nSpherical) + 1 * 4(nRevolute)
      joint angular velocities 3 * 10(nSpherical) + 1 * 4(nRevolute)
      ball position 3
      ball velocity 3
      */
      obDouble_.setZero(obDim_);
      actionDim_ = gcDim_ - 7;
      rewards_.initializeFromConfigurationFile(cfg["reward"]);

      
    }

  void init() final {}

  void switchMode(const Yaml::Node& preprocess){
    is_preprocess_ = preprocess.template As<bool>();
  }

  void loadData(){
    data_gc_.setZero(maxLen_, gcDim_);
    std::ifstream gcfile(resourceDir_ + "/walk_gc.txt");
    float data;
    int row = 0, col = 0;
    while (gcfile >> data) {
      data_gc_.coeffRef(row, col) = data;
      col++;
      if (col == 20 || col == 29){ // skip the wrist joints
        col += 4;
      }
      if (col == gcDim_){
        col = 0;
        data_gc_.coeffRef(row, 20) = 1; data_gc_.coeffRef(row, 21) = 0; data_gc_.coeffRef(row, 22) = 0; data_gc_.coeffRef(row, 23) = 0;
        data_gc_.coeffRef(row, 29) = 1; data_gc_.coeffRef(row, 30) = 0; data_gc_.coeffRef(row, 31) = 0; data_gc_.coeffRef(row, 32) = 0;
        row++;
      }
    }
    dataLen_ = row; phase_speed_ = 1.0 / (float) dataLen_;
    data_gc_ = data_gc_.topRows(dataLen_);
  }

  void preprocess(){
    data_gv_.setZero(dataLen_, gvDim_);
    Eigen::MatrixXd data_pos_;
    data_pos_.setZero(dataLen_, posDim_);
    data_ee_.setZero(dataLen_, eeDim_);
    data_com_.setZero(dataLen_, comDim_);

    Mat<3, 3> rootRotInv;
    Vec<3> rootPos, jointPos_W, jointPos_B, comPos_W, comPos_B;

    // FK SOLVER
    
    for(int frameIdx = 0; frameIdx < dataLen_; frameIdx++) {
      character_->setState(data_gc_.row(frameIdx), gv_init_);
      character_->getState(gc_, gv_);
      getRootTransform(rootRotInv, rootPos);
      int posIdx = 0;
      for (int bodyIdx = 1; bodyIdx < 15; bodyIdx ++){
        character_->getBodyPosition(bodyIdx, jointPos_W);
        matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
        data_pos_.row(frameIdx).segment(posIdx, 3) = jointPos_B.e();
        posIdx += 3;
      }
      // end-effectors
      data_ee_.row(frameIdx).segment(0, 3) = data_pos_.row(frameIdx).segment(12, 3); // right wrists
      data_ee_.row(frameIdx).segment(3, 3) = data_pos_.row(frameIdx).segment(21, 3); // left wrist
      data_ee_.row(frameIdx).segment(6, 3) = data_pos_.row(frameIdx).segment(30, 3); // right ankle
      data_ee_.row(frameIdx).segment(9, 3) = data_pos_.row(frameIdx).segment(39, 3); // left ankle
      // center-of-mass
      comPos_W = character_->getCOM();
      matvecmul(rootRotInv, comPos_W - rootPos, comPos_B);
      data_com_.row(frameIdx).segment(0, 3) = comPos_B.e();
    }

    // CALCULATE ANGULAR VELOCITY
    Eigen::VectorXd prevFrame, nextFrame, prevGC, nextGC;
    for (int frameIdx = 0; frameIdx < dataLen_; frameIdx++){
      int prevFrameIdx = std::max(frameIdx - 1, 0);
      int nextFrameIdx = std::min(frameIdx + 1, dataLen_ - 1);
      Eigen::VectorXd prevFrame = data_gc_.row(prevFrameIdx), nextFrame = data_gc_.row(nextFrameIdx);
      float dt = (nextFrameIdx - prevFrameIdx) * control_dt_;

      // root has position
      int gcIdx = 0, gvIdx = 0;
      prevGC = prevFrame.segment(gcIdx, 3); nextGC = nextFrame.segment(gcIdx, 3);
      data_gv_.row(frameIdx).segment(gvIdx, 3) = (nextGC - prevGC) / dt;
      gcIdx += 3, gvIdx += 3;

      for (int jointIdx = 0; jointIdx < 15; jointIdx++){
        if (jointIdx == 4 || jointIdx == 7 || jointIdx == 10 || jointIdx == 13) {
          prevGC = prevFrame.segment(gcIdx, 1); nextGC = nextFrame.segment(gcIdx, 1);
          data_gv_.row(frameIdx).segment(gvIdx, 1) = (nextGC - prevGC) / dt;
          gcIdx += 1; gvIdx += 1;
        }
        else {
          prevGC = prevFrame.segment(gcIdx, 4); nextGC = nextFrame.segment(gcIdx, 4);
          data_gv_.row(frameIdx).segment(gvIdx, 3) = getAngularVelocity(prevGC, nextGC, dt);
          gcIdx += 4; gvIdx += 3;
        }
      }
    }

    std::ofstream gvFile(resourceDir_ + "/walk_gv_gt.txt", std::ios::out | std::ios::trunc);
    if (gvFile){
      gvFile << data_gv_;
      gvFile.close();
    }
    std::ofstream eeFile(resourceDir_ + "/walk_ee_gt.txt", std::ios::out | std::ios::trunc);
    if (eeFile){
      eeFile << data_ee_;
      eeFile.close();
    }
    std::ofstream comFile(resourceDir_ + "/walk_com_gt.txt", std::ios::out | std::ios::trunc);
    if (comFile){
      comFile << data_com_;
      comFile.close();
    }
  }

  Eigen::VectorXd getAngularVelocity(Eigen::VectorXd prev, Eigen::VectorXd next, float dt){
    Eigen::Quaterniond prevq, nextq, diffq;
    prevq.w() = prev(0); prevq.vec() = prev.segment(1, 3);
    nextq.w() = next(0); nextq.vec() = next.segment(1, 3);
    diffq = prevq.inverse() * nextq;
    float deg, gain;
    if (std::abs(diffq.w()) > 0.999999f) {
      gain = 0;
    }
    else if (diffq.w() < 0){
      deg = std::acos(-diffq.w());
      gain = -2.0f * deg / (std::sin(deg) * dt);
    }
    else{
      deg = std::acos(diffq.w());
      gain = 2.0f * deg / (std::sin(deg) * dt);
    }
    return gain * diffq.vec();
  }


  void loadGT(){
    data_gv_.setZero(dataLen_, gvDim_);
    std::ifstream gvfile(resourceDir_ + "/walk_gv_gt.txt");
    float data;
    int row = 0, col = 0;
    while (gvfile >> data) {
      data_gv_.coeffRef(row, col) = data;
      col++;
      if (col == gvDim_){
        col = 0;
        row++;
      }
    }

    data_ee_.setZero(dataLen_, eeDim_);
    std::ifstream eefile(resourceDir_ + "/walk_ee_gt.txt");
    row = 0, col = 0;
    while (eefile >> data) {
      data_ee_.coeffRef(row, col) = data;
      col++;
      if (col == eeDim_){
        col = 0;
        row++;
      }
    }

    data_com_.setZero(dataLen_, comDim_);
    std::ifstream comfile(resourceDir_ + "/walk_com_gt.txt");
    row = 0, col = 0;
    while (comfile >> data) {
      data_com_.coeffRef(row, col) = data;
      col++;
      if (col == comDim_){
        col = 0;
        row++;
      }
    }
  }

  void reset() final {
    sim_step_ = 0;
    total_reward_ = 0;
    
    // select random frame
    index_ = rand() % dataLen_;
    phase_ = index_ * phase_speed_;
    gc_init_ = data_gc_.row(index_);

    // right arm higher
    gc_init_[15] = 1; gc_ref_[16] = 0; gc_ref_[17] = 0; gc_ref_[18] = 0;
    gc_init_[19] = 1.57;
    gc_init_[20] = 0.707; gc_init_[21] = 0; gc_init_[22] = 0.707; gc_init_[23] = 0;
    
    pTarget_ << gc_init_;

    // TODO
    // gv_ref_.segment(0, gvDim_) = data_gv_.row(index_);
    character_->setState(gc_init_, gv_init_);
    
    // ball position initialization
    Vec<3> right_hand_pos;
    size_t right_hand_idx = character_->getFrameIdxByName("right_wrist"); // 9
    character_->getFramePosition(right_hand_idx, right_hand_pos);
    ball_gc_init_[0] = right_hand_pos[0];
    ball_gc_init_[1] = right_hand_pos[1];
    ball_gc_init_[2] = right_hand_pos[2] - 0.151; // ball 0.11, hand 0.04
    ball_gv_init_[2] = 0.05;
    ball_gc_init_[3] = 1;

    ball_->setState(ball_gc_init_, ball_gv_init_);

    from_ground_ = false;
    from_hand_ = false;
    is_ground_ = false;
    is_hand_ = false;
    ground_hand_ = false;

    contact_terminal_flag_ = false;

    updateObservation();
  }

  void updateObservation() {
    character_->getState(gc_, gv_);
    ball_->getState(ball_gc_, ball_gv_);

    Mat<3,3> rootRotInv;
    Vec<3> rootPos;
    getRootTransform(rootRotInv, rootPos);

    Vec<3> jointPos_W, jointPos_B, jointVel_W, jointVel_B, comPos_W, comPos_B;
    int obIdx = 0;
    int gcIdx = 7;
    int gvIdx = 6;
    int eeIdx = 0;

    // joint pos, orn, linvel, angvel
    for(size_t bodyIdx = 1; bodyIdx < 15; bodyIdx++){
      character_->getBodyPosition(bodyIdx, jointPos_W);
      character_->getVelocity(bodyIdx, jointVel_W);
      matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
      matvecmul(rootRotInv, jointVel_W, jointVel_B);

      obDouble_.segment(obIdx, 3) = jointPos_B.e();
      obIdx += 3;
      obDouble_.segment(obIdx, 3) = jointVel_B.e();
      obIdx += 3;

      if (bodyIdx == 4 || bodyIdx == 7 || bodyIdx == 10 || bodyIdx == 13) { // revolute jointIdx
        obDouble_.segment(obIdx, 1) = gc_.segment(gcIdx, 1);
        obIdx += 1; gcIdx += 1;
      }
      else {
        obDouble_.segment(obIdx, 4) = gc_.segment(gcIdx, 4);
        obIdx += 4; gcIdx += 4;
      }

      if (bodyIdx == 4 || bodyIdx == 7 || bodyIdx == 10 || bodyIdx == 13) { // revolute jointIdx
        obDouble_.segment(obIdx, 1) = gv_.segment(gvIdx, 1);
        obIdx += 1; gvIdx += 1;
      }
      else {
        obDouble_.segment(obIdx, 3) = gv_.segment(gvIdx, 3);
        obIdx += 3; gvIdx += 3;
      }

      // for ee reward
      if (bodyIdx == 5 || bodyIdx == 8 || bodyIdx == 11 || bodyIdx == 14){
        ee_.segment(eeIdx, 3) = jointPos_B.e();
        eeIdx += 3;
      }
    }

    // for com reward
    comPos_W = character_ -> getCOM();
    matvecmul(rootRotInv, comPos_W - rootPos, comPos_B);
    com_ = comPos_B.e();

    // ball pos, lin vel
    matvecmul(rootRotInv, ball_gc_.head(3) - rootPos.e(), jointPos_B);
    matvecmul(rootRotInv, ball_gv_.head(3), jointVel_B);
    obDouble_.segment(obIdx, 3) = jointPos_B.e();
    obIdx += 3;
    obDouble_.segment(obIdx, 3) = jointVel_B.e();

    // TODO
    // obDouble_.tail(1) << phase_;
  }

  void getRootTransform(Mat<3,3>& rot, Vec<3>& pos) {
    double yaw = atan2(2 * (gc_[3] * gc_[4] + gc_[5] * gc_[6]), 1 - 2 * (gc_[4] * gc_[4] + gc_[5] * gc_[5]));
    Vec<4> quat;
    quat[0] = cos(yaw / 2); quat[1] = 0; quat[2] = 0; quat[3] = - sin(yaw / 2);
    raisim::quatToRotMat(quat, rot);
    pos[0] = gc_[0]; pos[1] = gc_[1]; pos[2] = gc_[2];
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    gc_ref_ = data_gc_.row(index_);
    
    int actionIdx = 0;
    int targetIdx = 0;
    for (int jointIdx=1; jointIdx<15; jointIdx++)
    {
      targetIdx = actionIdx + 7;
      if (jointIdx == 4 || jointIdx == 7 || jointIdx == 10 || jointIdx == 13)
      {
        pTarget_.segment(targetIdx, 1) << pTarget_.segment(targetIdx + 7, 1) + action.cast<double>().segment(actionIdx, 1);
        // TODO
        // pTarget_.segment(actionIdx, 1) << action.cast<double>().segment(actionIdx, 1);
        actionIdx += 1;
      }
      else
      {
        pTarget_.segment(targetIdx, 4) << pTarget_.segment(targetIdx, 4) + action.cast<double>().segment(actionIdx, 4);
        // TODO
        // pTarget_.segment(actionIdx, 4) << action.cast<double>().segment(actionIdx, 4);
        pTarget_.segment(targetIdx, 4) << pTarget_.segment(targetIdx, 4).normalized();
        actionIdx += 4;
      }
    }

    character_->setPdTarget(pTarget_, vTarget_);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
    {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();

      for(auto& contact: ball_->getContacts()){
        if(contact.getPosition()[2] < 0.01)
        {
          if (from_ground_) {
            contact_terminal_flag_ = true;
            break;
          }
          if (is_hand_) {
            contact_terminal_flag_ = true;
            break;
          }
          is_ground_ = true;
          from_ground_ = true;
          from_hand_ = false;
        }
        else{
          auto& pair_contact = world_->getObject(contact.getPairObjectIndex())->getContacts()[contact.getPairContactIndexInPairObject()];
          if (character_->getBodyIdx("right_wrist") == pair_contact.getlocalBodyIndex()){
            if (is_ground_) {
              contact_terminal_flag_ = true;
              break;
            }
            if (from_ground_) {
              ground_hand_ = true;
            }
            is_hand_ = true;
            from_hand_ = true;
            from_ground_ = false;
          }
          else{
            contact_terminal_flag_ = true;
            break;
          }
        }
      }
    }

    is_hand_ = false;
    is_ground_ = false;

    index_ += 1;
    phase_ += phase_speed_;
    sim_step_ += 1;
    if (index_ >= dataLen_){
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
    // imitation reward
    double orn_err = 0, orn_reward = 0;
    double vel_err = 0, vel_reward = 0;
    double ee_err = 0, ee_reward = 0;
    double com_err = 0, com_reward = 0;

    Vec<4> quat, quat_ref, quat_err;
    Mat<3,3> mat, mat_ref, mat_err;

    for (size_t jointIdx = 0; jointIdx < 14; jointIdx++) {
      // masked
      if (jointIdx == 2 || jointIdx == 3 || jointIdx == 4)
      {
        continue;
      }
      if (jointIdx == 3 || jointIdx == 6 || jointIdx == 9 || jointIdx == 12)
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
    rewards_.record("orientation", orn_reward);

    vel_err = (gv_.tail(gvDim_ - 6) - data_gv_.row(index_).tail(gvDim_ - 6)).squaredNorm();
    vel_reward = exp(-0.1 * vel_err);
    rewards_.record("velocity", vel_reward);

    ee_err = (ee_ - data_ee_.row(index_)).squaredNorm();
    ee_reward = exp(-40 * ee_err);
    rewards_.record("end effector", ee_reward);

    com_err = (com_ - data_com_.row(index_)).squaredNorm();
    com_reward = exp(-10 * com_err);
    rewards_.record("com", com_reward);

    double ball_dist = (obDouble_[47] - obDouble_[162]) * (obDouble_[47] - obDouble_[162]) + (obDouble_[48] - obDouble_[163]) * (obDouble_[48] - obDouble_[163]);
    double dribble_reward = 0;
    double dist_reward = 0;
    if (ground_hand_) {
        dribble_reward += 1;
        ground_hand_ = false;
    }
    dist_reward += exp(-ball_dist);
    rewards_.record("dribble", dribble_reward);
    rewards_.record("ball distance", dist_reward);
  }
  
  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }

  void getState(Eigen::Ref<EigenVec> state) final {
    state = stateDouble_.cast<float>();
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
      return true;
    }
    
    // unwanted contact state
    if (contact_terminal_flag_) {
      return true;
    }

    // ball too far
    if ((obDouble_[47] - obDouble_[162]) * (obDouble_[47] - obDouble_[162]) + (obDouble_[48] - obDouble_[163]) * (obDouble_[48] - obDouble_[163]) > 1)
    {
      return true;
    }

    return false;
  }

  private:
    // MODE
    bool is_preprocess_ = false;

    bool visualizable_ = false;
    raisim::ArticulatedSystem* character_;
    raisim::ArticulatedSystem* ball_;

    int gcDim_, gvDim_, nJoints_, ball_gcDim_, ball_gvDim_;
    int posDim_, comDim_, eeDim_;
    
    Eigen::VectorXd ball_gc_, ball_gv_, ball_gc_init_, ball_gv_init_;
    Eigen::VectorXd gc_, gv_, gc_init_, gv_init_, gc_ref_, gv_ref_;
    
    Eigen::VectorXd com_, com_ref_, ee_, ee_ref_;

    int dataLen_;
    int maxLen_=1000;
    int index_ = 0;

    Eigen::MatrixXd data_gc_, data_gv_, data_ee_, data_com_;

    Eigen::VectorXd pTarget_, vTarget_;

    Eigen::VectorXd obDouble_, stateDouble_;

    float phase_ = 0;
    float phase_speed_ = 0;
    float max_phase_ = 1;
    int sim_step_ = 0;
    int max_sim_step_ = 2000;
    double total_reward_ = 0;

    // int joint_start_index[12] = {7, 11, 15, 19, 20, 24, 25, 29, 30, 34, 38, 39};
    int joint_start_index[14] = {7, 11, 15, 19, 20, 24, 28, 29, 33, 37, 38, 42, 46, 47};

    bool contact_terminal_flag_ = false;

    bool from_ground_ = false;
    bool from_hand_ = false;
    bool is_ground_ = false;
    bool is_hand_ = false;
    bool ground_hand_ = false;

};

}