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

      setup(cfg);

      setWorld(visualizable_);

      setCharacter();
      
      setController();

      setBall();

      setData();

      setAgent();
    }

  void init() final {}

  void setup(const Yaml::Node& cfg){
    // EXPERIMENT SETTINGS

    motion_data_ = cfg["motion data"]["file name"].template As<std::string>();
    data_with_wrist_ = cfg["motion data"]["has wrist"].template As<bool>();
    control_dt_ = 1.0 / cfg["motion data"]["fps"].template As<float>();
    is_preprocess_ = cfg["motion data"]["preprocess"].template As<bool>();
    vis_kin_ =cfg["motion data"]["visualize kinematic"].template As<bool>();

    use_char_phase_ = cfg["phase usage"]["character"].template As<bool>();
    use_ball_phase_ = cfg["phase usage"]["ball"].template As<bool>();

    orn_scale_ = cfg["error sensitivity"]["orientation"].template As<float>();
    vel_scale_ = cfg["error sensitivity"]["velocity"].template As<float>();
    ee_scale_ = cfg["error sensitivity"]["end effector"].template As<float>();
    com_scale_ = cfg["error sensitivity"]["com"].template As<float>();

    dribble_ = cfg["task"]["dribble"].template As<bool>();
    use_ball_state_ = cfg["task"]["ball state"].template As<bool>();
    mask_ = cfg["task"]["mask"].template As<bool>();
  }

  void setWorld(bool visualizable){
    // PHYSICS WORLD SETUP
    world_ = std::make_unique<raisim::World>();
    if (visualizable) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
    }
    world_->addGround(0, "steel");
    world_->setERP(1.0);

    world_->setMaterialPairProp("default",  "ball", 1.0, 0.0, 0.0001); // 0.8 <-> 0.0 for hard/soft contact
    world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
    world_->setMaterialPairProp("ball", "steel", 5.0, 0.85, 0.0001);
  }

  void setCharacter(){
    // CHARACTER SETUP
    sim_character_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid_dribble.urdf"); 
    sim_character_->setName("sim character");
    if (visualizable_){
      server_->focusOn(sim_character_);
    }

    if (vis_kin_){
      kin_character_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid_dribble.urdf"); 
      kin_character_->setName("kin character");
    }

    gcDim_ = sim_character_->getGeneralizedCoordinateDim(); // 51
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_ref_.setZero(gcDim_);

    gvDim_ = sim_character_->getDOF(); // 40
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_ref_.setZero(gvDim_);
    
    com_.setZero(comDim_); com_ref_.setZero(comDim_);
    ee_.setZero(eeDim_); ee_ref_.setZero(eeDim_);
  }

  void setController(){
    // CONTROLLER SETUP
    controlDim_ = gvDim_ - 6; // no control over root pos/orn
    sim_character_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);

    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(controlDim_).setConstant(250.0);
    jointDgain.setZero(); jointDgain.tail(controlDim_).setConstant(25.);
    // neck
    jointPgain.segment(v_start_[neckIdx_], v_dim_[neckIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[neckIdx_], v_dim_[neckIdx_]).setConstant(5.0);
    // ankles
    jointPgain.segment(v_start_[rAnkleIdx_], v_dim_[rAnkleIdx_]).setConstant(150.0);
    jointDgain.segment(v_start_[rAnkleIdx_], v_dim_[rAnkleIdx_]).setConstant(15.0);
    jointPgain.segment(v_start_[lAnkleIdx_], v_dim_[lAnkleIdx_]).setConstant(150.0);
    jointDgain.segment(v_start_[lAnkleIdx_], v_dim_[lAnkleIdx_]).setConstant(15.0);
    // arms
    jointPgain.segment(v_start_[rShoulderIdx_], v_dim_[rShoulderIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[rShoulderIdx_], v_dim_[rShoulderIdx_]).setConstant(5.0);
    jointPgain.segment(v_start_[rElbowIdx_], v_dim_[rElbowIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[rElbowIdx_], v_dim_[rElbowIdx_]).setConstant(5.0);
    jointPgain.segment(v_start_[rWristIdx_], v_dim_[rWristIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[rWristIdx_], v_dim_[rWristIdx_]).setConstant(5.0);
    jointPgain.segment(v_start_[lShoulderIdx_], v_dim_[lShoulderIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[lShoulderIdx_], v_dim_[lShoulderIdx_]).setConstant(5.0);
    jointPgain.segment(v_start_[lElbowIdx_], v_dim_[lElbowIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[lElbowIdx_], v_dim_[lElbowIdx_]).setConstant(5.0);
    jointPgain.segment(v_start_[lWristIdx_], v_dim_[lWristIdx_]).setConstant(50.0);
    jointDgain.segment(v_start_[lWristIdx_], v_dim_[lWristIdx_]).setConstant(5.0);
    
    sim_character_->setPdGains(jointPgain, jointDgain);
    sim_character_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
  }

  void setBall(){
    // BALL
    ball_ = world_->addArticulatedSystem(resourceDir_ + "/ball3D.urdf");
    ball_->setName("ball");
    ball_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    ball_gcDim_ = ball_->getGeneralizedCoordinateDim();
    ball_gc_init_.setZero(ball_gcDim_);
    ball_gc_.setZero(ball_gcDim_);

    ball_gvDim_ = ball_->getDOF();
    ball_gv_init_.setZero(ball_gvDim_);
    ball_gv_.setZero(ball_gvDim_);
  }

  void setData(){
    // DATA PREPARATION
    loadData();
    if (is_preprocess_) {
      preprocess();
    }
    loadGT();
  }

  void setAgent(){
    // AGENT
    stateDim_ = obDim_; // TODO
    stateDouble_.setZero(stateDim_); // TODO
    obDim_ = (posDim_) + (posDim_) + (gcDim_ - 7) + (gvDim_ - 6);
    /*
    joint positions 3 * 14 || 0~41
    joint linear velocities 3 * 14 || 41~83
    joint orientations 4 * 10(nSpherical) + 1 * 4(nRevolute) || 84~127
    joint angular velocities 3 * 10(nSpherical) + 1 * 4(nRevolute) || 128~161
    */
    if (use_ball_state_) obDim_ += 6;
    if (use_ball_phase_) obDim_ += 2;
    if (use_char_phase_) obDim_ += 2;
    /*
    (optional)
    ball position 3 || 162~164
    ball velocity 3 || 165~167
    char phase 2
    ball phase 2
    */
    obDouble_.setZero(obDim_);
    actionDim_ = gcDim_ - 7; // TODO: 6D? quatToRotMat and rotMatToQuat seems handy
    rewards_.initializeFromConfigurationFile(cfg_["reward"]);
  }

  void loadData(){
    data_gc_.setZero(maxLen_, gcDim_);
    std::ifstream gcfile(resourceDir_ + "/" + motion_data_ + ".txt");
    float data;
    int row = 0, col = 0;
    while (gcfile >> data) {
      data_gc_.coeffRef(row, col) = data;
      col++;
      if (!data_with_wrist_ && (col == c_start_[rWristIdx_] || col == c_start_[lWristIdx_])){ // skip the wrist joints
        data_gc_.coeffRef(row, col) = 1; data_gc_.coeffRef(row, col + 1) = 0; data_gc_.coeffRef(row, col + 2) = 0; data_gc_.coeffRef(row, col + 3) = 0;
        col += 4;
      }
      if (col == gcDim_){
        col = 0;
        row++;
      }
    }
    dataLen_ = row;
    data_gc_ = data_gc_.topRows(dataLen_);
    char_phase_speed_ = 2.0 * M_PI / (float) dataLen_;
  }

  void preprocess(){
    data_gv_.setZero(dataLen_, gvDim_);
    data_ee_.setZero(dataLen_, eeDim_);
    data_com_.setZero(dataLen_, comDim_);

    Mat<3, 3> rootRotInv;
    Vec<3> rootPos, jointPos_W, jointPos_B, comPos_W, comPos_B;
    
    // SOLVE FK FOR EE & COM
    for(int frameIdx = 0; frameIdx < dataLen_; frameIdx++) {
      sim_character_->setState(data_gc_.row(frameIdx), gv_init_);
      sim_character_->getState(gc_, gv_);
      getRootTransform(rootRotInv, rootPos);
      
      int eeIdx = 0;
      for (int bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx ++){
        if (is_ee_[bodyIdx - 1]){
          sim_character_->getBodyPosition(bodyIdx, jointPos_W);
          matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
          data_ee_.row(frameIdx).segment(eeIdx, 3) = jointPos_B.e();
          eeIdx += 3;
        }
      }
      // center-of-mass (world-frame!)
      comPos_W = sim_character_->getCOM();
      data_com_.row(frameIdx).segment(0, 3) = comPos_W.e();
    }
    
    // CALCULATE ANGULAR VELOCITY FOR GV
    Eigen::VectorXd prevFrame, nextFrame, prevGC, nextGC;
    for (int frameIdx = 0; frameIdx < dataLen_; frameIdx++){
      int prevFrameIdx = std::max(frameIdx - 1, 0);
      int nextFrameIdx = std::min(frameIdx + 1, dataLen_ - 1);
      Eigen::VectorXd prevFrame = data_gc_.row(prevFrameIdx), nextFrame = data_gc_.row(nextFrameIdx);
      float dt = (nextFrameIdx - prevFrameIdx) * control_dt_;

      // root position
      prevGC = prevFrame.segment(0, 3); nextGC = nextFrame.segment(0, 3);
      data_gv_.row(frameIdx).segment(0, 3) = (nextGC - prevGC) / dt;
      // root orientation
      prevGC = prevFrame.segment(3, 4); nextGC = nextFrame.segment(3, 4);
      data_gv_.row(frameIdx).segment(3, 3) = getAngularVelocity(prevGC, nextGC, dt);

      for (int jointIdx = 0; jointIdx < nJoints_; jointIdx++){
        prevGC = prevFrame.segment(c_start_[jointIdx], c_dim_[jointIdx]);
        nextGC = nextFrame.segment(c_start_[jointIdx], c_dim_[jointIdx]);
        if (c_dim_[jointIdx] == 1) {
          data_gv_.row(frameIdx).segment(v_start_[jointIdx], v_dim_[jointIdx]) = (nextGC - prevGC) / dt;
        }
        else {
          data_gv_.row(frameIdx).segment(v_start_[jointIdx], v_dim_[jointIdx]) = getAngularVelocity(prevGC, nextGC, dt);
        }
      }
    }

    // WRITE FILES
    std::ofstream gvFile(resourceDir_ + "/" + motion_data_ + "_gv.txt", std::ios::out | std::ios::trunc);
    if (gvFile){
      gvFile << data_gv_;
      gvFile.close();
    }
    std::ofstream eeFile(resourceDir_ + "/" + motion_data_ + "_ee.txt", std::ios::out | std::ios::trunc);
    if (eeFile){
      eeFile << data_ee_;
      eeFile.close();
    }
    std::ofstream comFile(resourceDir_ + "/" + motion_data_ + "_com.txt", std::ios::out | std::ios::trunc);
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
    std::ifstream gvfile(resourceDir_ + "/" + motion_data_ + "_gv.txt");
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
    std::ifstream eefile(resourceDir_ + "/" + motion_data_ + "_ee.txt");
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
    std::ifstream comfile(resourceDir_ + "/" + motion_data_ + "_com.txt");
    row = 0, col = 0;
    while (comfile >> data) {
      data_com_.coeffRef(row, col) = data;
      col++;
      if (col == comDim_){
        col = 0;
        row++;
      }
    }

    // TODO: integrate loop transformation code
    std::ifstream loopdispfile(resourceDir_ + "/" + motion_data_ + "_loop_disp.txt");
    row = 0, col = 0;
    loop_disp_.setZero();
    while (loopdispfile >> data) {
      loop_disp_[col] = data;
      col++;
    }

    std::ifstream loopturnfile(resourceDir_ + "/" + motion_data_ + "_loop_turn.txt");
    Vec<4> loop_turn_quat;
    loop_turn_quat.setZero();
    loop_turn_quat[0] = 1;
    row = 0, col = 0;
    while (loopturnfile >> data) {
      loop_turn_quat[col] = data;
      col++;
    }
    raisim::quatToRotMat(loop_turn_quat, loop_turn_);
  }

  void reset() final {
    sim_step_ = 0;
    total_reward_ = 0;

    n_loops_ = 0;
    loop_disp_acc_.setZero();
    loop_turn_acc_.setIdentity(); // raisim::quatToRotMat({1, 0, 0, 0}, loop_turn_acc_);

    initializeCharacter();
    initializeBall();
    
    // flags
    resetFlags();

    updateObservation();
  }

  void initializeCharacter(){
    // select random frame
    index_ = rand() % dataLen_;
    char_phase_ = index_ * char_phase_speed_;
    gc_init_ = data_gc_.row(index_);
    // fix right arm higher
    gv_init_ = data_gv_.row(index_);
    

    // TODO: Noisier initialization scheme as the learning progresses
    if (dribble_){
      gc_init_[c_start_[2]] = 1; gc_init_[c_start_[2] + 1] = 0; gc_init_[c_start_[2] + 2] = 0; gc_init_[c_start_[2] + 3] = 0;
      gc_init_[c_start_[3]] = 1.57;
      gc_init_[c_start_[4]] = 0.707; gc_init_[c_start_[4] + 1] = 0; gc_init_[c_start_[4] + 2] = 0.707; gc_init_[c_start_[4] + 3] = 0;

      gv_init_[v_start_[2]] = 0; gv_init_[v_start_[2] + 1] = 0; gv_init_[v_start_[2] + 2] = 0;
      gv_init_[v_start_[3]] = 0;
      gv_init_[v_start_[4]] = 0; gv_init_[v_start_[4] + 1] = 0; gv_init_[v_start_[4] + 2] = 0;
    }
    
    pTarget_ << gc_init_;
    vTarget_.setZero();
    
    sim_character_->setState(gc_init_, gv_init_);
  }

  void initializeBall(){
    // ball state initialization
    if (dribble_){
      Vec<3> right_hand_pos;
      size_t right_hand_idx = sim_character_->getFrameIdxByName("right_wrist"); // 9
      sim_character_->getFramePosition(right_hand_idx, right_hand_pos);
      ball_gc_init_[0] = right_hand_pos[0];
      ball_gc_init_[1] = right_hand_pos[1];
      ball_gc_init_[2] = right_hand_pos[2] - 0.151; // ball 0.11, hand 0.04
      ball_gc_init_[3] = 1;
      ball_gv_init_[2] = 0.05;
    }
    else{
      ball_gc_init_[0] = 0; ball_gc_init_[1] = 100; ball_gc_init_[2] = 5; ball_gc_init_[3] = 1;
    }
    ball_->setState(ball_gc_init_, ball_gv_init_);
  }

  void resetFlags(){
    from_ground_ = false;
    from_hand_ = false;
    is_ground_ = false;
    is_hand_ = false;
    ground_hand_ = false;

    fall_flag_ = false;
    contact_terminal_flag_ = false;
  }

  void updateObservation() {
    sim_character_->getState(gc_, gv_);
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
    for(size_t bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx++){
      
      sim_character_->getBodyPosition(bodyIdx, jointPos_W);
      matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
      obDouble_.segment(obIdx, 3) = jointPos_B.e();
      obIdx += 3;

      sim_character_->getVelocity(bodyIdx, jointVel_W);
      matvecmul(rootRotInv, jointVel_W, jointVel_B);
      obDouble_.segment(obIdx, 3) = jointVel_B.e();
      obIdx += 3;

      obDouble_.segment(obIdx, c_dim_[bodyIdx - 1]) = gc_.segment(gcIdx, c_dim_[bodyIdx - 1]);
      obIdx += c_dim_[bodyIdx - 1]; gcIdx += c_dim_[bodyIdx - 1];
      obDouble_.segment(obIdx, v_dim_[bodyIdx - 1]) = gv_.segment(gvIdx, v_dim_[bodyIdx - 1]);
      obIdx += v_dim_[bodyIdx - 1]; gvIdx += v_dim_[bodyIdx - 1];

      // for ee reward. not recorded to observation
      if (is_ee_[bodyIdx - 1]){
        ee_.segment(eeIdx, 3) = jointPos_B.e();
        eeIdx += 3;
      }
    }

    if (use_ball_state_){
      if (dribble_){
        // ball pos, lin vel
        matvecmul(rootRotInv, ball_gc_.head(3) - rootPos.e(), jointPos_B);
        matvecmul(rootRotInv, ball_gv_.head(3), jointVel_B);
        obDouble_.segment(obIdx, 3) = jointPos_B.e();
        obIdx += 3;
        obDouble_.segment(obIdx, 3) = jointVel_B.e();
        obIdx += 3;
        ball_dist_ = (obDouble_[47] - obDouble_[162]) * (obDouble_[47] - obDouble_[162]) + (obDouble_[48] - obDouble_[163]) * (obDouble_[48] - obDouble_[163]);
      }
      else{ // for transfer learning?
        obDouble_.segment(obIdx, 6).setZero();
      }
    }

    // char phase
    if (use_char_phase_){
      obDouble_[obIdx] = std::cos(char_phase_);
      obDouble_[obIdx + 1] = std::sin(char_phase_);
      obIdx += 2;
    }

    // ball phase
    if (use_ball_phase_){
      obDouble_[obIdx] = std::cos(ball_phase_);
      obDouble_[obIdx + 1] = std::sin(ball_phase_);
      obIdx += 2;
    }

    // for com reward. not recorded to observation
    comPos_W = sim_character_ -> getCOM();
    com_ = comPos_W.e();
  }

  void getRootTransform(Mat<3,3>& rot, Vec<3>& pos) {
    Vec<4> rootRot, defaultRot, rootRotRel;
    rootRot[0] = gc_[3]; rootRot[1] = gc_[4]; rootRot[2] = gc_[5]; rootRot[3] = gc_[6];
    defaultRot[0] = 1 / sqrt(2); defaultRot[1] =  - 1 / sqrt(2); defaultRot[2] = 0; defaultRot[3] = 0;
    raisim::quatMul(defaultRot, rootRot, rootRotRel);
    double yaw = atan2(2 * (rootRotRel[0] * rootRotRel[2] + rootRotRel[1] * rootRotRel[3]), 1 - 2 * (rootRotRel[2] * rootRotRel[2] + rootRotRel[3] * rootRotRel[3]));
    Vec<4> quat;
    quat[0] = cos(yaw / 2); quat[1] = 0; quat[2] = 0; quat[3] = - sin(yaw / 2);
    raisim::quatToRotMat(quat, rot);
    pos[0] = gc_[0]; pos[1] = gc_[1]; pos[2] = gc_[2];
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    gc_ref_ = data_gc_.row(index_);
    gv_ref_ = data_gv_.row(index_);
    ee_ref_ = data_ee_.row(index_);
    com_ref_ = data_com_.row(index_);

    if (vis_kin_){
      Eigen::VectorXd kin_gc_, kin_gv_;
      kin_gc_ << gc_ref_; kin_gv_ << gv_ref_;
      kin_gc_[1] += 5;
      kin_character_->setState(kin_gc_, kin_gv_);
    }

    int actionIdx = 0;
    int controlIdx;
    for (int jointIdx = 0; jointIdx < nJoints_; jointIdx++)
    {
      controlIdx = actionIdx + 7;
      pTarget_.segment(controlIdx, c_dim_[jointIdx]) << pTarget_.segment(controlIdx, c_dim_[jointIdx]) + action.cast<double>().segment(actionIdx, c_dim_[jointIdx]);
      if (c_dim_[jointIdx] == 4){
        pTarget_.segment(controlIdx, c_dim_[jointIdx]) << pTarget_.segment(controlIdx, c_dim_[jointIdx]).normalized();
      }
      actionIdx += c_dim_[jointIdx];
    }

    sim_character_->setPdTarget(pTarget_, vTarget_);
    // sim_character_->setState(data_gc_.row(index_), data_gv_.row(index_));

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
    {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();

      if (dribble_){
        checkBallContact();
      }
      
      checkCharacterContact();
      
    }

    setBallPhaseSpeed();

    is_hand_ = false;
    is_ground_ = false;

    updateObservation();
    if (gc_[2] < root_height_threshold_){
      fall_flag_ = true;
    }

    computeReward();

    updateTargetMotion();

    double current_reward = rewards_.sum();
    total_reward_ += current_reward;

    return current_reward;
  }

  void checkBallContact(){
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
        if (sim_character_->getBodyIdx("right_wrist") == pair_contact.getlocalBodyIndex()){
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

  void checkCharacterContact(){
    for(auto& contact: sim_character_->getContacts()){
      if (contact.getPosition()[2] < 0.01){
        if ((contact.getlocalBodyIndex() != rAnkleIdx_ + 1) && (contact.getlocalBodyIndex() != lAnkleIdx_ + 1)){
          fall_flag_ = true;
          break;
        }
      }
    }
  }

  void setBallPhaseSpeed(){
    if (is_hand_){
      ball_phase_ = M_PI;
      float v = - std::abs(ball_gv_[2]);
      float h = std::abs(ball_gc_[2]);
      float g = std::abs(world_->getGravity()[2]);
      float t = (std::sqrt(v * v + 2 * g * h) - v) / g;
      ball_phase_speed_ = M_PI / t;
    }
    if (is_ground_){
      ball_phase_ = 0;
    }
  }

  void updateTargetMotion(){
    index_ += 1;
    char_phase_ += char_phase_speed_;
    ball_phase_ += ball_phase_speed_;
    sim_step_ += 1;
    if (index_ >= dataLen_){
      index_ = 0;
      char_phase_ = 0;
      n_loops_ += 1;
      Vec<3> temp_disp; Mat<3,3> temp_turn;
      matvecmul(loop_turn_acc_, loop_disp_, temp_disp);
      loop_disp_acc_ += temp_disp;
      temp_turn = loop_turn_acc_;
      raisim::matmul(loop_turn_, temp_turn, loop_turn_acc_);
    }
  }

  void computeReward() {
    // imitation reward
    double orn_err = 0, orn_reward = 0;
    double vel_err = 0, vel_reward = 0;
    double ee_err = 0, ee_reward = 0;
    double com_err = 0, com_reward = 0;

    Vec<4> quat, quat_ref, quat_err;
    Mat<3,3> mat, mat_ref, mat_err;

    for (size_t jointIdx = 0; jointIdx < nJoints_; jointIdx++) {
      if (mask_ && is_rightarm_[jointIdx])
      {
        continue;
      }
      if (c_dim_[jointIdx] == 1)
      {
        orn_err += std::pow(gc_[c_start_[jointIdx]] - gc_ref_[c_start_[jointIdx]], 2);
      }
      else {
        quat[0] = gc_[c_start_[jointIdx]]; quat[1] = gc_[c_start_[jointIdx]+1]; 
        quat[2] = gc_[c_start_[jointIdx]+2]; quat[3] = gc_[c_start_[jointIdx]+3];
        quat_ref[0] = gc_ref_[c_start_[jointIdx]]; quat_ref[1] = gc_ref_[c_start_[jointIdx]+1]; 
        quat_ref[2] = gc_ref_[c_start_[jointIdx]+2]; quat_ref[3] = gc_ref_[c_start_[jointIdx]+3];
        raisim::quatToRotMat(quat, mat);
        raisim::quatToRotMat(quat_ref, mat_ref);
        raisim::mattransposematmul(mat, mat_ref, mat_err);
        raisim::rotMatToQuat(mat_err, quat_err);
        orn_err += std::pow(acos(std::max(std::min(1.0, quat_err[0]), -1.0)) * 2, 2);
      }
    }

    orn_reward = exp(-orn_scale_ * orn_err);
    rewards_.record("orientation", orn_reward);

    if (mask_){
      vel_err = (gv_.segment(0, v_start_[rShoulderIdx_]) - gv_ref_.segment(0, v_start_[rShoulderIdx_])).squaredNorm();
      vel_err += (gv_.tail(gvDim_ - v_start_[lShoulderIdx_]) - gv_ref_.tail(gvDim_ - v_start_[lShoulderIdx_])).squaredNorm();
    }
    else{
      vel_err = (gv_.tail(controlDim_) - gv_ref_.tail(controlDim_)).squaredNorm();
    }
    vel_reward = exp(- vel_scale_ * vel_err);
    rewards_.record("velocity", vel_reward);

    if (mask_){
      ee_err = (ee_.segment(3, eeDim_ - 3) - ee_ref_.segment(3, eeDim_ - 3)).squaredNorm();
    }
    else{
      ee_err = (ee_ - ee_ref_).squaredNorm();
    }
    ee_reward = exp(- ee_scale_ * ee_err);
    rewards_.record("end effector", ee_reward);

    com_ref_ = loop_turn_acc_ * com_ref_;
    com_ref_ = com_ref_ + loop_disp_acc_.e();

    com_err = (com_ - com_ref_).squaredNorm();
    com_reward = exp(-com_scale_ * com_err);
    rewards_.record("com", com_reward);
    
    double contact_reward = 0;
    if (ground_hand_) {
        contact_reward = 1;
        ground_hand_ = false;
    }
    rewards_.record("contact", contact_reward);

    double dist_reward = 0;
    if (dribble_){
      dist_reward += exp(-ball_dist_);
    }
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

  int isTerminalState(float& terminalReward) final {

    if (time_limit_reached()) return 4;
    
    // low root height
    if (fall_flag_) {
      return 1;
    }
    if (dribble_){
      // unwanted contact state
      if (contact_terminal_flag_) {
        return 2;
      }

      // ball too far
      if (ball_dist_ > 1.0)
      {
        return 3;
      }
    }
    return 0;
  }

  private:

    bool dribble_ = false;

    bool is_preprocess_ = false;

    bool use_ball_state_ = false;
    bool mask_ = false;

    bool visualizable_ = false;

    bool vis_kin_ = false;
    raisim::ArticulatedSystem *sim_character_, *kin_character_;
    raisim::ArticulatedSystem *ball_;

    int gcDim_, gvDim_, controlDim_, ball_gcDim_, ball_gvDim_;
    
    Eigen::VectorXd ball_gc_, ball_gv_, ball_gc_init_, ball_gv_init_;
    Eigen::VectorXd gc_, gv_, gc_init_, gv_init_, gc_ref_, gv_ref_;
    
    Eigen::VectorXd com_, com_ref_, ee_, ee_ref_;

    float ball_dist_;

    int dataLen_;
    std::string motion_data_;
    int data_with_wrist_ = false;
    int maxLen_ = 1000;
    int index_ = 0;

    Eigen::MatrixXd data_gc_, data_gv_, data_ee_, data_com_;

    Eigen::VectorXd pTarget_, vTarget_;

    Eigen::VectorXd obDouble_, stateDouble_;

    int sim_step_ = 0;
    int max_sim_step_ = 100;
    double total_reward_ = 0;

    int nJoints_ = 14;

    int posDim_ = 3 * nJoints_, comDim_ = 3, eeDim_ = 12;

    int c_start_[14] = {7, 11,  15, 19, 20,  24, 28, 29,  33, 37, 38,  42, 46, 47};
    int v_start_[14] = {6,  9,  12, 15, 16,  19, 22, 23,  26, 29, 30,  33, 36, 37};
    int c_dim_[14] = {4, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4};
    int v_dim_[14] = {3, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3};
    int is_ee_[14] = {0, 0,  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};
    int is_rightarm_[14] = {0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0, 0, 0};
    
    int chestIdx_ = 0, neckIdx_ = 1;
    int rShoulderIdx_ = 2, rElbowIdx_ = 3, rWristIdx_ = 4;
    int lShoulderIdx_ = 5, lElbowIdx_ = 6, lWristIdx_ = 7;
    int rHipIdx_ = 8, rKneeIdx_ = 9, rAnkleIdx_ = 10;
    int lHipIdx_ = 11, lKneeIdx_ = 12, lAnkleIdx_ = 13;
    
    float orn_scale_, vel_scale_, ee_scale_, com_scale_;

    int ee_pos_start[4] = {12, 21, 30, 39};

    bool contact_terminal_flag_ = false;
    float root_height_threshold_ = 0.5;
    bool fall_flag_ = false;

    bool from_ground_ = false;
    bool from_hand_ = false;
    bool is_ground_ = false;
    bool is_hand_ = false;
    bool ground_hand_ = false;

    int n_loops_;

    bool use_char_phase_ = false;
    float char_phase_ = 0;
    float char_phase_speed_ = 0;
    float char_max_phase_ = M_PI * 2;
    bool use_ball_phase_ = false;
    float ball_phase_ = 0;
    float ball_phase_speed_ = 0;
    float ball_max_phase_ = M_PI * 2;

    Vec<3> loop_disp_;
    Mat<3,3> loop_turn_;

    Vec<3> loop_disp_acc_;
    Mat<3,3> loop_turn_acc_;
};

}