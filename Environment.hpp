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
      
      // SETTINGS
      setData(cfg["motion_data"], cfg["wrist"], cfg["control_dt"]);
      setPreprocess(cfg["preprocess"]);
      setPhaseUsage(cfg["use_char_phase"], cfg["use_ball_phase"]);
      setRewardScale(cfg["orn_scale"], cfg["vel_scale"], cfg["ee_scale"], cfg["com_scale"]);
      setTask(cfg["dribble_task"]);


      // WORLD SETUP
      world_ = std::make_unique<raisim::World>();
      if (visualizable_) {
        server_ = std::make_unique<raisim::RaisimServer>(world_.get());
        server_->launchServer();
      }
      world_->addGround(0, "steel");
      world_->setERP(1.0);
      // friction, restitution, resThreshold
      world_->setMaterialPairProp("default",  "ball", 1.0, 0.0, 0.0001); // 0.8 -> 0.0 for soft contact?
      world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
      world_->setMaterialPairProp("ball", "steel", 5.0, 0.85, 0.0001);


      // CHARACTER SETUP
      character_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid_dribble.urdf"); 
      character_->setName("character");
      if (visualizable_){
        server_->focusOn(character_);
      }

      gcDim_ = character_->getGeneralizedCoordinateDim(); // 51
      gc_.setZero(gcDim_);
      gc_init_.setZero(gcDim_);

      gvDim_ = character_->getDOF(); // 40
      gv_.setZero(gvDim_);
      gv_init_.setZero(gvDim_);
      
      com_.setZero(comDim_);
      ee_.setZero(eeDim_);
      

      // CONTROLLER SETUP
      controlDim_ = gvDim_ - 6; // no control over root pos/orn
      character_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      pTarget_.setZero(gcDim_);
      vTarget_.setZero(gvDim_);

      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
      jointPgain.setZero(); jointPgain.tail(controlDim_).setConstant(250.0);
      jointDgain.setZero(); jointDgain.tail(controlDim_).setConstant(25.);
      // neck
      jointPgain.segment(v_start_[1], v_dim_[1]).setConstant(50.0); jointDgain.segment(v_start_[1], v_dim_[1]).setConstant(5.0);
      // right ankle
      jointPgain.segment(v_start_[10], v_dim_[10]).setConstant(150.0); jointDgain.segment(v_start_[10], v_dim_[10]).setConstant(15.0);
      // left ankle
      jointPgain.segment(v_start_[13], v_dim_[13]).setConstant(150.0); jointDgain.segment(v_start_[13], v_dim_[13]).setConstant(15.0);

      // right arm
      jointPgain.segment(v_start_[2], v_dim_[2]).setConstant(50.0); jointDgain.segment(v_start_[2], v_dim_[2]).setConstant(5.0);
      jointPgain.segment(v_start_[3], v_dim_[3]).setConstant(50.0); jointDgain.segment(v_start_[3], v_dim_[3]).setConstant(5.0);
      jointPgain.segment(v_start_[4], v_dim_[4]).setConstant(50.0); jointDgain.segment(v_start_[4], v_dim_[4]).setConstant(5.0);
      // left arm
      jointPgain.segment(v_start_[5], v_dim_[5]).setConstant(50.0); jointDgain.segment(v_start_[5], v_dim_[5]).setConstant(5.0);
      jointPgain.segment(v_start_[6], v_dim_[6]).setConstant(50.0); jointDgain.segment(v_start_[6], v_dim_[6]).setConstant(5.0);
      jointPgain.segment(v_start_[7], v_dim_[7]).setConstant(50.0); jointDgain.segment(v_start_[7], v_dim_[7]).setConstant(5.0);
      
      character_->setPdGains(jointPgain, jointDgain);
      character_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

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

      // std::cout << "GC" << std::endl;
      // std::cout << "===================" << std::endl;
      // std::cout << data_gc_.row(38) << std::endl;

      // std::cout << "GV" << std::endl;
      // std::cout << "===================" << std::endl;
      // std::cout << data_gv_.row(38) << std::endl;

      // std::cout << "EE" << std::endl;
      // std::cout << "===================" << std::endl;
      // std::cout << data_ee_.row(38) << std::endl;

      // std::cout << "COM" << std::endl;
      // std::cout << "===================" << std::endl;
      // std::cout << data_com_.row(38) << std::endl;

      // AGENT
      stateDim_ = obDim_; // TODO
      stateDouble_.setZero(stateDim_); // TODO
      obDim_ = (posDim_) + (posDim_) + (gcDim_ - 7) + (gvDim_ - 6) + (ball_gcDim_ - 4) + (ball_gvDim_ - 3);
      if (use_ball_phase_) obDim_ += 2;
      if (use_char_phase_) obDim_ += 2;
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

  void setTask(const Yaml::Node& dribble_task)
  {
    dribble_ = dribble_task.template As<bool>();
  }

  void setData(const Yaml::Node& motion_data, const Yaml::Node& wrist, const Yaml::Node& dt){
    motion_data_ = motion_data.template As<std::string>();
    data_with_wrist_ = wrist.template As<bool>();
    control_dt_ = dt.template As<float>();
  }

  void setPreprocess(const Yaml::Node& preprocess){
    is_preprocess_ = preprocess.template As<bool>();
  }

  void setPhaseUsage(const Yaml::Node& char_phase, const Yaml::Node& ball_phase){
    use_char_phase_ = char_phase.template As<bool>();
    use_ball_phase_ = ball_phase.template As<bool>();
  }

  void setRewardScale(const Yaml::Node& orn_scale, const Yaml::Node& vel_scale, const Yaml::Node& ee_scale, const Yaml::Node& com_scale){
    orn_scale_ = orn_scale.template As<float>();
    vel_scale_ = vel_scale.template As<float>();
    ee_scale_ = ee_scale.template As<float>();
    com_scale_ = com_scale.template As<float>();
  }

  void loadData(){
    data_gc_.setZero(maxLen_, gcDim_);
    std::ifstream gcfile(resourceDir_ + "/" + motion_data_ + ".txt");
    float data;
    int row = 0, col = 0;
    while (gcfile >> data) {
      data_gc_.coeffRef(row, col) = data;
      col++;
      if (!data_with_wrist_ && (col == c_start_[4] || col == c_start_[7])){ // skip the wrist joints
        data_gc_.coeffRef(row, col) = 1; data_gc_.coeffRef(row, col + 1) = 0; data_gc_.coeffRef(row, col + 2) = 0; data_gc_.coeffRef(row, col + 3) = 0;
        col += 4;
      }
      if (col == gcDim_){
        col = 0;
        row++;
      }
    }
    dataLen_ = row;
    char_phase_speed_ = 2.0 * M_PI / (float) dataLen_;
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

    // SOLVE FK
    for(int frameIdx = 0; frameIdx < dataLen_; frameIdx++) {
      character_->setState(data_gc_.row(frameIdx), gv_init_);
      character_->getState(gc_, gv_);
      getRootTransform(rootRotInv, rootPos);
      
      // joint positions
      int posIdx = 0;
      for (int bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx ++){
        character_->getBodyPosition(bodyIdx, jointPos_W);
        matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
        data_pos_.row(frameIdx).segment(posIdx, 3) = jointPos_B.e();
        posIdx += 3;
      }

      // end-effectors
      data_ee_.row(frameIdx).segment(0, 3) = data_pos_.row(frameIdx).segment(ee_pos_start[0], 3); // right wrists
      data_ee_.row(frameIdx).segment(3, 3) = data_pos_.row(frameIdx).segment(ee_pos_start[1], 3); // left wrist
      data_ee_.row(frameIdx).segment(6, 3) = data_pos_.row(frameIdx).segment(ee_pos_start[2], 3); // right ankle
      data_ee_.row(frameIdx).segment(9, 3) = data_pos_.row(frameIdx).segment(ee_pos_start[3], 3); // left ankle
      
      // center-of-mass (world-frame!)
      comPos_W = character_->getCOM();
      data_com_.row(frameIdx).segment(0, 3) = comPos_W.e();
    }

    
    // CALCULATE ANGULAR VELOCITY
    Eigen::VectorXd prevFrame, nextFrame, prevGC, nextGC;
    for (int frameIdx = 0; frameIdx < dataLen_; frameIdx++){
      int prevFrameIdx = std::max(frameIdx - 1, 0);
      int nextFrameIdx = std::min(frameIdx + 1, dataLen_ - 1);
      Eigen::VectorXd prevFrame = data_gc_.row(prevFrameIdx), nextFrame = data_gc_.row(nextFrameIdx);
      float dt = (nextFrameIdx - prevFrameIdx) * control_dt_;

      int gcIdx = 0, gvIdx = 0;
      // root position
      prevGC = prevFrame.segment(gcIdx, 3); nextGC = nextFrame.segment(gcIdx, 3);
      data_gv_.row(frameIdx).segment(gvIdx, 3) = (nextGC - prevGC) / dt;
      gcIdx += 3, gvIdx += 3;
      // root orientation
      prevGC = prevFrame.segment(gcIdx, 4); nextGC = nextFrame.segment(gcIdx, 4);
      data_gv_.row(frameIdx).segment(gvIdx, 3) = getAngularVelocity(prevGC, nextGC, dt);
      gcIdx += 4; gvIdx += 3;

      for (int jointIdx = 0; jointIdx < nJoints_; jointIdx++){
        if (c_dim_[jointIdx] == 1) {
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
    n_loops_ = 0;
    loop_disp_acc_.setZero();
    raisim::quatToRotMat({1, 0, 0, 0}, loop_turn_acc_);
    total_reward_ = 0;

    // select random frame
    index_ = rand() % dataLen_;
    char_phase_ = index_ * char_phase_speed_;
    gc_init_ = data_gc_.row(index_);
    // fix right arm higher
    if (dribble_){
      gc_init_[c_start_[2]] = 1; gc_init_[c_start_[2] + 1] = 0; gc_init_[c_start_[2] + 2] = 0; gc_init_[c_start_[2] + 3] = 0;
      gc_init_[c_start_[3]] = 1.57;
      gc_init_[c_start_[4]] = 0.707; gc_init_[c_start_[4] + 1] = 0; gc_init_[c_start_[4] + 2] = 0.707; gc_init_[c_start_[4] + 3] = 0;
    }
    gv_init_ = data_gv_.row(index_);
    
    pTarget_ << gc_init_;

    character_->setState(gc_init_, gv_init_);
    
    // ball state initialization
    if (dribble_){
      Vec<3> right_hand_pos;
      size_t right_hand_idx = character_->getFrameIdxByName("right_wrist"); // 9
      character_->getFramePosition(right_hand_idx, right_hand_pos);
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


    // flags
    from_ground_ = false;
    from_hand_ = false;
    is_ground_ = false;
    is_hand_ = false;
    ground_hand_ = false;

    fall_flag_ = false;
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
    for(size_t bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx++){
      
      character_->getBodyPosition(bodyIdx, jointPos_W);
      matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
      obDouble_.segment(obIdx, 3) = jointPos_B.e();
      obIdx += 3;

      character_->getVelocity(bodyIdx, jointVel_W);
      matvecmul(rootRotInv, jointVel_W, jointVel_B);
      obDouble_.segment(obIdx, 3) = jointVel_B.e();
      obIdx += 3;

      if (c_dim_[bodyIdx - 1] == 1) { // revolute jointIdx
        obDouble_.segment(obIdx, 1) = gc_.segment(gcIdx, 1);
        obIdx += 1; gcIdx += 1;
        obDouble_.segment(obIdx, 1) = gv_.segment(gvIdx, 1);
        obIdx += 1; gvIdx += 1;
      }
      else {
        obDouble_.segment(obIdx, 4) = gc_.segment(gcIdx, 4);
        obIdx += 4; gcIdx += 4;
        obDouble_.segment(obIdx, 3) = gv_.segment(gvIdx, 3);
        obIdx += 3; gvIdx += 3;
      }

      // for ee reward
      if (is_ee_[bodyIdx - 1]){
        ee_.segment(eeIdx, 3) = jointPos_B.e();
        eeIdx += 3;
      }
    }

    // ball pos, lin vel
    matvecmul(rootRotInv, ball_gc_.head(3) - rootPos.e(), jointPos_B);
    matvecmul(rootRotInv, ball_gv_.head(3), jointVel_B);
    obDouble_.segment(obIdx, 3) = jointPos_B.e();
    obIdx += 3;
    obDouble_.segment(obIdx, 3) = jointVel_B.e();
    obIdx += 3;

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

    // for com reward
    comPos_W = character_ -> getCOM();
    com_ = comPos_W.e();

    // for ball dist reward
    ball_dist_ = (obDouble_[47] - obDouble_[162]) * (obDouble_[47] - obDouble_[162]) + (obDouble_[48] - obDouble_[163]) * (obDouble_[48] - obDouble_[163]);
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

    int actionIdx = 0;
    int controlIdx;
    for (int jointIdx=0; jointIdx < nJoints_; jointIdx++)
    {
      controlIdx = actionIdx + 7;
      if (c_dim_[jointIdx] == 1)
      {
        pTarget_.segment(controlIdx, 1) << pTarget_.segment(controlIdx, 1) + action.cast<double>().segment(actionIdx, 1);
      }
      else
      {
        pTarget_.segment(controlIdx, 4) << pTarget_.segment(controlIdx, 4) + action.cast<double>().segment(actionIdx, 4);
        pTarget_.segment(controlIdx, 4) << pTarget_.segment(controlIdx, 4).normalized();
      }
      actionIdx += c_dim_[jointIdx];
    }

    // character_->setPdTarget(pTarget_, vTarget_);
    character_->setState(data_gc_.row(index_), data_gv_.row(index_));

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
    {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();

      if (dribble_){
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

              // TODO: ball phase control
            }
            else{
              contact_terminal_flag_ = true;
              break;
            }
          }
        }
      }

      for(auto& contact: character_->getContacts()){
        if (contact.getPosition()[2] < 0.01){
          if ((contact.getlocalBodyIndex() != 11) && (contact.getlocalBodyIndex() != 14)){
            fall_flag_ = true;
            break;
          }
        }
      }
    }

    updateObservation();
    computeReward();

    is_hand_ = false;
    is_ground_ = false;

    if (gc_[2] < root_height_threshold_){
      fall_flag_ = true;
    }

    index_ += 1;
    char_phase_ += char_phase_speed_;
    sim_step_ += 1;
    if (index_ >= dataLen_){
      index_ = 0;
      char_phase_ = 0;
      n_loops_ += 1;

      Vec<3> temp_disp;
      Mat<3,3> temp_turn;
      matvecmul(loop_turn_acc_, loop_disp_, temp_disp);
      loop_disp_acc_ += temp_disp;
      temp_turn = loop_turn_acc_;
      raisim::matmul(loop_turn_, temp_turn, loop_turn_acc_);
    }

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

    for (size_t jointIdx = 0; jointIdx < nJoints_; jointIdx++) {
      // masked
      if (jointIdx == 2 || jointIdx == 3 || jointIdx == 4)
      {
        continue;
      }
      if (jointIdx == 3 || jointIdx == 6 || jointIdx == 9 || jointIdx == 12)
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

    vel_err = (gv_.tail(controlDim_) - data_gv_.row(index_).tail(controlDim_)).squaredNorm();
    vel_reward = exp(- vel_scale_ * vel_err);
    rewards_.record("velocity", vel_reward);

    ee_err = (ee_ - data_ee_.row(index_)).squaredNorm();
    ee_reward = exp(-ee_scale_ * ee_err);
    rewards_.record("end effector", ee_reward);

    com_ref_ = loop_turn_acc_ * data_com_.row(index_);
    com_ref_ = com_ref_ + loop_disp_acc_.e();
    // com_ref_ = data_com_.row(index_);
    // // TODO: appropriate root transformation for smooth looping should be determined at preprocessing step
    // com_ref_[0] += 1.2775 * n_loops_;

    com_err = (com_ - com_ref_).squaredNorm();
    com_reward = exp(-com_scale_ * com_err);
    rewards_.record("com", com_reward);
    
    double dribble_reward = 0;
    if (ground_hand_) {
        dribble_reward = 1;
        ground_hand_ = false;
    }
    rewards_.record("dribble", dribble_reward);

    double dist_reward = 0;
    dist_reward += exp(-ball_dist_);
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
    if (fall_flag_) {
      return true;
    }
    if (dribble_){
      // unwanted contact state
      if (contact_terminal_flag_) {
        return true;
      }

      // ball too far
      if (ball_dist_ > 1.0)
      {
        return true;
      }
    }
    return false;
  }

  private:

    bool dribble_ = false;

    bool is_preprocess_ = false;

    bool visualizable_ = false;
    raisim::ArticulatedSystem* character_;
    raisim::ArticulatedSystem* ball_;

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
    int max_sim_step_ = 1000;
    double total_reward_ = 0;

    int nJoints_ = 14;

    int posDim_ = 3 * nJoints_, comDim_ = 3, eeDim_ = 12;

    int c_start_[14] = {7, 11,  15, 19, 20,  24, 28, 29,  33, 37, 38,  42, 46, 47};
    int v_start_[14] = {6,  9,  12, 15, 16,  19, 22, 23,  26, 29, 30,  33, 36, 37};
    int c_dim_[14] = {4, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4};
    int v_dim_[14] = {3, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3};
    int is_ee_[14] = {0, 0,  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};

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