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
      setWorld();
      setCharacter();
      setController();
      setBall();
      setData();
      setAgent(cfg);
    }

  void init() final {}

  void setup(const Yaml::Node& cfg){
    // EXPERIMENT SETTINGS
    charFileName_ = cfg["character"]["file name"].template As<std::string>();
    visKin_ = cfg["character"]["visualize kinematic"].template As<bool>();
    restitution_ = cfg["character"]["restitution"].template As<float>();

    motionFileName_ = cfg["motion data"]["file name"].template As<std::string>();
    dataHasWrist_ = cfg["motion data"]["has wrist"].template As<bool>();
    isPreprocess_ = cfg["motion data"]["preprocess"].template As<bool>();

    control_dt_ = 1.0 / cfg["motion data"]["fps"].template As<float>();
    simulation_dt_ = cfg["simulation_dt"].template As<float>();

    useCharPhase_ = cfg["phase usage"]["character"].template As<bool>();
    useBallPhase_ = cfg["phase usage"]["ball"].template As<bool>();

    ornScale_ = cfg["error sensitivity"]["orientation"].template As<float>();
    velScale_ = cfg["error sensitivity"]["velocity"].template As<float>();
    eeScale_ = cfg["error sensitivity"]["end effector"].template As<float>();
    comScale_ = cfg["error sensitivity"]["com"].template As<float>();
    energyScale_ = cfg["error sensitivity"]["energy efficiency"].template As<float>();

    dribble_ = cfg["task"]["dribble"].template As<bool>();
    useBallState_ = cfg["task"]["ball state"].template As<bool>();
    mask_ = cfg["task"]["mask"].template As<bool>();
  }

  void setWorld(){
    world_ = std::make_unique<raisim::World>();
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
    }
    world_->addGround(0, "steel");
    world_->setERP(1.0);
    world_->setMaterialPairProp("default",  "ball", 1.0, restitution_, 0.0001);
    world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
    world_->setMaterialPairProp("ball", "steel", 5.0, 0.85, 0.0001);
  }

  void setCharacter(){
    // CHARACTER SETUP
    simChar_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid_dribble.urdf"); 
    simChar_->setName("sim character");
    if (visualizable_){
      server_->focusOn(simChar_);
    }

    if (visKin_){
      kinChar_ = world_->addArticulatedSystem(resourceDir_ + "/humanoid_dribble.urdf"); 
      kinChar_->setName("kin character");
    }

    gcDim_ = simChar_->getGeneralizedCoordinateDim(); // 51
    gc_.setZero(gcDim_); gcInit_.setZero(gcDim_); gcRef_.setZero(gcDim_);

    gvDim_ = simChar_->getDOF(); // 40
    controlDim_ = gvDim_ - 6;
    gv_.setZero(gvDim_); gvInit_.setZero(gvDim_); gvRef_.setZero(gvDim_);

    prevGV_.setZero(gvDim_);
    
    com_.setZero(comDim_); comRef_.setZero(comDim_);
    ee_.setZero(eeDim_); eeRef_.setZero(eeDim_);

  }

  void setController(){
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    
    jointPgain.setZero(); jointPgain.tail(controlDim_).setConstant(250.0);
    jointDgain.setZero(); jointDgain.tail(controlDim_).setConstant(25.);

    jointPgain.segment(vStart_[neckIdx_], vDim_[neckIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[neckIdx_], vDim_[neckIdx_]).setConstant(5.0);

    jointPgain.segment(vStart_[rAnkleIdx_], vDim_[rAnkleIdx_]).setConstant(150.0);
    jointDgain.segment(vStart_[rAnkleIdx_], vDim_[rAnkleIdx_]).setConstant(15.0);
    jointPgain.segment(vStart_[lAnkleIdx_], vDim_[lAnkleIdx_]).setConstant(150.0);
    jointDgain.segment(vStart_[lAnkleIdx_], vDim_[lAnkleIdx_]).setConstant(15.0);
    jointPgain.segment(vStart_[rShoulderIdx_], vDim_[rShoulderIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[rShoulderIdx_], vDim_[rShoulderIdx_]).setConstant(5.0);
    jointPgain.segment(vStart_[rElbowIdx_], vDim_[rElbowIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[rElbowIdx_], vDim_[rElbowIdx_]).setConstant(5.0);
    jointPgain.segment(vStart_[rWristIdx_], vDim_[rWristIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[rWristIdx_], vDim_[rWristIdx_]).setConstant(5.0);
    jointPgain.segment(vStart_[lShoulderIdx_], vDim_[lShoulderIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[lShoulderIdx_], vDim_[lShoulderIdx_]).setConstant(5.0);
    jointPgain.segment(vStart_[lElbowIdx_], vDim_[lElbowIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[lElbowIdx_], vDim_[lElbowIdx_]).setConstant(5.0);
    jointPgain.segment(vStart_[lWristIdx_], vDim_[lWristIdx_]).setConstant(50.0);
    jointDgain.segment(vStart_[lWristIdx_], vDim_[lWristIdx_]).setConstant(5.0);
    
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_); 
    
    simChar_->setPdGains(jointPgain, jointDgain);
    simChar_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
  }

  void setBall(){
    ball_ = world_->addArticulatedSystem(resourceDir_ + "/basketball.urdf");
    ball_->setName("ball");
    ball_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    ballGCDim_ = ball_->getGeneralizedCoordinateDim();
    ballGVDim_ = ball_->getDOF();

    ballGCInit_.setZero(ballGCDim_);
    ballGVInit_.setZero(ballGVDim_);

    ballGC_.setZero(ballGCDim_);
    ballGV_.setZero(ballGVDim_);
  }
  
  void setData(){
    // DATA PREPARATION
    loadData();
    if (isPreprocess_) {
      preprocess();
    }
    loadGT();
  }

  void setAgent(const Yaml::Node& cfg){
    obDim_ = (posDim_) + (posDim_) + (gcDim_ - 7) + (gvDim_ - 6);
    /*
    joint positions 3 * 14 || 0~41
    joint linear velocities 3 * 14 || 41~83
    joint orientations 4 * 10(nSpherical) + 1 * 4(nRevolute) || 84~127
    joint angular velocities 3 * 10(nSpherical) + 1 * 4(nRevolute) || 128~161
    */
    if (useBallState_) obDim_ += 6;
    if (useBallPhase_) obDim_ += 2;
    if (useCharPhase_) obDim_ += 2;
    /*
    (optional)
    ball position 3 || 162~164
    ball velocity 3 || 165~167
    char phase 2
    ball phase 2
    */
    obDouble_.setZero(obDim_);

    stateDim_ = obDim_; // TODO
    stateDouble_.setZero(stateDim_); // TODO

    actionDim_ = gcDim_ - 7; // TODO: 6D? quatToRotMat and rotMatToQuat seems handy / or additional joint torques?

    rewards_.initializeFromConfigurationFile(cfg_["reward"]);
  }

  void loadData(){
    dataGC_.setZero(maxLen_, gcDim_);
    std::ifstream gcfile(resourceDir_ + "/" + motionFileName_ + ".txt");
    float data;
    int row = 0, col = 0;
    while (gcfile >> data) {
      dataGC_.coeffRef(row, col) = data;
      col++;
      if (!dataHasWrist_ && (col == cStart_[rWristIdx_] || col == cStart_[lWristIdx_])){ // skip the wrist joints
        dataGC_.coeffRef(row, col) = 1; dataGC_.coeffRef(row, col + 1) = 0; dataGC_.coeffRef(row, col + 2) = 0; dataGC_.coeffRef(row, col + 3) = 0;
        col += 4;
      }
      if (col == gcDim_){
        col = 0;
        row++;
      }
    }
    dataLen_ = row;
    dataGC_ = dataGC_.topRows(dataLen_);
    charPhaseSpeed_ = 2.0 * M_PI / (float) dataLen_;
  }

  void preprocess(){
    dataGV_.setZero(dataLen_, gvDim_);
    dataEE_.setZero(dataLen_, eeDim_);
    dataCom_.setZero(dataLen_, comDim_);

    Mat<3, 3> rootRotInv;
    Vec<3> rootPos, jointPos_W, jointPos_B, comPos_W, comPos_B;
    
    // SOLVE FK FOR EE & COM
    for(int frameIdx = 0; frameIdx < dataLen_; frameIdx++) {
      simChar_->setState(dataGC_.row(frameIdx), gvInit_);
      simChar_->getState(gc_, gv_);
      getRootTransform(rootRotInv, rootPos);
      
      int eeIdx = 0;
      for (int bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx ++){
        if (isEE_[bodyIdx - 1]){
          simChar_->getBodyPosition(bodyIdx, jointPos_W);
          matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
          dataEE_.row(frameIdx).segment(eeIdx, 3) = jointPos_B.e();
          eeIdx += 3;
        }
      }
      // center-of-mass (world-frame!)
      comPos_W = simChar_->getCOM();
      dataCom_.row(frameIdx).segment(0, 3) = comPos_W.e();
    }
    
    // CALCULATE ANGULAR VELOCITY FOR GV
    Eigen::VectorXd prevFrame, nextFrame, prevGC, nextGC;
    for (int frameIdx = 0; frameIdx < dataLen_; frameIdx++){
      int prevFrameIdx = std::max(frameIdx - 1, 0);
      int nextFrameIdx = std::min(frameIdx + 1, dataLen_ - 1);
      Eigen::VectorXd prevFrame = dataGC_.row(prevFrameIdx), nextFrame = dataGC_.row(nextFrameIdx);
      float dt = (nextFrameIdx - prevFrameIdx) * control_dt_;

      // root position
      prevGC = prevFrame.segment(0, 3); nextGC = nextFrame.segment(0, 3);
      dataGV_.row(frameIdx).segment(0, 3) = (nextGC - prevGC) / dt;
      // root orientation
      prevGC = prevFrame.segment(3, 4); nextGC = nextFrame.segment(3, 4);
      dataGV_.row(frameIdx).segment(3, 3) = getAngularVelocity(prevGC, nextGC, dt);

      for (int jointIdx = 0; jointIdx < nJoints_; jointIdx++){
        prevGC = prevFrame.segment(cStart_[jointIdx], cDim_[jointIdx]);
        nextGC = nextFrame.segment(cStart_[jointIdx], cDim_[jointIdx]);
        if (cDim_[jointIdx] == 1) {
          dataGV_.row(frameIdx).segment(vStart_[jointIdx], vDim_[jointIdx]) = (nextGC - prevGC) / dt;
        }
        else {
          dataGV_.row(frameIdx).segment(vStart_[jointIdx], vDim_[jointIdx]) = getAngularVelocity(prevGC, nextGC, dt);
        }
      }
    }

    // WRITE FILES
    std::ofstream gvFile(resourceDir_ + "/" + motionFileName_ + "_gv.txt", std::ios::out | std::ios::trunc);
    if (gvFile){
      gvFile << dataGV_;
      gvFile.close();
    }
    std::ofstream eeFile(resourceDir_ + "/" + motionFileName_ + "_ee.txt", std::ios::out | std::ios::trunc);
    if (eeFile){
      eeFile << dataEE_;
      eeFile.close();
    }
    std::ofstream comFile(resourceDir_ + "/" + motionFileName_ + "_com.txt", std::ios::out | std::ios::trunc);
    if (comFile){
      comFile << dataCom_;
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
    dataGV_.setZero(dataLen_, gvDim_);
    std::ifstream gvfile(resourceDir_ + "/" + motionFileName_ + "_gv.txt");
    float data;
    int row = 0, col = 0;
    while (gvfile >> data) {
      dataGV_.coeffRef(row, col) = data;
      col++;
      if (col == gvDim_){
        col = 0;
        row++;
      }
    }

    dataEE_.setZero(dataLen_, eeDim_);
    std::ifstream eefile(resourceDir_ + "/" + motionFileName_ + "_ee.txt");
    row = 0, col = 0;
    while (eefile >> data) {
      dataEE_.coeffRef(row, col) = data;
      col++;
      if (col == eeDim_){
        col = 0;
        row++;
      }
    }

    dataCom_.setZero(dataLen_, comDim_);
    std::ifstream comfile(resourceDir_ + "/" + motionFileName_ + "_com.txt");
    row = 0, col = 0;
    while (comfile >> data) {
      dataCom_.coeffRef(row, col) = data;
      col++;
      if (col == comDim_){
        col = 0;
        row++;
      }
    }

    // TODO: integrate loop transformation code
    std::ifstream loopdispfile(resourceDir_ + "/" + motionFileName_ + "_loop_disp.txt");
    row = 0, col = 0;
    loopDisplacement_.setZero();
    while (loopdispfile >> data) {
      loopDisplacement_[col] = data;
      col++;
    }

    std::ifstream loopturnfile(resourceDir_ + "/" + motionFileName_ + "_loop_turn.txt");
    Vec<4> loop_turn_quat;
    loop_turn_quat.setZero();
    loop_turn_quat[0] = 1;
    row = 0, col = 0;
    while (loopturnfile >> data) {
      loop_turn_quat[col] = data;
      col++;
    }
    raisim::quatToRotMat(loop_turn_quat, loopTurn_);
  }
  
  void read_data(){
    std::ifstream gcfile(resourceDir_ + "/walk.txt");
    float data;
    int i = 0, j = 0;
    while (gcfile >> data){
      dataGC_.coeffRef(j, i) = data;
      i++;
      // if (i >= gcDim_){
        if (i >= gcDim_ - 8){
        i = 0;
        j++;
      }
    }
    // std::ifstream gvfile(resourceDir_ + "/walk_gv.txt");
    // i = 0; j = 0;
    // while (gvfile >> data){
    //   data_gv_.coeffRef(j, i) = data;
    //   i++;
    //   // if (i >= gvDim_){
    //   if (i >= gvDim_ - 6){
    //     i = 0;
    //     j++;
    //   }
    // }
  }

  void reset() final {
    sim_step_ = 0;
    total_reward_ = 0;

    nLoops_ = 0;
    loopDisplacementAccumulated_.setZero();
    loopTurnAccumulated_.setIdentity(); // raisim::quatToRotMat({1, 0, 0, 0}, loop_turn_acc_);
    
    initializeCharacter();
    initializeBall();
    // flags
    resetFlags();

    updateObservation();
  }

  void initializeCharacter(){
    // select random frame
    index_ = rand() % dataLen_;
    charPhase_ = index_ * charPhaseSpeed_;
    gcInit_ = dataGC_.row(index_);
    gvInit_ = dataGV_.row(index_);
    
    // TODO: Noisier initialization scheme as the learning progresses
    if (dribble_){
      // gcInit_[cStartIdx_[2]] = 1; gcInit_[cStartIdx_[2] + 1] = 0; gcInit_[cStartIdx_[2] + 2] = 0; gcInit_[cStartIdx_[2] + 3] = 0; 
      gcInit_[cStart_[2]] = 0.866; gcInit_[cStart_[2] + 1] = -0.5; gcInit_[cStart_[2] + 2] = 0; gcInit_[cStart_[2] + 3] = 0;
      gcInit_[cStart_[3]] = 1.57;
      gcInit_[cStart_[4]] = 0.966; gcInit_[cStart_[4] + 1] = 0; gcInit_[cStart_[4] + 2] = 0.259; gcInit_[cStart_[4] + 3] = 0;

      gvInit_[vStart_[2]] = 0; gvInit_[vStart_[2] + 1] = 0; gvInit_[vStart_[2] + 2] = 0;
      gvInit_[vStart_[3]] = 0;
      gvInit_[vStart_[4]] = 0; gvInit_[vStart_[4] + 1] = 0; gvInit_[vStart_[4] + 2] = 0;
    }
    
    pTarget_ << gcInit_;
    vTarget_.setZero();
    
    simChar_->setState(gcInit_, gvInit_);
    simChar_->setPdTarget(pTarget_, vTarget_);
  }

  void initializeBall(){
    // ball state initialization
    if (dribble_){
      Vec<3> rightHandPos;
      size_t rightHandIdx = simChar_->getFrameIdxByName("right_wrist"); // 9
      simChar_->getFramePosition(rightHandIdx, rightHandPos);
      // ballGCInit_[0] = rightHandPos[0] + 0.1;
      ballGCInit_[0] = rightHandPos[0] + 0.08850; // half the hand size
      ballGCInit_[1] = rightHandPos[1];
      ballGCInit_[2] = rightHandPos[2] - 0.171; // ball 0.14, hand 0.03
      ballGCInit_[3] = 1;

      ballGVInit_[0] = gvInit_[0];
      ballGVInit_[1] = gvInit_[1];
      ballGVInit_[2] = 0.05;
    }
    else{
      ballGCInit_[0] = 0; ballGCInit_[1] = 100; ballGCInit_[2] = 5; ballGCInit_[3] = 1;
    }
    ball_->setState(ballGCInit_, ballGVInit_);
    
  }

  void resetFlags(){
    fromGround_ = false;
    fromHand_ = false;
    isGround_ = false;
    isHand_ = false;
    groundHand_ = false;
    fallFlag_ = false;
    contactTerminalFlag_ = false;
  }

  void updateObservation() {
    simChar_->getState(gc_, gv_);
    ball_->getState(ballGC_, ballGV_);

    Mat<3,3> rootRotInv;
    Vec<3> rootPos;
    getRootTransform(rootRotInv, rootPos);

    Vec<3> jointPos_W, jointPos_B, jointVel_W, jointVel_B;
    int obIdx = 0;
    int gcIdx = 7;
    int gvIdx = 6;
    int eeIdx = 0;

    // joint pos, orn, linvel, angvel
    for(size_t bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx++){
      simChar_->getBodyPosition(bodyIdx, jointPos_W);
      matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
      obDouble_.segment(obIdx, 3) = jointPos_B.e();
      obIdx += 3;

      simChar_->getVelocity(bodyIdx, jointVel_W);
      matvecmul(rootRotInv, jointVel_W, jointVel_B);
      obDouble_.segment(obIdx, 3) = jointVel_B.e();
      obIdx += 3;

      obDouble_.segment(obIdx, cDim_[bodyIdx - 1]) = gc_.segment(gcIdx, cDim_[bodyIdx - 1]);
      obIdx += cDim_[bodyIdx - 1]; gcIdx += cDim_[bodyIdx - 1];
      obDouble_.segment(obIdx, vDim_[bodyIdx - 1]) = gv_.segment(gvIdx, vDim_[bodyIdx - 1]);
      obIdx += vDim_[bodyIdx - 1]; gvIdx += vDim_[bodyIdx - 1];

      // for ee-reward. not recorded to observation.
      if (isEE_[bodyIdx - 1]){
        ee_.segment(eeIdx, 3) = jointPos_B.e();
        eeIdx += 3;
      }
    }

    if (useBallState_){
      if (dribble_){
        // ball pos, lin vel
        matvecmul(rootRotInv, ballGC_.head(3) - rootPos.e(), jointPos_B);
        matvecmul(rootRotInv, ballGV_.head(3), jointVel_B);
        obDouble_.segment(obIdx, 3) = jointPos_B.e();
        obIdx += 3;
        obDouble_.segment(obIdx, 3) = jointVel_B.e();
        obIdx += 3;
        Vec<3> rHandPos_W;
        simChar_->getPosition(rWristIdx_ + 1, rHandCenter, rHandPos_W);
        ballDist_ = (rHandPos_W[0] - ballGC_[0]) * (rHandPos_W[0] - ballGC_[0]) + (rHandPos_W[1] - ballGC_[1]) * (rHandPos_W[1] - ballGC_[1]);
        
      }
      else{ // for transfer learning?
        obDouble_.segment(obIdx, 6).setZero();
      }
    }

    // char phase
    if (useCharPhase_){
      obDouble_[obIdx] = std::cos(charPhase_);
      obDouble_[obIdx + 1] = std::sin(charPhase_);
      obIdx += 2;
    }

    // ball phase
    if (useBallPhase_){
      obDouble_[obIdx] = std::cos(ballPhase_);
      obDouble_[obIdx + 1] = std::sin(ballPhase_);
      obIdx += 2;
    }

    // for com reward. not recorded to observation
    com_ = (simChar_->getCOM()).e();
  }

  void getRootTransform(Mat<3,3>& rot, Vec<3>& pos) {
    Vec<4> rootRot, defaultRot, rootRotRel;
    rootRot[0] = gc_[3]; rootRot[1] = gc_[4]; rootRot[2] = gc_[5]; rootRot[3] = gc_[6];
    defaultRot[0] = 0.707; defaultRot[1] =  - 0.707; defaultRot[2] = 0; defaultRot[3] = 0;
    raisim::quatMul(defaultRot, rootRot, rootRotRel);
    double yaw = atan2(2 * (rootRotRel[0] * rootRotRel[2] + rootRotRel[1] * rootRotRel[3]), 1 - 2 * (rootRotRel[2] * rootRotRel[2] + rootRotRel[3] * rootRotRel[3]));
    Vec<4> quat;
    quat[0] = cos(yaw / 2); quat[1] = 0; quat[2] = 0; quat[3] = - sin(yaw / 2);
    raisim::quatToRotMat(quat, rot);
    pos[0] = gc_[0]; pos[1] = gc_[1]; pos[2] = gc_[2];
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    prevGV_ = gv_;

    gcRef_ = dataGC_.row(index_);
    gvRef_ = dataGV_.row(index_);
    eeRef_ = dataEE_.row(index_);
    comRef_ = dataCom_.row(index_);

    if (visKin_){
      Eigen::VectorXd kinGC_, kinGV_;
      kinGC_ = gcRef_; kinGV_ = gvRef_;
      kinGC_[1] += 1.5;
      kinChar_->setState(kinGC_, kinGV_);
    }

    int actionIdx = 0;
    int controlIdx;
    for (int jointIdx = 0; jointIdx < nJoints_; jointIdx++)
    {
      controlIdx = actionIdx + 7;
      pTarget_.segment(controlIdx, cDim_[jointIdx]) << pTarget_.segment(controlIdx, cDim_[jointIdx]) + action.cast<double>().segment(actionIdx, cDim_[jointIdx]);
      if (cDim_[jointIdx] == 4){
        pTarget_.segment(controlIdx, cDim_[jointIdx]) << pTarget_.segment(controlIdx, cDim_[jointIdx]).normalized();
      }
      actionIdx += cDim_[jointIdx];
    }

    simChar_->setPdTarget(pTarget_, vTarget_);
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

    isHand_ = false;
    isGround_ = false;

    updateObservation();
    if (gc_[2] < rootHeightThreshold_){
      fallFlag_ = true;
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
        
        if (fromGround_) {
          contactTerminalFlag_ = true;
          break;
        }
        if (isHand_) {
          contactTerminalFlag_ = true;
          break;
        }
        isGround_ = true;
        fromGround_ = true;
        fromHand_ = false;
      }
      else{
        auto& pair_contact = world_->getObject(contact.getPairObjectIndex())->getContacts()[contact.getPairContactIndexInPairObject()];
        // todo
        if (simChar_->getBodyIdx("right_wrist") == pair_contact.getlocalBodyIndex() || simChar_->getBodyIdx("right_elbow") == pair_contact.getlocalBodyIndex()){
          if (isGround_) {
            contactTerminalFlag_ = true;
            break;
          }
          if (contact.getNormal()[2] > 0) {
            contactTerminalFlag_ = true;
            break;
          }
          if (fromGround_) {
            groundHand_ = true;
          }
          isHand_ = true;
          fromHand_ = true;
          fromGround_ = false;
        }
        else{
          contactTerminalFlag_ = true;
          break;
        }
      }
    }
  }

  void checkCharacterContact(){
    for(auto& contact: simChar_->getContacts()){
      if (contact.getPosition()[2] < 0.01){
        if ((contact.getlocalBodyIndex() != rAnkleIdx_ + 1) && (contact.getlocalBodyIndex() != lAnkleIdx_ + 1)){
          fallFlag_ = true;
          break;
        }
      }
    }
  }

  void setBallPhaseSpeed(){
    if (isHand_){
      ballPhase_ = M_PI;
      float v = - std::abs(ballGV_[2]);
      float h = std::abs(ballGC_[2]);
      float g = std::abs(world_->getGravity()[2]);
      float t = (std::sqrt(v * v + 2 * g * h) - v) / g;
      ballPhaseSpeed_ = M_PI / t;
    }
    if (isGround_){
      ballPhase_ = 0;
    }
  }

  void updateTargetMotion(){
    index_ += 1;
    charPhase_ += charPhaseSpeed_;
    ballPhase_ += ballPhaseSpeed_;
    sim_step_ += 1;
    if (index_ >= dataLen_){
      index_ = 0;
      charPhase_ = 0;
      nLoops_ += 1;
      Vec<3> temp_disp; Mat<3,3> temp_turn;
      matvecmul(loopTurnAccumulated_, loopDisplacement_, temp_disp);
      loopDisplacementAccumulated_ += temp_disp;
      temp_turn = loopTurnAccumulated_;
      raisim::matmul(loopTurn_, temp_turn, loopTurnAccumulated_);
    }
  }

  void computeReward() {
    
    // imitation reward
    double ornErr = 0, ornReward = 0;
    double velErr = 0, velReward = 0;
    double eeErr = 0, eeReward = 0;
    double comErr = 0, comReward = 0;

    Vec<4> quat, quatRef, quatErr;
    Mat<3,3> mat, matRef, matErr;

    for (size_t jointIdx = 0; jointIdx < nJoints_; jointIdx++) {
      if (mask_ && isRightArm_[jointIdx])
      {
        continue;
      }
      if (cDim_[jointIdx] == 1)
      {
        ornErr += std::pow(gc_[cStart_[jointIdx]] - gcRef_[cStart_[jointIdx]], 2);
      }
      else {
        quat[0] = gc_[cStart_[jointIdx]]; quat[1] = gc_[cStart_[jointIdx]+1]; 
        quat[2] = gc_[cStart_[jointIdx]+2]; quat[3] = gc_[cStart_[jointIdx]+3];
        quatRef[0] = gcRef_[cStart_[jointIdx]]; quatRef[1] = gcRef_[cStart_[jointIdx]+1]; 
        quatRef[2] = gcRef_[cStart_[jointIdx]+2]; quatRef[3] = gcRef_[cStart_[jointIdx]+3];
        raisim::quatToRotMat(quat, mat);
        raisim::quatToRotMat(quatRef, matRef);
        raisim::mattransposematmul(mat, matRef, matErr);
        raisim::rotMatToQuat(matErr, quatErr);
        ornErr += std::pow(acos(std::max(std::min(1.0, quatErr[0]), -1.0)) * 2, 2);
      }
    }

    ornReward = exp(-ornScale_ * ornErr);
    rewards_.record("orientation", ornReward);

    if (mask_){
      velErr = (gv_.segment(0, vStart_[rShoulderIdx_]) - gvRef_.segment(0, vStart_[rShoulderIdx_])).squaredNorm();
      velErr += (gv_.tail(gvDim_ - (vStart_[rWristIdx_] + vDim_[rWristIdx_])) - gvRef_.tail(gvDim_ - (vStart_[rWristIdx_] + vDim_[rWristIdx_]))).squaredNorm();
    }
    else{
      velErr = (gv_.tail(controlDim_) - gvRef_.tail(controlDim_)).squaredNorm();
    }
    velReward = exp(- velScale_ * velErr);
    rewards_.record("velocity", velReward);

    if (mask_){
      eeErr = (ee_.segment(3, eeDim_ - 3) - eeRef_.segment(3, eeDim_ - 3)).squaredNorm();
    }
    else{
      eeErr = (ee_ - eeRef_).squaredNorm();
    }
    eeReward = exp(- eeScale_ * eeErr);
    rewards_.record("end effector", eeReward);

    comRef_ = loopTurnAccumulated_ * comRef_;
    comRef_ = comRef_ + loopDisplacementAccumulated_.e();

    comErr = (com_ - comRef_).squaredNorm();
    comReward = exp(-comScale_ * comErr);
    rewards_.record("com", comReward);
    
    double contactReward = 0;
    if (groundHand_) {
        contactReward = 1;
        groundHand_ = false;
    }
    rewards_.record("contact", contactReward);

    double distReward = 0;
    if (dribble_){
      distReward += exp(-ballDist_);
    }
    rewards_.record("ball distance", distReward);

    double energyReward = 0;
    energyReward += exp(- energyScale_ * (prevGV_ - gv_).squaredNorm());
    rewards_.record("energy efficiency", energyReward);
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
    if (fallFlag_) {
      return true;
    }

    if (dribble_){
      // unwanted contact state
      if (contactTerminalFlag_) {
        return true;
      }

      // ball too far
      if (ballDist_ > ballDistThreshold_){
        return true;
      }
    }
    return false;
  }

  private:
    bool dribble_, useBallState_, mask_;
    std::string charFileName_, motionFileName_;
    bool dataHasWrist_, isPreprocess_, visKin_, useCharPhase_, useBallPhase_;
    float ornScale_, velScale_, eeScale_, comScale_, energyScale_;

    bool visualizable_ = false;
    raisim::ArticulatedSystem *simChar_, *kinChar_;
    raisim::ArticulatedSystem *ball_;

    float restitution_;

    int nJoints_ = 14;
    
    int chestIdx_ = 0;
    int neckIdx_ = 1;
    int rShoulderIdx_ = 2;
    int rElbowIdx_ = 3;
    int rWristIdx_ = 4;
    int lShoulderIdx_ = 5;
    int lElbowIdx_ = 6;
    int lWristIdx_ = 7;
    int rHipIdx_ = 8;
    int rKneeIdx_ = 9;
    int rAnkleIdx_ = 10;
    int lHipIdx_ = 11;
    int lKneeIdx_ = 12;
    int lAnkleIdx_ = 13;

    int cStart_[14] = {7, 11,  15, 19, 20,  24, 28, 29,  33, 37, 38,  42, 46, 47};
    int vStart_[14] = {6,  9,  12, 15, 16,  19, 22, 23,  26, 29, 30,  33, 36, 37};
    int cDim_[14] = {4, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4};
    int vDim_[14] = {3, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3};
    int isEE_[14] = {0, 0,  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};
    int isRightArm_[14] = {0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0, 0, 0};


    Vec<3> rHandCenter = {0, -0.08850, 0};


    int gcDim_, gvDim_, controlDim_, ballGCDim_, ballGVDim_;
    int posDim_ = 3 * nJoints_, comDim_ = 3, eeDim_ = 12;

    Eigen::VectorXd gc_, gv_, gcInit_, gvInit_, gcRef_, gvRef_;
    Eigen::VectorXd prevGV_;
    Eigen::MatrixXd dataGC_, dataGV_, dataEE_, dataCom_;
    Eigen::VectorXd com_, comRef_, ee_, eeRef_;
    
    Vec<3> loopDisplacement_;
    Mat<3,3> loopTurn_;
    Vec<3> loopDisplacementAccumulated_;
    Mat<3,3> loopTurnAccumulated_;

    Eigen::VectorXd pTarget_, vTarget_;
    Eigen::VectorXd obDouble_, stateDouble_;
    
    Eigen::VectorXd ballGC_, ballGV_, ballGCInit_, ballGVInit_;

    int maxLen_ = 1000;
    int dataLen_;
    int index_ = 0;

    float charPhase_ = 0, charPhaseSpeed_ = 0;
    float ballPhase_ = 0, ballPhaseSpeed_ = 0;
    float rootHeightThreshold_ = 0.5, ballDistThreshold_ = 0.5;
    float ballDist_;

    int sim_step_ = 0;
    int max_sim_step_ = 2000;
    double total_reward_ = 0;

    bool contactTerminalFlag_ = false, fallFlag_ = false;
    bool fromGround_ = false, fromHand_ = false, isGround_ = false, isHand_ = false, groundHand_ = false;
    
    int nLoops_;
};

}