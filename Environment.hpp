#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

#include <queue>

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

      setCam(); 
    }

  void init() final {}

  void setup(const Yaml::Node& cfg){
    // EXPERIMENT SETTINGS
    useCam_ = cfg["use_cam"].template As<bool>();

    charFileName_ = cfg["character"]["file name"].template As<std::string>();
    visKin_ = cfg["character"]["visualize kinematic"].template As<bool>();
    restitution_ = cfg["character"]["restitution"].template As<float>();

    armSpread_ = cfg["character"]["arm spread"].template As<float>();

    motionFileName_ = cfg["motion data"]["file name"].template As<std::string>();
    dataHasWrist_ = cfg["motion data"]["has wrist"].template As<bool>();
    isPreprocess_ = cfg["motion data"]["preprocess"].template As<bool>();
    dataLen_ = cfg["motion data"]["num frames"].template As<int>();

    control_dt_ = 1.0 / cfg["motion data"]["fps"].template As<float>();
    simulation_dt_ = cfg["simulation_dt"].template As<float>();

    useCharPhase_ = cfg["phase usage"]["character"].template As<bool>();
    useBallPhase_ = cfg["phase usage"]["ball"].template As<bool>();

    ornScale_ = cfg["error sensitivity"]["orientation"].template As<float>();
    handBallDistScale_ = cfg["error sensitivity"]["hand ball distance"].template As<float>();
    rWristOrnScale_ = cfg["error sensitivity"]["right wrist orientation"].template As<float>();

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
    simChar_ = world_->addArticulatedSystem(resourceDir_ + "/" + charFileName_ + ".urdf"); 
    simChar_->setName("sim character");
    if (visualizable_){
      server_->focusOn(simChar_);
    }

    if (visKin_){
      kinChar_ = world_->addArticulatedSystem(resourceDir_ + "/" + charFileName_ + ".urdf"); 
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
    dataGC_.setZero(dataLen_, gcDim_);
    charPhaseSpeed_ = 2.0 * M_PI / (float) dataLen_;
    std::ifstream gcfile(resourceDir_ + "/" + motionFileName_ + ".txt");
    float data;
    int row = 0, col = 0;
    while (gcfile >> data) {
      dataGC_.coeffRef(row, col) = data;

      if (charFileName_ == "ybot" && col == 2){
        dataGC_.coeffRef(row, col) = data - 0.15;
      }
      if (charFileName_ == "golem" && col == 2){
        dataGC_.coeffRef(row, col) = data - 0.12;
      }
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

    // TODO: integrate loop transformation
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

  void setCam(){
    if (useCam_){
      focus_ = world_->addArticulatedSystem(resourceDir_ + "/focus.urdf"); 
      focus_->setName("focus");
      focusGC_.setZero(7); focusGV_.setZero(6);
      focusGC_[2] = 1; focusGC_[3] = 1;
      focus_->setState(focusGC_, focusGV_);

      camGC_.setZero();
      camDisplacement_.setZero();
      camDirection_.setZero();
      camDisplacement_[0] = 3; camDisplacement_[1] = 0; camDisplacement_[2] = 1;
    }
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

    std::queue<double> xempty, yempty;
    std::swap(xqueue_, xempty);
    std::swap(yqueue_, yempty);


    for (int i=0; i<camWindow_; i++){
      xqueue_.push(gcInit_[0]);
      yqueue_.push(gcInit_[1]);
    }
    xave_ = gcInit_[0];
    yave_ = gcInit_[1];

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
      // humanoid
      gcInit_[cStart_[rShoulderIdx_]] = 1; gcInit_[cStart_[rShoulderIdx_] + 1] = 0; gcInit_[cStart_[rShoulderIdx_] + 2] = 0; gcInit_[cStart_[rShoulderIdx_] + 3] = 0;
      simChar_->setState(gcInit_, gvInit_);
      Mat<3, 3> rShoulderOrn;
      simChar_->getFrameOrientation("right_shoulder", rShoulderOrn);
      float zrot = std::atan2(-rShoulderOrn[2], rShoulderOrn[5]);
      float cz = std::cos(zrot/2), sz = std::sin(zrot/2), cx = std::cos(armSpread_/2), sx = std::sin(armSpread_/2), cx2=std::cos((M_PI/2-armSpread_)/2), sx2=std::sin((M_PI/2-armSpread_)/2);
      gcInit_[cStart_[rShoulderIdx_]] = cz * cx; gcInit_[cStart_[rShoulderIdx_] + 1] = - cz * sx; gcInit_[cStart_[rShoulderIdx_] + 2] = - sz * sx; gcInit_[cStart_[rShoulderIdx_] + 3] = sz * cx;
      gcInit_[cStart_[rElbowIdx_]] = 1.57;
      gcInit_[cStart_[rWristIdx_]] = cx2; gcInit_[cStart_[rWristIdx_] + 1] = 0; gcInit_[cStart_[rWristIdx_] + 2] = sx2; gcInit_[cStart_[rWristIdx_] + 3] = 0;

      gvInit_[vStart_[rShoulderIdx_]] = 0; gvInit_[vStart_[rShoulderIdx_] + 1] = 0; gvInit_[vStart_[rShoulderIdx_] + 2] = 0;
      gvInit_[vStart_[rElbowIdx_]] = 0;
      gvInit_[vStart_[rWristIdx_]] = 0; gvInit_[vStart_[rWristIdx_] + 1] = 0; gvInit_[vStart_[rWristIdx_] + 2] = 0;

      // ybot
      // gcInit_[cStart_[rCollarIdx_]] = 1; gcInit_[cStart_[rCollarIdx_] + 1] = 0; gcInit_[cStart_[rCollarIdx_] + 2] = 0; gcInit_[cStart_[rCollarIdx_] + 3] = 0;
      // gvInit_[vStart_[rCollarIdx_]] = 0; gvInit_[vStart_[rCollarIdx_] + 1] = 0; gvInit_[vStart_[rCollarIdx_] + 2] = 0;
      // gcInit_[cStart_[rShoulderIdx_]] = 1; gcInit_[cStart_[rShoulderIdx_] + 1] = 0; gcInit_[cStart_[rShoulderIdx_] + 2] = 0; gcInit_[cStart_[rShoulderIdx_] + 3] = 0;
      // gvInit_[vStart_[rShoulderIdx_]] = 0; gvInit_[vStart_[rShoulderIdx_] + 1] = 0; gvInit_[vStart_[rShoulderIdx_] + 2] = 0;
      // gvInit_[vStart_[rElbowIdx_]] = 0;
      // gvInit_[vStart_[rWristIdx_]] = 0; gvInit_[vStart_[rWristIdx_] + 1] = 0; gvInit_[vStart_[rWristIdx_] + 2] = 0;
      // simChar_->setState(gcInit_, gvInit_);
      
      // Mat<3, 3> rShoulderOrn;
      // simChar_->getFrameOrientation("right_shoulder", rShoulderOrn);
      // float zrot = std::atan2(-rShoulderOrn[2], rShoulderOrn[5]);
      // float cz = std::cos(zrot/2), sz = std::sin(zrot/2), cx = std::cos(armSpread_/2), sx = std::sin(armSpread_/2);
      // gcInit_[cStart_[rShoulderIdx_]] = cz * cx; gcInit_[cStart_[rShoulderIdx_] + 1] = - cz * sx; gcInit_[cStart_[rShoulderIdx_] + 2] = - sz * sx; gcInit_[cStart_[rShoulderIdx_] + 3] = sz * cx;
      // // gcInit_[cStart_[rElbowIdx_]] = 1.57;
      // // golem
      // gcInit_[cStart_[rElbowIdx_]] = -1.57;
      // gcInit_[cStart_[rWristIdx_]] = 1; gcInit_[cStart_[rWristIdx_]+1] = 0; gcInit_[cStart_[rWristIdx_]+2] = 0; gcInit_[cStart_[rWristIdx_]+3] = 0;
      // simChar_->setState(gcInit_, gvInit_);

      // Mat<3, 3> rWristOrn;
      // simChar_->getFrameOrientation("right_wrist", rWristOrn);
      // float zrot2 = std::atan2(-rWristOrn[2], rWristOrn[5]);
      // gcInit_[cStart_[rWristIdx_]] = std::cos(zrot2/2); gcInit_[cStart_[rWristIdx_]+1] = 0; gcInit_[cStart_[rWristIdx_]+2] = 0; gcInit_[cStart_[rWristIdx_]+3] = std::sin(zrot2/2);
    }
    
    pTarget_ << gcInit_;
    vTarget_.setZero();

    // golem
    // gvInit_[0] = 1;
    
    simChar_->setState(gcInit_, gvInit_);
    simChar_->setPdTarget(pTarget_, vTarget_);
  }

  void initializeBall(){
    // ball state initialization
    if (dribble_){
      simChar_->getPosition(rWristIdx_ + 1, rHandCenter_, rightHandPos_);
      // ballGCInit_[0] = rightHandPos[0] + 0.1;
      ballGCInit_[0] = rightHandPos_[0]; // half the hand size
      ballGCInit_[1] = rightHandPos_[1];
      ballGCInit_[2] = rightHandPos_[2] - 0.4; // ball 0.14, hand 0.015
      ballGCInit_[3] = 1;

      ballGVInit_[0] = gvInit_[0];
      // ballGVInit_[1] = gvInit_[1];
      // ballGVInit_[2] = 0.05;
      ballGVInit_[2] = 0.5; //todo 1.0
    }
    else{
      ballGCInit_[0] = 0; ballGCInit_[1] = 100; ballGCInit_[2] = 5; ballGCInit_[3] = 1;
    }
    ball_->setState(ballGCInit_, ballGVInit_);
  }

  void resetFlags(){
    fromGround_ = true;
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
        simChar_->getPosition(rWristIdx_ + 1, rHandCenter_, rightHandPos_);
        handBallDist_ = (rightHandPos_[0] - ballGC_[0]) * (rightHandPos_[0] - ballGC_[0]) + (rightHandPos_[1] - ballGC_[1]) * (rightHandPos_[1] - ballGC_[1]);
        rootBallDist_ = (gc_[0] - ballGC_[0]) * (gc_[0] - ballGC_[0]) + (gc_[1] - ballGC_[1]) * (gc_[1] - ballGC_[1]);
      }
      else{ // for transfer learning?
        obDouble_.segment(obIdx, 6).setZero();
      }

      rootBallAngle_ = std::atan2(jointPos_B[0], - jointPos_B[1]);
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

    Mat<3, 3> rWristOrn;
    simChar_->getFrameOrientation("right_wrist", rWristOrn);
    raisim::matmul(rootRotInv, rWristOrn, rWristOrn_);

    if (useCam_ && server_){
      xave_ = xave_ * camWindow_ - xqueue_.front() + gc_[0];
      xave_ /= camWindow_;
      yave_ = yave_ * camWindow_ - yqueue_.front() + gc_[1];
      yave_ /= camWindow_;
      xqueue_.push(gc_[0]);
      xqueue_.pop();
      yqueue_.push(gc_[1]);
      yqueue_.pop();

      Vec<3> camDisplacement;
      raisim::matTransposevecmul(rootRotInv, camDisplacement_, camDisplacement);
      
      focusGC_.head(2) = gc_.head(2);
      focusGV_.head(3) = gv_.head(3);
      focus_->setState(focusGC_, focusGV_);
      camGC_[0] = focusGC_[0]; camGC_[1] = focusGC_[1]; camGC_[2] = focusGC_[2];
      camGC_[0] = xave_; camGC_[1] = yave_; camGC_[2] = focusGC_[2];
      // server_->setCameraPositionAndLookAt(camGC_ + camDisplacement.e(), -camDisplacement.e());
      server_->setCameraPositionAndLookAt(camGC_ + camDisplacement_.e(), focusGC_);
    }

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
        // if (dribble_ && (jointIdx == rWristIdx_)){
        //   pTarget_[controlIdx + 3] = - pTarget_[controlIdx + 1] * pTarget_[controlIdx + 2] / (pTarget_[controlIdx] + 1e-10);
        // }
        // if (dribble_ && rotationType_[jointIdx]){
        //   applyJointLimit(controlIdx, jointIdx);
        // }
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
      
      // checkCharacterContact();
    }

    setBallPhaseSpeed();

    isHand_ = false;
    isGround_ = false;

    updateObservation();
    // if (gc_[2] < rootHeightThreshold_){
    //   fallFlag_ = true;
    // }
    Vec<3> neckPos_W;
    simChar_->getBodyPosition(neckIdx_ + 1, neckPos_W);
    if (neckPos_W[2] < neckHeightThreshold_){
      fallFlag_ = true;
    }

    computeReward();

    updateTargetMotion();

    double current_reward = rewards_.sum();
    total_reward_ += current_reward;

    return current_reward;
  }

  void applyJointLimit(int controlIdx, int jointIdx){
    int rotationType = rotationType_[jointIdx];
    if (rotationType == 1){
      pTarget_[controlIdx + 1] = (pTarget_[controlIdx + 2] * pTarget_[controlIdx + 3]) / (pTarget_[controlIdx] + 1e-10);
    }
    if (rotationType == 2){
      bool isPositive = pTarget_[controlIdx + 2] > 0;
      if (isPositive){
        pTarget_[controlIdx + 2] = std::abs(pTarget_[controlIdx + 1] * pTarget_[controlIdx + 3] / (pTarget_[controlIdx] + 1e-10));
      }
      else{
        pTarget_[controlIdx + 2] = - std::abs(pTarget_[controlIdx + 1] * pTarget_[controlIdx + 3] / (pTarget_[controlIdx] + 1e-10));
      }
    }
    if (rotationType == 3){
      pTarget_[controlIdx + 3] = - (pTarget_[controlIdx + 1] * pTarget_[controlIdx + 2]) / (pTarget_[controlIdx] + 1e-10);
    }
    if (rotationType == 4){
      bool isPositive = pTarget_[controlIdx + 3] > 0;
      if (isPositive){
        pTarget_[controlIdx + 3] = std::abs(pTarget_[controlIdx + 1] * pTarget_[controlIdx + 2] / (pTarget_[controlIdx] + 1e-10));
      }
      else{
        pTarget_[controlIdx + 3] = - std::abs(pTarget_[controlIdx + 1] * pTarget_[controlIdx + 2] / (pTarget_[controlIdx] + 1e-10));
      }
    }
    if (rotationType == 5){
      pTarget_[controlIdx + 2] = (pTarget_[controlIdx + 3] * pTarget_[controlIdx + 1]) / (pTarget_[controlIdx] + 1e-10);
    }
    if (rotationType == 10){
      double lowerBound = simChar_->getJointLimits()[vStart_[jointIdx]][0];
      double upperBound = simChar_->getJointLimits()[vStart_[jointIdx]][1];
      pTarget_[controlIdx] = std::min(std::max(lowerBound, pTarget_[controlIdx]), upperBound);
    }
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
        if (simChar_->getBodyIdx("right_wrist") == pair_contact.getlocalBodyIndex()){
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
      float h = std::abs(ballGC_[2]) - 0.11;
      float g = std::abs(world_->getGravity()[2]);
      float t = (std::sqrt(v * v + 2 * g * h) - v) / g;
      ballPhaseSpeed_ = M_PI / (t / control_dt_);
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

    Vec<4> quat, quatRef, quatErr;
    Mat<3,3> mat, matRef, matErr;

    for (size_t jointIdx = 0; jointIdx < nJoints_; jointIdx++) {
      if (mask_ && isMask_[jointIdx])
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


    double contactReward = 0;
    if (groundHand_) {
      contactReward = 1;
      // float handBallTrueDist = (rightHandPos_[0] - ballGC_[0])*(rightHandPos_[0] - ballGC_[0])
      //   + (rightHandPos_[1] - ballGC_[1])*(rightHandPos_[1] - ballGC_[1])
      //   + (rightHandPos_[2] - ballGC_[2])*(rightHandPos_[2] - ballGC_[2]);
      // handBallTrueDist = std::sqrt(handBallTrueDist) - 0.15;
      // contactReward = exp(-handBallTrueDist);
      groundHand_ = false;
    }
    rewards_.record("contact", contactReward);

    double handDistReward = 0;
    double rootDistReward = 0;
    if (dribble_){
      handDistReward += exp(-handBallDistScale_ * handBallDist_);
    }
    rewards_.record("hand ball distance", handDistReward);

    double rWristOrnReward = 0;
    // humanoid
    rWristOrnReward = exp(- rWristOrnScale_ * (2 - rWristOrn_[1] + rWristOrn_[3]));
    // ybot
    // rWristOrnReward = exp(- rWristOrnScale_ * (2 - rWristOrn_[6] - rWristOrn_[5]));
    //golem
    // rWristOrnReward = exp(- rWristOrnScale_ * (2 - rWristOrn_[6] - rWristOrn_[5]));
    rewards_.record("right wrist orientation", rWristOrnReward);

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
      // if (handBallDist_ > ballDistThreshold_){
      //   return true;
      // }

      if (rootBallDist_ > 1.0){
        return true;
      }

      // if (ballGC_[2] > 1.5){
      //   return true;
      // }
    }
    return false;
  }

  private:
    bool dribble_, useBallState_, mask_;
    std::string charFileName_, motionFileName_;
    bool dataHasWrist_, isPreprocess_, visKin_, useCharPhase_, useBallPhase_;
    float ornScale_, velScale_, eeScale_, comScale_, energyScale_, handBallDistScale_, rootBallDistScale_, rootBallVelScale_;
    
    float rWristOrnScale_;

    double desiredRootBallDist_ = 0.3;

    float armSpread_ = M_PI / 4;

    bool visualizable_ = false;
    
    bool useCam_ = false;
    raisim::ArticulatedSystem *simChar_, *kinChar_;
    raisim::ArticulatedSystem *ball_;
    raisim::ArticulatedSystem *focus_;
    Eigen::VectorXd focusGC_, focusGV_;
    // Eigen::Vector3d camDisplacement_, camDirection_;
    Vec<3> camDisplacement_, camDirection_;
    Eigen::Vector3d camGC_;
    

    std::queue<double> xqueue_;
    std::queue<double> yqueue_;
    double xave_;
    double yave_;
    size_t camWindow_ = 30;

    float restitution_;

    // humanoid

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
    int isMask_[14] = {0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0, 0, 0};

    int rotationType_[14] = {0, 0, 2, 10, 3, 2, 10, 3, 2, 10, 0, 2, 10, 0};
    
    Vec<3> rHandCenter_ = {0, -0.08850, 0};

    // // ybot
    // int nJoints_ = 18;

    // int spineIdx_ = 0; // 7
    // int spine1Idx_ = 1; // 11
    // int spine2Idx_ = 2; // 15
    // int neckIdx_ = 3; // 19
    // int rCollarIdx_ = 4; // 23
    // int rShoulderIdx_ = 5; // 27
    // int rElbowIdx_ = 6; // 31
    // int rWristIdx_ = 7; // 32
    // int lCollarIdx_ = 8; // 36
    // int lShoulderIdx_ = 9; // 40
    // int lElbowIdx_ = 10; // 44
    // int lWristIdx_ = 11; // 45
    // int rHipIdx_ = 12; // 49
    // int rKneeIdx_ = 13; // 53
    // int rAnkleIdx_ = 14; // 54
    // int lHipIdx_ = 15; // 58
    // int lKneeIdx_ = 16; // 62
    // int lAnkleIdx_ = 17; // 63

    // int cStart_[18] = {7, 11, 15, 19, 23, 27, 31, 32, 36, 40, 44, 45, 49, 53, 54, 58, 62, 63};
    // int cDim_[18] = {4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4};
    // int vStart_[18] = {6,  9, 12, 15, 18, 21, 24, 25, 28, 31, 34, 35, 38, 41, 42, 45, 48, 49};
    // int vDim_[18] = {3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3};
    // int isEE_[18] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    // int isMask_[18] = {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // int rotationType_[18] = {0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0};

    // Vec<3> rHandCenter_ = {0, 0, 0.08850};

    // golem
    // int nJoints_ = 17;

    // int backIdx_ = 0;
    // int chestIdx_ = 1;
    // int neckIdx_ = 2;
    // int rCollarIdx_ = 3;
    // int rShoulderIdx_ = 4;
    // int rElbowIdx_ = 5;
    // int rWristIdx_ = 6;
    // int lCollarIdx_ = 7;
    // int lShoulderIdx_ = 8;
    // int lElbowIdx_ = 9;
    // int lWristIdx_ = 10;
    // int rHipIdx_ = 11;
    // int rKneeIdx_ = 12;
    // int rAnkleIdx_ = 13;
    // int lHipIdx_ = 14;
    // int lKneeIdx_ = 15;
    // int lAnkleIdx_ = 16;

    // int cStart_[17] = {7, 11, 15, 19, 23, 27, 28, 32, 36, 40, 41, 45, 49, 50, 54, 58, 59};
    // int cDim_[17] = {4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4};
    // int vStart_[17] = {6, 9, 12, 15, 18, 21, 22, 25, 28, 31, 32, 35, 38, 39, 42, 45, 46};
    // int vDim_[17] = {3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3};
    // int isEE_[17] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    // int isMask_[17] = {0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // int rotationType_[18] = {0, 0, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0};

    // Vec<3> rHandCenter_ = {0, 0, 0.15};

    Vec<3> rightHandPos_;

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
    float rootHeightThreshold_ = 0.5, ballDistThreshold_ = 0.5, neckHeightThreshold_ = 1.0;
    
    double handBallDist_;
    double rootBallDist_;

    int sim_step_ = 0;
    int max_sim_step_ = 2000;
    double total_reward_ = 0;

    bool contactTerminalFlag_ = false, fallFlag_ = false;
    bool fromGround_ = false, fromHand_ = false, isGround_ = false, isHand_ = false, groundHand_ = false;
    
    int nLoops_;

    Mat<3, 3> rWristOrn_;

    double desiredRootBallX_;
    double desiredRootBallY_;
    double desiredRootBallAng_;

    double rootBallAngle_;

    double rootBallAngleScale_;
};

}