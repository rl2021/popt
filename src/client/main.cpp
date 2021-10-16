#include "iostream"
#include "../lopt.h"
#include "gazebo_msgs/ModelStates.h"
#include "sensor_msgs/JointState.h"
#include <tf/tf.h>
#include "tf_conversions/tf_eigen.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <iDynTree/Model/FreeFloatingState.h>
#include <iDynTree/KinDynComputations.h>
#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/Core/EigenHelpers.h>

#include <ros/package.h>
#include "std_msgs/Float64MultiArray.h"
#include "boost/thread.hpp"
#include "gazebo_msgs/SetModelState.h"
#include "gazebo_msgs/SetModelConfiguration.h"
#include <std_srvs/Empty.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>
#include <towr/terrain/examples/height_map_examples.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>
#include <towr/initialization/gait_generator.h>
#include <map>
#include <unistd.h>
#include <unordered_map>
#include "gazebo_msgs/ContactsState.h"
#include "../topt.h"


#include "boost/thread.hpp"

using namespace std;

class DOGCTRL {
    public:
        DOGCTRL();
        void jointStateCallback(const sensor_msgs::JointState & msg);
        void modelStateCallback(const gazebo_msgs::ModelStates & msg);
        void run();
        void ctrl_loop();
        void publish_cmd(  Eigen::VectorXd tau  );
        void eebr_cb(gazebo_msgs::ContactsStateConstPtr eebr);
        void eebl_cb(gazebo_msgs::ContactsStateConstPtr eebl);
        void eefr_cb(gazebo_msgs::ContactsStateConstPtr eefr);
        void eefl_cb(gazebo_msgs::ContactsStateConstPtr eefl);
        void createrobot(std::string modelFile);
        // Compute the Jacobian
        void  computeJac();
        void  ComputeJaclinear();
        // Compute matrix transformation T needed to recompute matrices/vecotor after the coordinate transform to the CoM
        void computeTransformation(const Eigen::VectorXd &Vel_);
        void computeJacDotQDot();
        void computeJdqdCOMlinear();
        void update(Eigen::Matrix4d &eigenWorld_H_base, Eigen::Matrix<double,12,1> &eigenJointPos, Eigen::Matrix<double,12,1> &eigenJointVel, Eigen::Matrix<double,6,1> &eigenBasevel, Eigen::Vector3d &eigenGravity);			


    private:
        ros::NodeHandle _nh;
        ros::Subscriber _joint_state_sub; 
        ros::Subscriber _model_state_sub; 
        ros::Subscriber _eebl_sub;
        ros::Subscriber _eebr_sub;
        ros::Subscriber _eefl_sub;
        ros::Subscriber _eefr_sub;
        ros::Publisher  _joint_pub;

        Eigen::Matrix4d _world_H_base;
        Eigen::Matrix<double,12,1> _jnt_pos; 
        Eigen::Matrix<double,12,1> _jnt_vel;
        Eigen::Matrix<double,6,1> _base_pos;
        Eigen::Matrix<double,6,1> _base_vel;

        string _model_name;
        OPT *_o;
        
        bool _first_wpose;
        bool _first_jpos;
        unordered_map<int, string> _id2idname;  
        unordered_map<int, int> _id2index;      
        unordered_map<int, int> _index2id;      
        Eigen::VectorXd x_eigen;
       

        bool _contact_br;
        bool _contact_bl;
        bool _contact_fl;
        bool _contact_fr;


        // int for DoFs number
        unsigned int n;
        // Total mass of the robot
        double robot_mass;
        // KinDynComputations element
        iDynTree::KinDynComputations kinDynComp;
        // world to floating base transformation
        iDynTree::Transform world_H_base;
        // Joint position
        iDynTree::VectorDynSize jointPos;
        // Floating base velocity
        iDynTree::Twist         baseVel;
        // Joint velocity
        iDynTree::VectorDynSize jointVel;
        // Gravity acceleration
        iDynTree::Vector3       gravity; 
        // Position vector base+joints
        iDynTree::VectorDynSize  qb;
        // Velocity vector base+joints
        iDynTree::VectorDynSize  dqb;
        // Position vector COM+joints
        iDynTree::VectorDynSize  q;
        // Velocity vector COM+joints
        iDynTree::VectorDynSize  dq;
        // Joints limit vector
        iDynTree::VectorDynSize  qmin;
        iDynTree::VectorDynSize  qmax;
        // Center of Mass Position
        iDynTree::Vector6 CoM;
        // Center of mass velocity
        iDynTree::Vector6 CoM_vel;
        //Mass matrix
        iDynTree::FreeFloatingMassMatrix MassMatrix;
        //Bias Matrix
        iDynTree::VectorDynSize Bias;
        //Gravity Matrix
        iDynTree::MatrixDynSize GravMatrix;
        // Jacobian
        iDynTree::MatrixDynSize Jac;
        // Jacobian derivative
        iDynTree::MatrixDynSize JacDot;
        //CoM Jacobian
        iDynTree::MatrixDynSize Jcom;
        // Bias acceleration J_dot*q_dot
        iDynTree::MatrixDynSize Jdqd;
        // Transformation Matrix
        iDynTree::MatrixDynSize T;
        // Transformation matrix time derivative
        iDynTree::MatrixDynSize T_inv_dot;
        //Model
        iDynTree::Model model;
        iDynTree::ModelLoader mdlLoader;
        //Mass matrix in CoM representation
        iDynTree::FreeFloatingMassMatrix MassMatrixCOM;
        //Bias Matrix in CoM representation
        iDynTree::VectorDynSize BiasCOM;
        //Gravity Matrix in CoM representation
        iDynTree::MatrixDynSize GravMatrixCOM;
        // Jacobian in CoM representation
        iDynTree::MatrixDynSize JacCOM;
        //Jacobian in CoM representation (only linear part)
        iDynTree::MatrixDynSize JacCOM_lin;
        // Bias acceleration J_dot*q_dot in CoM representation
        iDynTree::MatrixDynSize JdqdCOM;
        // Bias acceleration J_dot*q_dot in CoM representation
        iDynTree::MatrixDynSize JdqdCOM_lin;
        

};



DOGCTRL::DOGCTRL() {
   
    _joint_state_sub = _nh.subscribe("/dogbot/joint_states", 1, &DOGCTRL::jointStateCallback, this);
    _model_state_sub = _nh.subscribe("/gazebo/model_states", 1, &DOGCTRL::modelStateCallback, this);
    _eebl_sub = _nh.subscribe("/dogbot/back_left_contactsensor_state",1, &DOGCTRL::eebl_cb, this);
    _eefl_sub = _nh.subscribe("/dogbot/front_left_contactsensor_state",1, &DOGCTRL::eefl_cb, this);
    _eebr_sub = _nh.subscribe("/dogbot/back_right_contactsensor_state",1, &DOGCTRL::eebr_cb, this);
    _eefr_sub = _nh.subscribe("/dogbot/front_right_contactsensor_state",1,&DOGCTRL::eefr_cb, this);
    
    _joint_pub = _nh.advertise<std_msgs::Float64MultiArray>("/dogbot/joint_position_controller/command", 1);
    _model_name = "dogbot";

    _first_wpose = false;
    _first_jpos = false;
    _contact_br = true; 
    _contact_bl = true; 
    _contact_bl = true; 
    _contact_fr = true;


    std::string path = ros::package::getPath("dogbot_description");
    path += "/urdf/dogbot.urdf";

    createrobot(path);

    model = kinDynComp.model();
	kinDynComp.setFrameVelocityRepresentation(iDynTree::MIXED_REPRESENTATION);
	// Resize matrices of the class given the number of DOFs
    n = model.getNrOfDOFs();
    
    robot_mass = model.getTotalMass();
    jointPos = iDynTree::VectorDynSize(n);
    baseVel = iDynTree::Twist();
    jointVel = iDynTree::VectorDynSize(n);
	q = iDynTree::VectorDynSize(6+n);
	dq = iDynTree::VectorDynSize(6+n);
	qb = iDynTree::VectorDynSize(6+n);
	dqb=iDynTree::VectorDynSize(6+n);
	qmin= iDynTree::VectorDynSize(n);
	qmax= iDynTree::VectorDynSize(n);
	Bias=iDynTree::VectorDynSize(n+6);
	GravMatrix=iDynTree::MatrixDynSize(n+6,1);
    MassMatrix=iDynTree::FreeFloatingMassMatrix(model) ;
    Jcom=iDynTree::MatrixDynSize(3,6+n);
	Jac=iDynTree::MatrixDynSize(24,6+n);	
	JacDot=iDynTree::MatrixDynSize(24,6+n);
	Jdqd=iDynTree::MatrixDynSize(24,1);
    T=iDynTree::MatrixDynSize(6+n,6+n);
	T_inv_dot=iDynTree::MatrixDynSize(6+n,6+n);
    MassMatrixCOM=iDynTree::FreeFloatingMassMatrix(model) ;
    BiasCOM=iDynTree::VectorDynSize(n+6);
	GravMatrixCOM=iDynTree::MatrixDynSize(n+6,1);
	JacCOM=iDynTree::MatrixDynSize(24,6+n);
	JacCOM_lin=iDynTree::MatrixDynSize(12,6+n);
	JdqdCOM=iDynTree::MatrixDynSize(24,1);
	JdqdCOM_lin=iDynTree::MatrixDynSize(12,1);
	x_eigen= Eigen::VectorXd::Zero(30);
    //---
    /*  TODO:
     *  Initialize OPTIMIZATION objcet _o
     *  OPT object takes as input:
     *      Number of control variables, number of stance contraints, number of swing contraints
     *  _o = new OPT( ... );
     *
     *  Remember to set the Q Matrix, the c vector and the contraints of the problem, before to call the optimization
     */

}



void DOGCTRL::createrobot(std::string modelFile) {  
    
    if( !mdlLoader.loadModelFromFile(modelFile) ) {
        std::cerr << "KinDynComputationsWithEigen: impossible to load model from " << modelFile << std::endl;
        return ;
    }
    if( !kinDynComp.loadRobotModel(mdlLoader.model()) )
    {
        std::cerr << "KinDynComputationsWithEigen: impossible to load the following model in a KinDynComputations class:" << std::endl
                  << mdlLoader.model().toString() << std::endl;
        return ;
    }

    _id2idname.insert( pair< int, string > ( 0, kinDynComp.getDescriptionOfDegreeOfFreedom(0) ));
    _id2idname.insert( pair< int, string > ( 1, kinDynComp.getDescriptionOfDegreeOfFreedom(1) ));
    _id2idname.insert( pair< int, string > ( 2, kinDynComp.getDescriptionOfDegreeOfFreedom(2) ));
    _id2idname.insert( pair< int, string > ( 3, kinDynComp.getDescriptionOfDegreeOfFreedom(3) ));
    _id2idname.insert( pair< int, string > ( 4, kinDynComp.getDescriptionOfDegreeOfFreedom(4) ));
    _id2idname.insert( pair< int, string > ( 5, kinDynComp.getDescriptionOfDegreeOfFreedom(5) ));
    _id2idname.insert( pair< int, string > ( 6, kinDynComp.getDescriptionOfDegreeOfFreedom(6) ));
    _id2idname.insert( pair< int, string > ( 7, kinDynComp.getDescriptionOfDegreeOfFreedom(7) ));
    _id2idname.insert( pair< int, string > ( 8, kinDynComp.getDescriptionOfDegreeOfFreedom(8) ));
    _id2idname.insert( pair< int, string > ( 9, kinDynComp.getDescriptionOfDegreeOfFreedom(9) ));
    _id2idname.insert( pair< int, string > ( 10, kinDynComp.getDescriptionOfDegreeOfFreedom(10) ));
    _id2idname.insert( pair< int, string > ( 11, kinDynComp.getDescriptionOfDegreeOfFreedom(11) ));
}


// Compute matrix transformation T needed to recompute matrices/vector after the coordinate transform to the CoM
void DOGCTRL::computeTransformation(const Eigen::VectorXd &Vel_) {

    //Set ausiliary matrices
    iDynTree::MatrixDynSize Jb(6,6+n);
    iDynTree::MatrixDynSize Jbc(3,n);
    iDynTree::Vector3 xbc;
    iDynTree::MatrixDynSize xbc_hat(3,3);
    iDynTree::MatrixDynSize xbc_hat_dot(3,3);
    iDynTree::MatrixDynSize Jbc_dot(6,6+n);
    iDynTree::Vector3 xbo_dot;

    //Set ausiliary matrices
    iDynTree::Vector3 xbc_dot;

    // Compute T matrix
    // Get jacobians of the floating base and of the com
    kinDynComp.getFrameFreeFloatingJacobian(0,Jb);
    kinDynComp.getCenterOfMassJacobian(Jcom);

    // Compute jacobian Jbc=d(xc-xb)/dq used in matrix T
    toEigen(Jbc)<<toEigen(Jcom).block<3,12>(0,6)-toEigen(Jb).block<3,12>(0,6);

    // Get xb (floating base position) and xc ( com position)
    iDynTree::Position xb = world_H_base.getPosition();
    iDynTree::Position xc= kinDynComp.getCenterOfMassPosition();

    // Vector xcb=xc-xb
    toEigen(xbc)=toEigen(xc)-toEigen(xb);

    // Skew of xcb
    toEigen(xbc_hat)<<0, -toEigen(xbc)[2], toEigen(xbc)[1],
    toEigen(xbc)[2], 0, -toEigen(xbc)[0],                          
    -toEigen(xbc)[1], toEigen(xbc)[0], 0;

    Eigen::Matrix<double,6,6> X;
    X<<Eigen::MatrixXd::Identity(3,3), toEigen(xbc_hat).transpose(), 
    Eigen::MatrixXd::Zero(3,3), Eigen::MatrixXd::Identity(3,3);

    Eigen::MatrixXd Mb_Mj= toEigen(MassMatrix).block(0,0,6,6).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(toEigen(MassMatrix).block(0,6,6,12));
    Eigen::Matrix<double,6,12> Js=X*Mb_Mj;

    // Matrix T for the transformation
    toEigen(T)<<Eigen::MatrixXd::Identity(3,3), toEigen(xbc_hat).transpose(), Js.block(0,0,3,12),
    Eigen::MatrixXd::Zero(3,3), Eigen::MatrixXd::Identity(3,3), Js.block(3,0,3,12),
    Eigen::MatrixXd::Zero(12,3),  Eigen::MatrixXd::Zero(12,3), Eigen::MatrixXd::Identity(12,12);

    //Compute time derivative of T 
    // Compute derivative of xbc
    toEigen(xbc_dot)=toEigen(kinDynComp.getCenterOfMassVelocity())-toEigen(baseVel.getLinearVec3());
    Eigen::VectorXd  mdr=robot_mass*toEigen(xbc_dot);
    Eigen::Matrix<double,3,3> mdr_hat;
    mdr_hat<<0, -mdr[2], mdr[1],
    mdr[2], 0, -mdr[0],                          
    -mdr[1], mdr[0], 0;

    //Compute skew of xbc
    toEigen(xbc_hat_dot)<<0, -toEigen(xbc_dot)[2], toEigen(xbc_dot)[1],
    toEigen(xbc_dot)[2], 0, -toEigen(xbc_dot)[0],                          
    -toEigen(xbc_dot)[1], toEigen(xbc_dot)[0], 0;

    Eigen::Matrix<double,6,6> dX;
    dX<<Eigen::MatrixXd::Zero(3,3), toEigen(xbc_hat_dot).transpose(),
    Eigen::MatrixXd::Zero(3,6);
    // Time derivative of Jbc
    kinDynComp.getCentroidalAverageVelocityJacobian(Jbc_dot);

    Eigen::Matrix<double,6,6> dMb;
    dMb<<Eigen::MatrixXd::Zero(3,3), mdr_hat.transpose(),
    mdr_hat, Eigen::MatrixXd::Zero(3,3);

    Eigen::MatrixXd inv_dMb1=(toEigen(MassMatrix).block(0,0,6,6).transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(dMb.transpose())).transpose();
    Eigen::MatrixXd inv_dMb2=-(toEigen(MassMatrix).block(0,0,6,6).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve( inv_dMb1));

    Eigen::Matrix<double,6,12> dJs=dX*Mb_Mj+X*inv_dMb2*toEigen(MassMatrix).block(0,6,6,12);

    toEigen(T_inv_dot)<<Eigen::MatrixXd::Zero(3,3), toEigen(xbc_hat_dot), -dJs.block(0,0,3,12),
    Eigen::MatrixXd::Zero(15,18);

}



// Compute Jacobian
void  DOGCTRL::computeJac() {     

    //Set ausiliary matrices
    iDynTree::MatrixDynSize Jac1(6,6+n);
    iDynTree::MatrixDynSize Jac2(6,6+n);
    iDynTree::MatrixDynSize Jac3(6,6+n);
    iDynTree::MatrixDynSize Jac4(6,6+n);

    // Compute Jacobian for each leg

    // Jacobian for back right leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("back_right_foot"), Jac1);

    // Jacobian for back left leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("back_left_foot"),Jac2);

    // Jacobian for front left leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("front_left_foot"), Jac3);

    // Jacobian for front right leg
    kinDynComp.getFrameFreeFloatingJacobian( kinDynComp.getFrameIndex("front_right_foot"), Jac4);

    // Full Jacobian
    toEigen(Jac)<<toEigen(Jac1), toEigen(Jac2), toEigen(Jac3), toEigen(Jac4);
    
}

void DOGCTRL::ComputeJaclinear() {
    
  Eigen::Matrix<double,12,24> B;
  B<< Eigen::MatrixXd::Identity(3,3) , Eigen::MatrixXd::Zero(3,21),
      Eigen::MatrixXd::Zero(3,6), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,15),
	  Eigen::MatrixXd::Zero(3,12), Eigen::MatrixXd::Identity(3,3),  Eigen::MatrixXd::Zero(3,9),
	  Eigen::MatrixXd::Zero(3,18), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,3);

  toEigen(JacCOM_lin)=B*toEigen(JacCOM);
    
}




void DOGCTRL::computeJdqdCOMlinear()
{
	Eigen::Matrix<double,12,24> B;
    B<< Eigen::MatrixXd::Identity(3,3) , Eigen::MatrixXd::Zero(3,21),
      Eigen::MatrixXd::Zero(3,6), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,15),
	  Eigen::MatrixXd::Zero(3,12), Eigen::MatrixXd::Identity(3,3),  Eigen::MatrixXd::Zero(3,9),
	  Eigen::MatrixXd::Zero(3,18), Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Zero(3,3);


    toEigen(JdqdCOM_lin)= Eigen::MatrixXd::Zero(12,1);
    toEigen(JdqdCOM_lin)=B*toEigen(JdqdCOM);
	
}



// Compute Bias acceleration: J_dot*q_dot
void  DOGCTRL::computeJacDotQDot() {
    

    // Bias acceleration for back right leg
    iDynTree::Vector6 Jdqd1=kinDynComp.getFrameBiasAcc("back_right_foot"); 

    // Bias acceleration for back left leg
    iDynTree::Vector6 Jdqd2=kinDynComp.getFrameBiasAcc("back_left_foot"); 

    // Bias acceleration for front left leg
    iDynTree::Vector6 Jdqd3=kinDynComp.getFrameBiasAcc("front_left_foot"); 
    // Bias acceleration for front right leg
    iDynTree::Vector6 Jdqd4=kinDynComp.getFrameBiasAcc("front_right_foot"); 
    toEigen(Jdqd)<<toEigen(Jdqd1), toEigen(Jdqd2), toEigen(Jdqd3), toEigen(Jdqd4);

	
}

// Get joints position and velocity
void DOGCTRL::jointStateCallback(const sensor_msgs::JointState & msg) {

    if( _first_jpos == false ) {

        for( int i=0; i<12; i++) {
            bool found = false;
            int index = 0;
            while( !found && index <  msg.name.size() ) {
                if( msg.name[index] == _id2idname.at( i )    ) {
                    found = true;

                    _id2index.insert( pair< int, int > ( i, index ));
                    _index2id.insert( pair< int, int > ( index, i ));

                }
                else index++;
            }
        }
    }

    for( int i=0; i<12; i++ ) {
        _jnt_pos( i, 0) = msg.position[    _id2index.at(i)    ];
    }

    for( int i=0; i<12; i++ ) {
        _jnt_vel( i, 0) = msg.velocity[    _id2index.at(i)    ];
    }
    
    _first_jpos = true;
}


// Get base position and velocity
void DOGCTRL::modelStateCallback(const gazebo_msgs::ModelStates & msg) {


    bool found = false;
    int index = 0;
    while( !found  && index < msg.name.size() ) {

        if( msg.name[index] == _model_name )
            found = true;
        else index++;
    }

    if( found ) {
        
        _world_H_base.setIdentity();
        
        //quaternion
        tf::Quaternion q(msg.pose[index].orientation.x, msg.pose[index].orientation.y, msg.pose[index].orientation.z,  msg.pose[index].orientation.w);
        q.normalize();
        Eigen::Matrix<double,3,3> rot;
        tf::matrixTFToEigen(tf::Matrix3x3(q),rot);

        //Roll, pitch, yaw
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

        //Set base pos (position and orientation)
        _base_pos << msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z, roll, pitch, yaw;
        //Set transformation matrix
        _world_H_base.block(0,0,3,3)= rot;
        _world_H_base.block(0,3,3,1)= _base_pos.block(0,0,3,1);

        //Set base vel
        _base_vel << msg.twist[index].linear.x, msg.twist[index].linear.y, msg.twist[index].linear.z, msg.twist[index].angular.x, msg.twist[index].angular.y, msg.twist[index].angular.z;
        _first_wpose = true;
    }
}





//Update elements of the class given the new state

void DOGCTRL::update (Eigen::Matrix4d &eigenWorld_H_base, Eigen::Matrix<double,12,1> &eigenJointPos, Eigen::Matrix<double,12,1> &eigenJointVel, Eigen::Matrix<double,6,1> &eigenBasevel, Eigen::Vector3d &eigenGravity)
{   

   
    // Update joints, base and gravity from inputs

    iDynTree::fromEigen(world_H_base,eigenWorld_H_base);
    iDynTree::toEigen(jointPos) = eigenJointPos;
    iDynTree::fromEigen(baseVel,eigenBasevel);
    toEigen(jointVel) = eigenJointVel;
    toEigen(gravity)  = eigenGravity;

    //Set the state for the robot 
    kinDynComp.setRobotState(world_H_base,jointPos,
    baseVel,jointVel,gravity);


    // Compute Center of Mass
    iDynTree::Vector3 base_angle=world_H_base.getRotation().asRPY();
    toEigen(CoM)<<toEigen(kinDynComp.getCenterOfMassPosition()),
    toEigen(base_angle);

    
	//Compute velocity of the center of mass

	toEigen(CoM_vel)<<toEigen(kinDynComp.getCenterOfMassVelocity()), eigenBasevel.block(3,0,3,1);
		   
    // Compute position base +joints
	toEigen(qb)<<toEigen(world_H_base.getPosition()), toEigen(base_angle), eigenJointPos;
    // Compute position COM+joints
	toEigen(q)<<toEigen(CoM), eigenJointPos;
   	toEigen(dq)<<toEigen(CoM_vel), eigenJointVel;
	toEigen(dqb) << eigenBasevel, eigenJointVel;
   
	// Joint limits

    toEigen(qmin)<< -1.75 , -1.75,-1.75,-1.75,-1.58, -2.62, -3.15, -0.02,  -1.58, -2.62, -3.15, -0.02;
    toEigen(qmax)<< 1.75, 1.75, 1.75, 1.75, 3.15, 0.02, 1.58, 2.62,  3.15, 0.02, 1.58, 2.62;

    // Get mass, bias (C(q,v)*v+g(q)) and gravity (g(q)) matrices
    //Initialize ausiliary vector
    iDynTree::FreeFloatingGeneralizedTorques bias_force(model);
    iDynTree::FreeFloatingGeneralizedTorques grav_force(model);
    //Compute Mass Matrix
    kinDynComp.getFreeFloatingMassMatrix(MassMatrix); 
    //Compute Coriolis + gravitational terms (Bias)
    kinDynComp.generalizedBiasForces(bias_force);
    toEigen(Bias)<<iDynTree::toEigen(bias_force.baseWrench()),
        iDynTree::toEigen(bias_force.jointTorques());

    
    //Compute Gravitational term
    kinDynComp.generalizedGravityForces(grav_force);
    toEigen(GravMatrix)<<iDynTree::toEigen(grav_force.baseWrench()),
            iDynTree::toEigen(grav_force.jointTorques());

    computeJac();	
    // Compute Bias Acceleration -> J_dot*q_dot
    computeJacDotQDot();
    
    Eigen::Matrix<double, 18,1> q_dot;

    q_dot<< eigenBasevel,
            eigenJointVel;

    // Compute Matrix needed for transformation from floating base representation to CoM representation
    computeTransformation(q_dot);
    // Compute Mass Matrix in CoM representation 
    toEigen(MassMatrixCOM)=toEigen(T).transpose().inverse()*toEigen(MassMatrix)*toEigen(T).inverse();
    // Compute Coriolis+gravitational term in CoM representation
    toEigen(BiasCOM)=toEigen(T).transpose().inverse()*toEigen(Bias)+toEigen(T).transpose().inverse()*toEigen(MassMatrix)*toEigen(T_inv_dot)*toEigen(dq);

    // Compute gravitational term in CoM representation	
    toEigen(GravMatrixCOM)=toEigen(T).transpose().inverse()*toEigen(GravMatrix);
    // Compute Jacobian term in CoM representation
    toEigen(JacCOM)=toEigen(Jac)*toEigen(T).inverse();
    ComputeJaclinear();
    // Compute Bias Acceleration -> J_dot*q_dot  in CoM representation
    toEigen(JdqdCOM)=toEigen(Jdqd)+toEigen(Jac)*toEigen(T_inv_dot)*toEigen(dq);
    computeJdqdCOMlinear();	
}

void  DOGCTRL::publish_cmd(  Eigen::VectorXd tau  ) {
    std_msgs::Float64MultiArray tau1_msg;
    
    // Fill Command message
    for(int i=11; i>=0; i--) {
        tau1_msg.data.push_back(  tau( _index2id.at(i) )    );
    }

    //Sending command
    _joint_pub.publish(tau1_msg);

}



void DOGCTRL::eebr_cb(gazebo_msgs::ContactsStateConstPtr eebr){
	if(eebr->states.empty()){ 
        _contact_br= false;
	}
	else {
		_contact_br= true;
	}
}

void DOGCTRL::eefl_cb(gazebo_msgs::ContactsStateConstPtr eefl){

	if(eefl->states.empty()){ 
        _contact_fl= false;
	}
	else {
		_contact_fl= true;
    }
}

void DOGCTRL::eebl_cb(gazebo_msgs::ContactsStateConstPtr eebl){

	if(eebl->states.empty()){ 
        _contact_bl= false;
	}
	else {
	    _contact_bl= true;
    }
}

void DOGCTRL::eefr_cb(gazebo_msgs::ContactsStateConstPtr eefr){
	if(eefr->states.empty()){ 
        _contact_fr= false;
	}
	else {
		_contact_fr= true;
	}
}
 
void DOGCTRL::ctrl_loop() {

    ros::ServiceClient pauseGazebo = _nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    ros::ServiceClient unpauseGazebo = _nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    std_srvs::Empty pauseSrv;
    
    //wait for first data...
    while( !_first_wpose  )
        usleep(0.1*1e6);

    while( !_first_jpos  )
        usleep(0.1*1e6);



    //Update robot state using the update function
    Eigen::Vector3d gravity;
    gravity << 0, 0, -9.8;
    update(_world_H_base, _jnt_pos, _jnt_vel, _base_vel, gravity);



    //TODO: Set the control sampling rate
    //ros::Rate r( /* SAMPLING RATE */ );
    while( ros::ok() ) {


        //If towr takes too much time, you can stop the simulation, use the trajectory generator function and restart the simulation
        if(pauseGazebo.call(pauseSrv))
            ROS_INFO("Simulation paused.");
        else
            ROS_INFO("Failed to pause simulation.");



        //TODO: Use this function to generate a trajectory
        // get_trajectory( /* ... */ );

        unpauseGazebo.call(pauseSrv); 

        /*
         *  TODO: Write your control here!
         *  remember to update the state of the robot as made previously
         *
         */

        /*
         *  TODO: Set the control torque
         *  publish_cmd( tau );
         */

        //r.sleep();        
    }
}


void DOGCTRL::run() {
    ros::spin();
}

int main(int argc, char** argv) {
    ros::init( argc, argv, "popt");
    DOGCTRL dc;
    dc.run();
}