import numpy as np
import ikfastpy

# Initialize kinematics for hdt robot arm
hdt_kin = ikfastpy.PyKinematics()
n_joints = hdt_kin.getDOF()

# joint_angles = [-3.1,-1.6,1.6,-1.6,-1.6,0.] # in radians
joint_angles = [0,0,0,0,0,0] # in radians, home
joint_angles = [0,0.3,0.15,0,0.4,0] # in radians, prepare
joint_angles = [0,0.3,0.15,0,-1.1,0] # in radians, pick_down
joint_angles = [0.2,0.2,0.15,0,-1.1,0] # in radians, test

# Test forward kinematics: get end effector pose from joint angles
print("\nTesting forward kinematics:\n")
print("Joint angles:")
print(joint_angles)
ee_pose = hdt_kin.forward(joint_angles)
ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
print("\nEnd effector pose:")
print(ee_pose)
print("\n-----------------------------")

# Test inverse kinematics: get joint angles from end effector pose
print("\nTesting inverse kinematics:\n")
joint_configs = hdt_kin.inverse(ee_pose.reshape(-1).tolist())
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)

# Check cycle-consistency of forward and inverse kinematics
assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
print("\nTest passed!")