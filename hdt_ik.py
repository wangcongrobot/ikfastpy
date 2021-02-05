import numpy as np
import ikfastpy
import datetime

t1 = datetime.datetime.now()

# Initialize kinematics for UR5 robot arm
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

# joint_angles = [-3.1,-1.6,1.6,-1.6,-1.6,0.] # in radians
joint_angles = [0.0,0.3,0.15,0.0,0.4,0.0] # in radians
# joint_angles = [0.0016144849504478032, 0.02851116174066437, 0.013754926474105922, -0.012710444727898285, 0.003797917117521088, -0.00937656560536304] # in radians
# joint_angles = [0.,0.,0.,0.,0.,0.] # in radians

def select(q_sols, q_d, w=[1]*6):
    """
    Select the optimal solutions among a set of feasible joint value solutions.

    Args:
        q_sols: A set of feasible joint value solutions (unit: radian)
        q_d: A list of desired joint value solution (unit: radian)
        w: A list of weight corresponding to robot joints

    Returns:
        A list of optimal joint value solution.
    """
    error = []
    for q in q_sols:
        error.append(sum([w[i] * (q[i] - q_d[i]) ** 2 for i in range(6)]))
    return q_sols[error.index(min(error))]

# Test forward kinematics: get end effector pose from joint angles
print("\nTesting forward kinematics:\n")
print("Joint angles:")
print(joint_angles)
ee_pose = ur5_kin.forward(joint_angles)
ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
print("\nEnd effector pose:")
print(ee_pose)
print("\n-----------------------------")

ee_pose[0,3] += 0.005
ee_pose[1,3] += -0.01
ee_pose[2,3] += -0.01
print(ee_pose)

# Test inverse kinematics: get joint angles from end effector pose
print("\nTesting inverse kinematics:\n")
for i in range(1):
    joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
    i += 1
    print("calculate IK")
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)

best_config = select(joint_configs, joint_angles)
print("best_config: ", best_config)
print("current: ", joint_angles)

# Check cycle-consistency of forward and inverse kinematics
# assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
# print("\nTest passed!")

t2 = datetime.datetime.now()
print("total time: ", (t2-t1).total_seconds())