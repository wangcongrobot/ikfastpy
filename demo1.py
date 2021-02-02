import numpy as np
import ikfastpy
import datetime

# Initialize kinematics for UR5 robot arm
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

joint_angles = [1.5505553484, -1.96206027666, 1.51348638535, 0.36519753933, 1.54755938053, 0.03253005445] # in radians

def select1(q_solutions, q_ref):
    for joint_config in q_solutions:
        results = np.sum((joint_config-q_ref)**2)
        print("results: ", results)

def select(q_sols, q_d, w=[1]*6):
    """Select the optimal solutions among a set of feasible joint value 
    solutions.

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
    
def main():
    # Test forward kinematics: get end effector pose from joint angles
    print("\nTesting forward kinematics:\n")
    print("Joint angles:")
    print(joint_angles)
    ee_pose = ur5_kin.forward(joint_angles)
    ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
    print("\nEnd effector pose:")
    print(ee_pose)
    print("\n-----------------------------")

    ee_pose[0,3] += 0.05
    ee_pose[1,3] += 0.05
    ee_pose[2,3] += 0.05
    print(ee_pose)

    # Test inverse kinematics: get joint angles from end effector pose
    print("\nTesting inverse kinematics:\n")
    t1 = datetime.datetime.now()
    joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
    n_solutions = int(len(joint_configs)/n_joints)
    t2 = datetime.datetime.now()
    t = t2 - t1
    print("time: {} us",t.microseconds/8.0)
    print("%d solutions found:"%(n_solutions))
    joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
    for joint_config in joint_configs:
        print(joint_config)

    q_best = select(joint_configs, joint_angles)
    select1(joint_configs, joint_angles)
    print("best solution: ", q_best)

    # Check cycle-consistency of forward and inverse kinematics
    assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
    print("\nTest passed!")

main()
