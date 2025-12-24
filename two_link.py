import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
import os


dt = 1 / 240
maxTime = 5.0
kp = 8.0                  # task-space gain (увеличен для быстрее сходимости)
lam = 1e-3                # damping (DLS)
max_joint_vel = 8.0       # joint velocity limit

# Target point in 3D (reachable)
xd, yd, zd = 0.2, 0.3, 1.2

# Time vector
logTime = np.arange(0.0, maxTime, dt)
sz = len(logTime)

# Logs for position
logX = np.zeros(sz)
logY = np.zeros(sz)
logZ = np.zeros(sz)

# Logs for velocity
logXVel = np.zeros(sz)
logYVel = np.zeros(sz)
logZVel = np.zeros(sz)

# Logs for joint angles
logQ = np.zeros((sz, 3))


p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

# Load URDF from the same directory as this script
urdf_path = os.path.join(os.path.dirname(__file__), "three_link.urdf.xml")
robotId = p.loadURDF(urdf_path, useFixedBase=True)


print("=" * 60)
print("Joint map:")
print("=" * 60)
numJoints = p.getNumJoints(robotId)
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    print(f"Index {i}: {ji[1].decode():15s} | Type: {ji[2]:2d} | Child link: {ji[12].decode()}")
print("=" * 60)


name_to_idx = {}
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    name_to_idx[ji[1].decode()] = i

# Controlled joints (3 DOF)
ctrl_joint_indices = [
    name_to_idx["joint_0"],
    name_to_idx["joint_1"],
    name_to_idx["joint_2"]
]

# End-effector link index
eefLinkIdx = name_to_idx["joint_eef2"]

print(f"Controlled joints: {ctrl_joint_indices}")
print(f"End-effector link index: {eefLinkIdx}")
print("=" * 60)


dof_joint_indices = []
for i in range(numJoints):
    jtype = p.getJointInfo(robotId, i)[2]
    if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        dof_joint_indices.append(i)

numDof = len(dof_joint_indices)
print(f"DOF joints: {dof_joint_indices}")
print(f"Total DOF: {numDof}")
print("=" * 60)

# Map: bullet joint index -> Jacobian column
dof_index_map = {jid: k for k, jid in enumerate(dof_joint_indices)}
ctrl_cols = [dof_index_map[j] for j in ctrl_joint_indices]

def get_q_dq_dof():
    """Return q and dq vectors of size = numDof (Bullet requirement)."""
    js = p.getJointStates(robotId, dof_joint_indices)
    q = [s[0] for s in js]
    dq = [s[1] for s in js]
    return q, dq


initial_pos = [0.3, 0.5, 0.5]
p.setJointMotorControlArray(
    robotId,
    ctrl_joint_indices,
    controlMode=p.POSITION_CONTROL,
    targetPositions=initial_pos,
    forces=[50.0] * 3
)

print("Initializing robot position...")
for _ in range(1000):
    p.stepSimulation()
    time.sleep(dt / 10)

print("Starting control loop...")
print("=" * 60)


for k in range(sz):
    # Current end-effector position
    linkState = p.getLinkState(robotId, eefLinkIdx, computeLinkVelocity=True)
    pos = np.array(linkState[0], dtype=float)
    vel = np.array(linkState[6], dtype=float)  # linear velocity

    # Log position
    logX[k], logY[k], logZ[k] = pos
    
    # Log velocity
    logXVel[k], logYVel[k], logZVel[k] = vel
    
    # Log joint angles
    js = p.getJointStates(robotId, ctrl_joint_indices)
    logQ[k, :] = [s[0] for s in js]

    # Cartesian error (current - desired)
    err = (pos - np.array([xd, yd, zd])).reshape(3, 1)

    # DOF-sized vectors
    q_dof, dq_dof = get_q_dq_dof()

    # Jacobian from Bullet
    jac_t, _ = p.calculateJacobian(
        robotId,
        eefLinkIdx,
        [0.0, 0.0, 0.0],
        q_dof,
        dq_dof,
        [0.0] * numDof
    )

    Jt = np.array(jac_t, dtype=float)      # 3 x numDof
    J = Jt[:, ctrl_cols]                   # 3 x 3 (controlled joints)

    # Damped Least Squares pseudo-inverse
    JJt = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJt + (lam * lam) * np.eye(3))

    # Joint velocity command
    w = (-kp) * (J_pinv @ err)
    w = np.clip(w.flatten(), -max_joint_vel, max_joint_vel)

    # Apply velocity control
    p.setJointMotorControlArray(
        robotId,
        ctrl_joint_indices,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=w.tolist(),
        forces=[50.0] * 3
    )

    p.stepSimulation()
    time.sleep(dt)  # Реальное время симуляции
    
    # Print progress
    if k % 240 == 0:
        error_norm = np.linalg.norm(err)
        print(f"Time: {logTime[k]:.2f}s | Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Error: {error_norm:.4f}")

# Final stats
final_error = np.linalg.norm(np.array([logX[-1], logY[-1], logZ[-1]]) - np.array([xd, yd, zd]))
print("=" * 60)
print(f"Final position: [{logX[-1]:.3f}, {logY[-1]:.3f}, {logZ[-1]:.3f}]")
print(f"Target position: [{xd:.3f}, {yd:.3f}, {zd:.3f}]")
print(f"Final error: {final_error:.4f}")
print("=" * 60)


fig = plt.figure(figsize=(16, 10))
fig.suptitle('Three-Link Robot Control Results', fontsize=16, fontweight='bold')

# Row 1: Position plots (X, Y, Z)
plt.subplot(3, 4, 1)
plt.title("X Position")
plt.grid(True)
plt.plot(logTime, logX, label="simX", linewidth=2)
plt.plot([logTime[0], logTime[-1]], [xd, xd], 'r--', label='refX', linewidth=2)
plt.ylabel("X [m]")
plt.legend(fontsize=8)

plt.subplot(3, 4, 2)
plt.title("Y Position")
plt.grid(True)
plt.plot(logTime, logY, label="simY", linewidth=2)
plt.plot([logTime[0], logTime[-1]], [yd, yd], 'r--', label='refY', linewidth=2)
plt.ylabel("Y [m]")
plt.legend(fontsize=8)

plt.subplot(3, 4, 3)
plt.title("Z Position")
plt.grid(True)
plt.plot(logTime, logZ, label="simZ", linewidth=2)
plt.plot([logTime[0], logTime[-1]], [zd, zd], 'r--', label='refZ', linewidth=2)
plt.ylabel("Z [m]")
plt.legend(fontsize=8)

# XY Cartesian Path
plt.subplot(3, 4, 4)
plt.title("XY Path")
plt.plot(logX, logY, 'b-', linewidth=2, label='Traj')
plt.plot(logX[0], logY[0], 'go', markersize=8, label='Start')
plt.plot(xd, yd, 'r*', markersize=12, label='Target')
plt.plot(logX[-1], logY[-1], 'bs', markersize=6, label='End')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True)
plt.axis('equal')
plt.legend(fontsize=8)

# Row 2: Velocity plots
plt.subplot(3, 4, 5)
plt.title("X Velocity")
plt.plot(logTime, logXVel, linewidth=2)
plt.ylabel("Vel [m/s]")
plt.grid(True)

plt.subplot(3, 4, 6)
plt.title("Y Velocity")
plt.plot(logTime, logYVel, linewidth=2)
plt.ylabel("Vel [m/s]")
plt.grid(True)

plt.subplot(3, 4, 7)
plt.title("Z Velocity")
plt.plot(logTime, logZVel, linewidth=2)
plt.ylabel("Vel [m/s]")
plt.grid(True)

# XZ Cartesian Path
plt.subplot(3, 4, 8)
plt.title("XZ Path")
plt.plot(logX, logZ, 'b-', linewidth=2, label='Traj')
plt.plot(logX[0], logZ[0], 'go', markersize=8, label='Start')
plt.plot(xd, zd, 'r*', markersize=12, label='Target')
plt.plot(logX[-1], logZ[-1], 'bs', markersize=6, label='End')
plt.xlabel("X [m]")
plt.ylabel("Z [m]")
plt.grid(True)
plt.legend(fontsize=8)

# Row 3: Joint angles
plt.subplot(3, 4, 9)
plt.title("Joint 0 Angle (Yaw)")
plt.plot(logTime, logQ[:, 0], linewidth=2)
plt.ylabel("Angle [rad]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.subplot(3, 4, 10)
plt.title("Joint 1 Angle")
plt.plot(logTime, logQ[:, 1], linewidth=2)
plt.ylabel("Angle [rad]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.subplot(3, 4, 11)
plt.title("Joint 2 Angle")
plt.plot(logTime, logQ[:, 2], linewidth=2)
plt.ylabel("Angle [rad]")
plt.xlabel("Time [s]")
plt.grid(True)

# 3D trajectory (optional bonus)
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(3, 4, 12, projection='3d')
ax.plot(logX, logY, logZ, 'b-', linewidth=2, label='Trajectory')
ax.scatter(logX[0], logY[0], logZ[0], c='g', marker='o', s=100, label='Start')
ax.scatter(xd, yd, zd, c='r', marker='*', s=200, label='Target')
ax.scatter(logX[-1], logY[-1], logZ[-1], c='b', marker='s', s=80, label='End')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Path')
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.show()
p.disconnect()