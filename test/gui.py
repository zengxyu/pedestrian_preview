def guiControl(self, total_step_num=10000, step_duration=1 / 240.):
    step_num = 0
    while step_num < total_step_num:

        step_num += 1
        # time.sleep(0.01)
        left_wheel_velocity = p.readUserDebugParameter(self.left_wheel_gui_id)
        right_wheel_velocity = p.readUserDebugParameter(self.right_wheel_gui_id)
        move_forward_velocity = p.readUserDebugParameter(self.move_forward_gui_id)
        angle_velocity = p.readUserDebugParameter(self.turn_gui_id)

        if left_wheel_velocity != 0.0 or right_wheel_velocity != 0.0:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=[self.left_wheel_id, self.right_wheel_id],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[left_wheel_velocity, right_wheel_velocity],
                physicsClientId=self.client_id
            )
            p.stepSimulation()

        if move_forward_velocity != 0.0:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=[self.left_wheel_id, self.right_wheel_id],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[move_forward_velocity, move_forward_velocity],
                physicsClientId=self.client_id
            )
            p.stepSimulation()

        if angle_velocity != 0.0:
            #  differential drive
            # 通过两个轮子不同的速度来实现转向
            # radian
            velocity = angle_velocity * self.wheel_dist / step_duration
            p.setJointMotorControl2(
                self.robot_id, self.right_wheel_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=velocity,
                physicsClientId=self.client_id
            )
            p.stepSimulation()
        location, _ = p.getBasePositionAndOrientation(self.robot_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=110,
            cameraPitch=-50,
            cameraTargetPosition=location
        )

        if step_num == total_step_num:
            print()


def addDebugGUI(self):
    self.left_wheel_gui_id = p.addUserDebugParameter("Left wheel velocity", -1., 1., 0)
    self.right_wheel_gui_id = p.addUserDebugParameter("Right wheel velocity", -1., 1., 0)
    self.move_forward_gui_id = p.addUserDebugParameter("Move forward", -1., 1., 0)
    self.turn_gui_id = p.addUserDebugParameter("Turn (radian)", -np.pi / 20, np.pi / 20, 0)