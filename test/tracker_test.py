from PythonLinearNonlinearControl.configs import TwoWheeledConfigModule
from PythonLinearNonlinearControl.controllers import iLQR
from PythonLinearNonlinearControl.models import TwoWheeledModel
from PythonLinearNonlinearControl.planners import ConstantPlanner

config = TwoWheeledConfigModule()
planner = ConstantPlanner(config)
model = TwoWheeledModel(config)
controller = iLQR(config, model)

