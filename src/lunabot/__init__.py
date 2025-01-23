from .geosearch import GeosearchEnv, Utils
import importlib.resources

# Helper to get the path to robot.png
def get_robot_image_path():
    return importlib.resources.files("lunabot").joinpath("robot.png")