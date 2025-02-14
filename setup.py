from setuptools import setup, find_packages

setup(
    name="lunabot",
    version="1.1.0",
    packages=find_packages(where="src"),
    include_package_data=True,
    package_dir={"": "src"},
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "matplotlib",
        "scipy",
        "noise",
        "Pillow",
        "torch"
    ],
    description="Lunar Geosearch RL environment",
    author="Your Name",
    license="MIT",
)