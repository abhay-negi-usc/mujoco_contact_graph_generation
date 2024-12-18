from setuptools import setup, find_packages

setup(
    name="mujoco_contact_graph",
    version="0.1.0",
    description="Contact graph generation and ICP Localization",
    author="Abhay N.",
    packages=find_packages(
        include=[
            "classification_model",
            "classification_model.*",
            "vector_prediction_model",
            "vector_prediction_model.*",
            "boundary_mapping",
            "boundary_mapping.*",
        ]
    ),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "mujoco",
        "matplotlib",
        "scipy",
    ],
    python_requires=">=3.8",
)
