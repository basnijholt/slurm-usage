from setuptools import setup

setup(
    name="slurm-usage",
    version="1.1.0",
    description="Command to list the current cluster usage per user.",
    url="https://github.com/basnijholt/slurm-usage",
    author="Bas Nijholt",
    license="MIT",
    install_requires=["rich"],
    py_modules=["slurm_usage"],
    entry_points={"console_scripts": ["slurm-usage=slurm_usage:main", "stats=slurm_usage:main"]},
)
