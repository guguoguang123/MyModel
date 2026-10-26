"""Editable-install compatibility for pip versions without PEP 660 support."""

from setuptools import find_packages, setup

setup(
    name="triwave-gnn",
    version="0.4.0",
    description="Pin-level circuit representation learning",
    packages=find_packages(),
    include_package_data=True,
    package_data={"triwave_gnn.pretrained": ["*.pth", "README.md"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "torch>=2.0", "torch-geometric>=2.8,<3"],
)
