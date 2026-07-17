#!/usr/bin/env python
import json
import os
import sys

from packaging.version import parse
from setuptools import find_packages, setup

with open("python_versions.json", "r") as f:
    supported_python_versions = json.load(f)

python_versions = [parse(v) for v in supported_python_versions]
min_version = min(python_versions)
max_version = max(python_versions)
if not (
    min_version <= parse(".".join([str(v) for v in sys.version_info[:2]])) <= max_version
):
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    # Python 3.5 does not support f-strings
    error = (
        "\n----------------------------------------\n"
        "Error: This repo requires python {min_version}-{max_version}.\n"
        "You are running python {py_version}".format(
            min_version=min_version.base_version,
            max_version=max_version.base_version,
            py_version=py_version,
        )
    )
    print(error, file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_gates_mncnh", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        "vivarium-dependencies<2.0.0",
        "vivarium-dependencies[pandas,numpy,scipy,click,tables,loguru]",
        "vivarium-build-utils>=4.0.0,<5.0.0",
        "vivarium-gbd-mapping>=6.0.0,<7.0.0",
        "vivarium-config-tree",
        "vivarium-engine>=5.3.0, <6.0.0",
        "vivarium-public-health>=6.3.1, <7.0.0",
        "click",
        "jinja2",
        "pyyaml",
        "statsmodels",
    ]

    setup_requires = ["setuptools_scm"]

    data_requirements = ["vivarium-inputs>=8.0.0,<9.0.0"]
    cluster_requirements = [
        "vivarium-cluster-tools[cluster]>=4.0.0,<5.0.0",
    ]
    test_requirements = [
        "vivarium-dependencies[pytest]",
        "papermill",
        "jupyterlab",
        "vivarium-testing-utils>=0.7.4",
    ]
    validation_requirements = ["vivarium-testing-utils[validation]"]
    lint_requirements = [
        "vivarium-dependencies[lint]",
    ]
    interactive_requirements = [
        "vivarium-dependencies[interactive]",
        "nbdime",
    ]

    setup(
        # name is declared statically in pyproject.toml's [project] block.
        # Setuptools errors if it's also passed here.
        description=about["__summary__"],
        long_description=long_description,
        license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        extras_require={
            "test": test_requirements,
            "cluster": cluster_requirements,
            "data": data_requirements
            + cluster_requirements
            + lint_requirements
            + test_requirements
            + validation_requirements,
            "validation": validation_requirements,
            "interactive": interactive_requirements,
            "dev": test_requirements
            + cluster_requirements
            + lint_requirements
            + interactive_requirements,
        },
        zip_safe=False,
        use_scm_version={
            "write_to": "src/vivarium_gates_mncnh/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
        entry_points={
            "console_scripts": [
                "make_artifacts=vivarium_gates_mncnh.tools.cli:make_artifacts",
            ],
        },
    )
