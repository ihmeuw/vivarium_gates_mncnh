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
        "vivarium_dependencies<2.0.0",
        "vivarium_dependencies[pandas,numpy,scipy,click,tables,loguru]",
        "vivarium_build_utils>=4.0.0,<5.0.0",
        "gbd_mapping>=5.0.0,<6.0.0",
        "layered_config_tree<5.0.0",
        # Temporarily install engine + public-health from the unreleased,
        # stacked age-sims branch of the vivarium-suite monorepo (engine 5.4.0
        # + public-health 6.4.0 priority work). Revert to version pins once
        # those releases land on PyPI.
        "vivarium_engine @ git+https://github.com/ihmeuw/vivarium-suite.git@albrja/mic-6867/age-sims#subdirectory=libs/engine",
        "vivarium_public_health @ git+https://github.com/ihmeuw/vivarium-suite.git@albrja/mic-6867/age-sims#subdirectory=libs/public-health",
        "click",
        "jinja2",
        "pyyaml",
        "statsmodels",
    ]

    setup_requires = ["setuptools_scm"]

    data_requirements = ["vivarium_inputs>=8.0.0,<9.0.0"]
    cluster_requirements = [
        # [cluster] pulls jobmon_installer_ihme (→ jobmon), which psimulate
        # needs at import time; a plain vivarium_cluster_tools install omits it.
        "vivarium_cluster_tools[cluster]>=4.0.0,<5.0.0",
        "drmaa",
    ]
    test_requirements = [
        "vivarium_dependencies[pytest]",
        "papermill",
        "jupyterlab",
        "vivarium_testing_utils>=0.7.0,<0.8.0",
    ]
    validation_requirements = ["vivarium_testing_utils[validation]>=0.7.0,<0.8.0"]
    lint_requirements = [
        "vivarium_dependencies[lint]",
    ]
    interactive_requirements = [
        "vivarium_dependencies[interactive]",
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
            + test_requirements,
            # TODO: fold validation_requirements back into "data" once a
            # vivarium_testing_utils release supports vivarium_inputs>=8.0.0
            # (0.7.1's [validation] extra caps vivarium_inputs<8.0.0).
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
