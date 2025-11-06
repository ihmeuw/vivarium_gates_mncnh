# Check if we're running in Jenkins
ifdef JENKINS_URL
# 	Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
# 	For local dev, use the installed vivarium_build_utils package if it exists
# 	First, check if we can import vivarium_build_utils and assign 'yes' or 'no'.
# 	We do this by importing the package in python and redirecting stderr to the null device.
# 	If the import is successful (&&), it will print 'yes', otherwise (||) it will print 'no'.
	VIVARIUM_BUILD_UTILS_AVAILABLE := $(shell python -c "import vivarium_build_utils" 2>/dev/null && echo "yes" || echo "no")
# 	If vivarium_build_utils is available, get the makefiles path or else set it to empty
	ifeq ($(VIVARIUM_BUILD_UTILS_AVAILABLE),yes)
		MAKE_INCLUDES := $(shell python -c "from vivarium_build_utils.resources import get_makefiles_path; print(get_makefiles_path())")
	else
		MAKE_INCLUDES :=
	endif
endif

# Set the package name as the last part of this file's parent directory path
PACKAGE_NAME = $(notdir $(CURDIR))

ifneq ($(MAKE_INCLUDES),) # not empty
# Include makefiles from vivarium_build_utils
include $(MAKE_INCLUDES)/base.mk
include $(MAKE_INCLUDES)/test.mk
else # empty
# Use this help message (since the vivarium_build_utils version is not available)
help:
	@echo
	@echo "For Make's standard help, run 'make --help'."
	@echo
	@echo "Most of our Makefile targets are provided by the vivarium_build_utils"
	@echo "package. To access them, you need to create a development environment first."
	@echo
	@echo "make build-env"
	@echo
	@echo "USAGE:"
	@echo "  make build-env name=<environment_name> [py=<python_version>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  type [optional]"
	@echo "      Type of conda environment. Either 'simulation' (default) or 'artifact'"
	@echo "  name [optional]"
	@echo "      Name of the conda environment to create (defaults to package name)"
	@echo "  include_timestamp [optional]"
	@echo "      If 'yes', appends a timestamp to the environment name. Defaults to 'no'"
	@echo "  py [optional]"
	@echo "      Python version (defaults to latest supported)"
	@echo
	@echo "EXAMPLE:"
	@echo "  make build-env name=vivarium_dev"
	@echo "  make build-env name=vivarium_dev py=3.9"
	@echo
	@echo "After creating the environment:"
	@echo "  1. Activate it: 'conda activate <environment_name>'"
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo
endif

build-env: # Create a new environment with installed packages
#   Handle arguments and set defaults
	@$(eval type ?= simulation)
	@$(if $(filter-out simulation artifact,$(type)),$(error Error: 'type' must be either 'simulation' or 'artifact', got '$(type)'))
	@$(eval include_timestamp ?= no)
	@$(if $(filter-out yes no,$(include_timestamp)),$(error Error: 'include_timestamp' must be either 'yes' or 'no', got '$(include_timestamp)'))
#	Set name with type suffix if not provided, otherwise use provided name as-is
	@$(eval name ?= $(PACKAGE_NAME)_$(type))
#	Append timestamp if requested (only if name wasn't explicitly provided)
	@$(if $(filter yes,$(include_timestamp)),$(eval override name := $(name)_$(shell date +%Y%m%d_%H%M%S)),)
#	Check if py is set, otherwise use the latest supported version
	@$(eval py ?= $(shell python -c "import json; versions = json.load(open('python_versions.json')); print(max(versions, key=lambda x: tuple(map(int, x.split('.')))))"))
	
	conda create -n $(name) python=$(py) --yes
# 	Bootstrap vivarium_build_utils into the new environment
	conda run -n $(name) pip install vivarium_build_utils
#	Install packages based on type
	@if [ "$(type)" = "simulation" ]; then \
		conda run -n $(name) make install ENV_REQS=dev; \
		conda install -n $(name) redis -c anaconda -y; \
	elif [ "$(type)" = "artifact" ]; then \
		conda run -n $(name) make install ENV_REQS=data; \
	fi

	@echo
	@echo "Finished building environment"
	@echo "  name: $(name)"
	@echo "  type: $(type)"
	@echo "  python version: $(py)"
	@echo
	@echo "Don't forget to activate it with: 'conda activate $(name)'"
	@echo
