# Check if we're running in Jenkins
ifdef JENKINS_URL
# 	Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
# 	For local dev, use the installed vivarium.build_utils package if it exists
# 	First, check if we can import vivarium.build_utils and assign 'yes' or 'no'.
# 	We do this by importing the package in python and redirecting stderr to the null device.
# 	If the import is successful (&&), it will print 'yes', otherwise (||) it will print 'no'.
	VIVARIUM_BUILD_UTILS_AVAILABLE := $(shell python -c "import vivarium.build_utils" 2>/dev/null && echo "yes" || echo "no")
# 	If vivarium.build_utils is available, get the makefiles path or else set it to empty
	ifeq ($(VIVARIUM_BUILD_UTILS_AVAILABLE),yes)
		MAKE_INCLUDES := $(shell python -c "from vivarium.build_utils.resources import get_makefiles_path; print(get_makefiles_path())")
	else
		MAKE_INCLUDES :=
	endif
endif

# Set the package name as the last part of this file's parent directory path
PACKAGE_NAME = $(notdir $(CURDIR))

# The package's name, read from pyproject.toml so there is only one copy of it.
# awk rather than Python, because environment.sh needs this before any
# environment exists.
DIST_NAME := $(shell awk -F'"' '/^\[project\]/{p=1} p && /^name[[:space:]]*=/{print $$2; exit}' $(CURDIR)/pyproject.toml)
$(if $(DIST_NAME),,$(error Could not read the [project] name from pyproject.toml))

# Helper function for validating enum arguments
validate_arg = $(if $(filter-out $(2),$(1)),$(error Error: '$(3)' must be one of: $(2), got '$(1)'))

# Macro for validating make target arguments
# Usage: $(call validate_make_args,target_name,allowed_args)
# Example: $(call validate_make_args,build-env,type name path)
define validate_make_args
	@allowed="$(2)"; \
	for arg in $(filter-out $(1),$(MAKECMDGOALS)) $(MAKEFLAGS); do \
		case $$arg in \
			*=*) \
				arg_name=$${arg%%=*}; \
				if ! echo " $$allowed " | grep -q " $$arg_name "; then \
					allowed_list=$$(echo $$allowed | sed 's/ /, /g'); \
					echo "Error: Invalid argument '$$arg_name'. Allowed arguments are: $$allowed_list" >&2; \
					exit 1; \
				fi \
				;; \
		esac; \
	done
endef

# Newest supported Python. The version files are resolved for it.
NEWEST_PYTHON := $(shell cat $(CURDIR)/python_versions.json | tr -d '[]" ' | tr ',' '\n' | sort -t. -k1,1n -k2,2n | tail -1)

# A comma that can go inside $(if ...) arguments.
comma := ,

# Environment types and the extra each installs. Keep in sync with ENV_TYPES in check_env_versions.py.
ENV_TYPES := simulation artifact
ENV_REQS_simulation := dev
ENV_REQS_artifact := data

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
	@echo "================================================================================"
	@echo "build-env: Create a full conda environment from scratch"
	@echo "================================================================================"
	@echo
	@echo "This target creates a new conda environment and installs all required"
	@echo "packages for development or artifact generation, depending on the 'type' argument."
	@echo "It is recommended to use this target only if you cannot use the 'build-shared-env' target,"
	@echo "either because you are not on the cluster or because you need to customize the environment,"
	@echo "particularly if you need non-python packages installed via conda."
	@echo "Packages are installed at the versions pinned in requirements/<type>.txt."
	@echo
	@echo "USAGE:"
	@echo "  make build-env [type=<environment type>] [name=<environment name>] [path=<environment path>] [py=<python version>] [include_timestamp=<yes|no>] [lfs=<yes|no>] [force=<yes|no>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  type [optional]"
	@echo "      Type of conda environment. Either 'simulation' (default) or 'artifact'"
	@echo "  name [optional]"
	@echo "      Name of the conda environment to create (defaults to <PACKAGE_NAME>_<TYPE>)"
	@echo "  path [optional]"
	@echo "      Absolute path where the environment should be created (overrides name for location)"
	@echo "  include_timestamp [optional]"
	@echo "      Whether to append a timestamp to the environment name. Either 'yes' or 'no' (default)"
	@echo "  lfs [optional]"
	@echo "      Whether to install git-lfs in the environment. Either 'yes' or 'no' (default)"
	@echo "  py [optional]"
	@echo "      Python version (defaults to, and must match, the latest supported: $(NEWEST_PYTHON))"
	@echo "  force [optional]"
	@echo "      Whether to remove and recreate an existing environment. Either 'yes' or 'no' (default)"
	@echo
	@echo "After creating the environment:"
	@echo "  1. Activate it: 'conda activate <environment_name>'"
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo
	@echo "================================================================================"
	@echo "build-shared-env: Create a lightweight venv on top of a shared conda environment"
	@echo "================================================================================"
	@echo
	@echo "This is the RECOMMENDED approach for development on the cluster. It creates a virtual"
	@echo "environment that inherits packages from a Jenkins-built shared conda environment,"
	@echo "while allowing you to install the local package in editable mode."
	@echo
	@echo "USAGE:"
	@echo "  make build-shared-env [type=<environment type>] [venv_dir=<directory>] [venv_name=<name>] [shared_env_dir=<path>] [force=<yes|no>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  type [optional]"
	@echo "      Type of shared environment to use. Either 'simulation' (default) or 'artifact'"
	@echo "  venv_dir [optional]"
	@echo "      Directory where venvs are stored (defaults to '.venv')"
	@echo "  venv_name [optional]"
	@echo "      Name of the venv to create (defaults to '<PACKAGE_NAME>_<TYPE>')"
	@echo "  shared_env_dir [optional]"
	@echo "      Base directory for shared environments (defaults to Jenkins shared env location)"
	@echo "  force [optional]"
	@echo "      Whether to remove and recreate an existing venv. Either 'yes' or 'no' (default)"
	@echo
	@echo "After creating the environment:"
	@echo "  1. Activate it: 'source <venv_dir>/<environment_name>/bin/activate'"
	@echo "  2. Run 'make help' again to see all available targets"
	@echo
endif

build-env: # Create a new environment with installed packages
#	Validate arguments - exit if unsupported arguments are passed
	$(call validate_make_args,build-env,type name path lfs py include_timestamp force)
	
#   Handle arguments and set defaults
#   type
	@$(eval type ?= simulation)
	@$(call validate_arg,$(type),$(ENV_TYPES),type)
#	name
	@$(eval name ?= $(PACKAGE_NAME)_$(type))
#	timestamp
	@$(eval include_timestamp ?= no)
	@$(call validate_arg,$(include_timestamp),yes no,include_timestamp)
	@$(if $(filter yes,$(include_timestamp)),$(eval override name := $(name)_$(shell date +%Y%m%d_%H%M%S)),)
#	path (optional - if set, use -p for conda create instead of -n)
	@$(eval path ?=)
#	lfs
	@$(eval lfs ?= no)
	@$(call validate_arg,$(lfs),yes no,lfs)
#	force
	@$(eval force ?= no)
	@$(call validate_arg,$(force),yes no,force)
#	python version
	@$(eval py ?= $(NEWEST_PYTHON))
#	The version files only work for NEWEST_PYTHON.
	@$(if $(filter-out $(NEWEST_PYTHON),$(py)),$(error Error: py=$(py) is not supported; omit py (the version files are resolved for Python $(NEWEST_PYTHON)$(comma) the newest in python_versions.json)))
#	Determine conda create flag: -p for path, -n for name
	@$(eval CONDA_CREATE_FLAG := $(if $(path),-p $(path),-n $(name)))
#	Determine conda run flag: -p for path, -n for name
	@$(eval CONDA_RUN_FLAG := $(if $(path),-p $(path),-n $(name)))

#	Check if environment already exists and handle based on force flag
	@if conda env list | grep -qE "$(if $(path),^$(path),^$(name))\s"; then \
		if [ "$(force)" = "yes" ]; then \
			echo "Removing existing environment..."; \
			conda remove $(CONDA_CREATE_FLAG) --all --yes; \
		else \
			echo "Error: Environment already exists at $(if $(path),$(path),$(name))" >&2; \
			echo "Use 'force=yes' to remove and recreate it, or specify a different location with 'name=<name>' or 'path=<path>'" >&2; \
			exit 1; \
		fi \
	fi

	conda create $(CONDA_CREATE_FLAG) python=$(py) --yes
#	Install the pinned uv and vivarium_build_utils first, so base.mk doesn't install unpinned ones.
	@versions_file="$(VERSIONS_DIR)/$(type).txt"; \
	if [ ! -f "$$versions_file" ]; then \
		echo "Error: version file $$versions_file not found" >&2; \
		exit 1; \
	fi; \
	uv_version=$$($(call pinned_version,$$versions_file,uv)); \
	vbu_version=$$($(call pinned_version,$$versions_file,vivarium-build-utils)); \
	if [ -z "$$uv_version" ] || [ -z "$$vbu_version" ]; then \
		echo "Error: $$versions_file must pin both uv and vivarium_build_utils with '=='" >&2; \
		echo "  (found uv: '$$uv_version', vivarium_build_utils: '$$vbu_version')" >&2; \
		exit 1; \
	fi; \
	echo "conda run $(CONDA_RUN_FLAG) pip install \"uv==$$uv_version\" \"vivarium_build_utils==$$vbu_version\""; \
	conda run $(CONDA_RUN_FLAG) pip install "uv==$$uv_version" "vivarium_build_utils==$$vbu_version"
#	Install the packages (pinned by the install target below). set -e so a failure stops here.
	@set -e; \
	conda run $(CONDA_RUN_FLAG) make install ENV_REQS=$(ENV_REQS_$(type)); \
	if [ "$(type)" = "simulation" ]; then \
		conda install $(CONDA_RUN_FLAG) redis -c anaconda -y; \
	fi
	@set -e; \
	if [ "$(lfs)" = "yes" ]; then \
		conda run $(CONDA_RUN_FLAG) conda install -c conda-forge git-lfs --yes; \
		conda run $(CONDA_RUN_FLAG) git lfs install; \
	fi
#	Stop if the installed versions don't match the version file.
	conda run $(CONDA_RUN_FLAG) python $(CHECK_ENV_VERSIONS) installed-matches-version-file --repo $(CURDIR) --type $(type)
#	Save a copy of the version files into the environment, so activation can tell if it's
#	out of date. A failed build never gets here.
	conda run $(CONDA_RUN_FLAG) python $(CHECK_ENV_VERSIONS) record --repo $(CURDIR) --type $(type)

	@echo
	@echo "Finished building environment"
	@$(if $(path),echo "  path: $(path)",echo "  name: $(name)")
	@echo "  type: $(type)"
	@echo "  git-lfs installed: $(lfs)"
	@echo "  python version: $(py)"
	@echo "  forced rebuild: $(force)"
	@echo
	@echo "After creating the environment:"
	@$(if $(path),echo "  1. Activate it: 'conda activate $(path)'",echo "  1. Activate it: 'conda activate $(name)'")
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo

# Default shared environment directory (set by Jenkins nightly builds)
SHARED_ENV_DIR ?= /mnt/team/simulation_science/priv/engineering/jenkins/shared_envs

build-shared-env: # Create a lightweight venv overlay on top of a shared conda environment
#	Validate arguments - exit if unsupported arguments are passed
	$(call validate_make_args,build-shared-env,type venv_dir venv_name shared_env_dir force)

#	Handle arguments and set defaults
#	type
	@$(eval type ?= simulation)
	@$(call validate_arg,$(type),simulation artifact,type)
#	venv_dir
	@$(eval venv_dir ?= .venv)
#	venv_name
	@$(eval venv_name ?= $(DIST_NAME)_$(type))
#	Construct full venv path
	@$(eval venv_path := $(venv_dir)/$(venv_name))
#	shared_env_dir
	@$(eval shared_env_dir ?= $(SHARED_ENV_DIR))
#	force
	@$(eval force ?= no)
	@$(call validate_arg,$(force),yes no,force)
#	Construct shared environment path
	@$(eval SHARED_ENV_NAME := $(DIST_NAME)_$(type)_current)
	@$(eval SHARED_ENV_PATH := $(shared_env_dir)/$(SHARED_ENV_NAME))

#	Verify shared environment exists
	@if [ ! -d "$(SHARED_ENV_PATH)" ]; then \
		echo "Error: Shared environment not found at $(SHARED_ENV_PATH)" >&2; \
		echo "Make sure the Jenkins nightly build has run successfully." >&2; \
		exit 1; \
	fi

#	Handle existing venv
	@if [ -d "$(venv_path)" ]; then \
		if [ "$(force)" = "yes" ]; then \
			echo "Clearing existing venv at $(venv_path)"; \
			rm -rf "$(venv_path)"; \
		else \
			echo "Warning: venv already exists at $(venv_path)" >&2; \
			echo "Use 'force=yes' to remove and recreate it, or specify a different location with 'venv_dir=<dir>' and 'venv_name=<name>'" >&2; \
			exit 1; \
		fi \
	fi

#	Create venv overlay with system-site-packages
	@echo "Creating venv overlay at $(venv_path)"
	@echo "  Base environment: $(SHARED_ENV_PATH)"
	$(SHARED_ENV_PATH)/bin/python -m venv --system-site-packages $(venv_path)

#	Patch activate scripts to include shared environment's bin/ in PATH
#	This ensures CLI entry points (e.g. psimulate) from the shared env are available.
#	We append to the activate script rather than sed-replacing the PATH line because
#	the exact format of that line varies across Python versions.
#	_OLD_VIRTUAL_PATH is set by the activate script before any PATH modification,
#	so we can reconstruct PATH with the correct precedence:
#	  venv bin > shared env bin > original PATH
	@echo "Patching activate scripts to inherit shared environment CLI entry points"
	@echo '' >> $(venv_path)/bin/activate
	@echo '# Include shared environment CLI entry points (e.g. psimulate)' >> $(venv_path)/bin/activate
	@echo 'PATH="$$VIRTUAL_ENV/bin:$(SHARED_ENV_PATH)/bin:$${_OLD_VIRTUAL_PATH}"' >> $(venv_path)/bin/activate
	@echo 'export PATH' >> $(venv_path)/bin/activate
	@if [ -f "$(venv_path)/bin/activate.fish" ]; then \
		echo '' >> $(venv_path)/bin/activate.fish; \
		echo '# Include shared environment CLI entry points (e.g. psimulate)' >> $(venv_path)/bin/activate.fish; \
		echo 'set -gx PATH "$$VIRTUAL_ENV/bin" "$(SHARED_ENV_PATH)/bin" $$_OLD_VIRTUAL_PATH' >> $(venv_path)/bin/activate.fish; \
	fi
	@if [ -f "$(venv_path)/bin/activate.csh" ]; then \
		echo '' >> $(venv_path)/bin/activate.csh; \
		echo '# Include shared environment CLI entry points (e.g. psimulate)' >> $(venv_path)/bin/activate.csh; \
		echo 'setenv PATH "$$VIRTUAL_ENV/bin:$(SHARED_ENV_PATH)/bin:$$_OLD_VIRTUAL_PATH"' >> $(venv_path)/bin/activate.csh; \
	fi

#	Install local package in editable mode (no-deps since shared env has dependencies)
	@echo "Installing local package in editable mode (--no-deps)"
	$(venv_path)/bin/pip install -e . --no-deps

	@echo
	@echo "Finished creating venv"
	@echo "  venv directory: $(venv_dir)"
	@echo "  venv name: $(venv_name)"
	@echo "  full path: $(venv_path)"
	@echo "  base environment: $(SHARED_ENV_PATH)"
	@echo "  type: $(type)"
	@echo
	@echo "After creating the environment:"
	@echo "  1. Activate it: 'source $(venv_path)/bin/activate'"
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo

print-dist-name: # Print the distribution name (used by environment.sh)
	@echo $(DIST_NAME)

# ------------------------------------------------------------------------------
# Pinned package versions
#
# requirements/<type>.txt pins exactly what each environment runs. The optional
# requirements/overrides.txt installs framework packages from git (never on main).
# ------------------------------------------------------------------------------
VERSIONS_DIR := $(CURDIR)/requirements
CHECK_ENV_VERSIONS := $(CURDIR)/src/vivarium_gates_mncnh/tools/check_env_versions.py

# Print the version a file pins for a package: $(call pinned_version,<file>,<name>)
pinned_version = awk -F'==' -v want='$(2)' '{ n = tolower($$1); gsub(/[ \t\r]/, "", n); gsub(/[-_.]+/, "-", n); if (n == want) { v = $$2; sub(/^[ \t]+/, "", v); sub(/[^0-9A-Za-z.+!].*/, "", v); print v; exit } }' "$(1)"

# Pin base.mk's `install` (used by build-env, the shared env build and Jenkins) to the
# version files. ENV_REQS=data means artifact; anything else means simulation.
install_env_type = $(if $(filter $(ENV_REQS_artifact),$(strip $(ENV_REQS))),artifact,simulation)

ifeq ($(wildcard $(VERSIONS_DIR)/overrides.txt),)
install: export UV_CONSTRAINT = $(VERSIONS_DIR)/$(install_env_type).txt
else
# With overrides, drop the overridden packages' pins, or uv can't install their git versions.
UV_CONSTRAINTS_FILE := $(CURDIR)/build/uv-constraints.txt
install: export UV_CONSTRAINT = $(UV_CONSTRAINTS_FILE)
install: export UV_OVERRIDE = $(VERSIONS_DIR)/overrides.txt
install: uv-constraints

# Write the version file minus the overridden packages. Runs before every install, with the
# active environment's python.
.PHONY: uv-constraints
uv-constraints:
	python $(CHECK_ENV_VERSIONS) write-install-constraints --repo $(CURDIR) --type $(install_env_type) --out $(UV_CONSTRAINTS_FILE)
endif

# Minimums used only when regenerating the version files.
RESOLVER_CONSTRAINTS := $(VERSIONS_DIR)/resolver-constraints.txt

# Shared by lock-versions and upgrade-versions: $(call compile_versions,<target>,<extra uv flags>)
# Resolves each type into a temp dir and copies the results in only if all succeed.
# uv, pip and setuptools are pinned too. Then prints the pins that moved.
# Run from an activated environment (needs uv and python).
define compile_versions
	$(call validate_make_args,$(1),type)
	@$(eval type ?= all)
	@$(call validate_arg,$(type),$(ENV_TYPES) all,type)
	@set -e; \
	tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	mkdir "$$tmp_dir/old" "$$tmp_dir/new"; \
	types=""; \
	for pair in $(foreach t,$(if $(filter all,$(type)),$(ENV_TYPES),$(type)),$(t):$(ENV_REQS_$(t))); do \
		t=$${pair%%:*}; \
		extra=$${pair#*:}; \
		types="$$types $$t"; \
		if [ -f "$(VERSIONS_DIR)/$$t.txt" ]; then \
			cp "$(VERSIONS_DIR)/$$t.txt" "$$tmp_dir/old/$$t.txt"; \
			cp "$(VERSIONS_DIR)/$$t.txt" "$$tmp_dir/new/$$t.txt"; \
		fi; \
		echo "Resolving $$t.txt (extra '$$extra', Python $(NEWEST_PYTHON))"; \
		printf 'uv\npip\nsetuptools\n' | uv pip compile $(CURDIR)/pyproject.toml - \
			--extra $$extra \
			--python-version $(NEWEST_PYTHON) \
			$(if $(wildcard $(RESOLVER_CONSTRAINTS)),-c $(RESOLVER_CONSTRAINTS)) \
			$(EXTRA_INDEX_FLAGS) \
			--no-emit-package vivarium-gates-mncnh \
			--custom-compile-command "make $(1) type=$$t" \
			-o "$$tmp_dir/new/$$t.txt" \
			$(2); \
	done; \
	for t in $$types; do \
		cp "$$tmp_dir/new/$$t.txt" "$(VERSIONS_DIR)/$$t.txt"; \
	done; \
	for t in $$types; do \
		if [ ! -f "$$tmp_dir/old/$$t.txt" ]; then \
			echo "Created $$t.txt."; \
			continue; \
		fi; \
		moved=$$(python $(CHECK_ENV_VERSIONS) show-changes "$$tmp_dir/old/$$t.txt" "$$tmp_dir/new/$$t.txt"); \
		if [ -n "$$moved" ]; then \
			echo "Pins that moved in $$t.txt:"; \
			echo "$$moved" | sed 's/^/    /'; \
		else \
			echo "No existing pins moved in $$t.txt."; \
		fi; \
	done
endef

lock-versions: # Resolve packages not yet pinned; prefers existing pins and reports any it had to move
#	type=simulation|artifact|all (default all). Use after changing a dependency.
	$(call compile_versions,lock-versions,)

upgrade-versions: # Re-resolve every package to the newest version pyproject.toml allows
#	Like lock-versions but ignores existing pins, so every package can move.
	$(call compile_versions,upgrade-versions,--upgrade)
