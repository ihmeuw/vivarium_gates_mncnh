# Check if we're running in Jenkins
ifdef JENKINS_URL
# 	Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
# 	For local dev, use the vivarium_build_utils installed in this checkout's .venv.
# 	Call its python by absolute path: make does not pass exported variables to
# 	$(shell), and we never want whatever python happens to be on PATH.
	VENV_DIR := $(CURDIR)/.venv
	VENV_PYTHON := $(VENV_DIR)/bin/python
	ifneq ($(wildcard $(VENV_PYTHON)),)
		MAKE_INCLUDES := $(shell $(VENV_PYTHON) -c "from vivarium.build_utils.resources import get_makefiles_path; print(get_makefiles_path())" 2>/dev/null)
	else
		MAKE_INCLUDES :=
	endif
# 	Run recipes in .venv without needing it activated.
	export VIRTUAL_ENV := $(VENV_DIR)
	export PATH := $(VENV_DIR)/bin:$(PATH)
endif

# Set the package name as the last part of this file's parent directory path
PACKAGE_NAME = $(notdir $(CURDIR))

# Helper function for validating enum arguments
validate_arg = $(if $(filter-out $(2),$(1)),$(error Error: '$(3)' must be one of: $(2), got '$(1)'))

# Macro for validating make target arguments
# Usage: $(call validate_make_args,target_name,allowed_args)
# Example: $(call validate_make_args,build-env,force)
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

# Fail with install instructions if uv is not on PATH.
define require_uv
	@command -v uv >/dev/null 2>&1 || { \
		echo "Error: uv is not on PATH. Install it with:" >&2; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; \
		exit 1; \
	}
endef

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
	@echo "package, which is installed in this checkout's .venv. To get them, build"
	@echo ".venv first. You need uv on your PATH."
	@echo
	@echo "Most people should run 'source environment.sh' instead, which also"
	@echo "activates .venv. The targets below work before .venv exists."
	@echo
	@echo "================================================================================"
	@echo "build-env: Build .venv from uv.lock"
	@echo "================================================================================"
	@echo
	@echo "USAGE:"
	@echo "  make build-env [force=<yes|no>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  force [optional]"
	@echo "      Whether to reinstall every package. Either 'yes' or 'no' (default)"
	@echo
	@echo "================================================================================"
	@echo "lock-versions: Update uv.lock after editing pyproject.toml"
	@echo "================================================================================"
	@echo
	@echo "Keeps every locked version that still fits pyproject.toml. Commit uv.lock after."
	@echo
	@echo "================================================================================"
	@echo "upgrade-versions: Move locked versions to the newest that fit pyproject.toml"
	@echo "================================================================================"
	@echo
	@echo "USAGE:"
	@echo "  make upgrade-versions [package=<name>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  package [optional]"
	@echo "      Upgrade only this package. By default every package is upgraded."
	@echo
	@echo "After building .venv, run 'make help' again to see all available targets."
	@echo
endif

build-env: # Build .venv from uv.lock
#	Validate arguments - exit if unsupported arguments are passed
	$(call validate_make_args,build-env,force)
	@$(eval force ?= no)
	@$(call validate_arg,$(force),yes no,force)
	$(call require_uv)
#	uv will not reuse a .venv that holds the old shared-env overlays
	@if [ -d .venv ] && [ ! -f .venv/pyvenv.cfg ]; then \
		echo "Error: .venv is not a venv. It holds the old shared-environment overlays:" >&2; \
		ls -d .venv/*/ >&2; \
		echo "Remove it, then run this again:" >&2; \
		echo "  rm -rf $(CURDIR)/.venv" >&2; \
		exit 1; \
	fi
	uv sync --locked $(if $(filter yes,$(force)),--reinstall,)

lock-versions: # Update uv.lock after editing pyproject.toml
	$(call validate_make_args,lock-versions,)
	$(call require_uv)
	uv lock

upgrade-versions: # Upgrade locked versions (all, or package=<name>)
	$(call validate_make_args,upgrade-versions,package)
	$(call require_uv)
	uv lock $(if $(package),--upgrade-package $(package),--upgrade)
