#!/bin/bash

set -e # exit on error

# Define variables
username=$(whoami)
env_type="simulation"
make_new="no"
use_shared="no"
install_git_lfs="no"
days_until_stale=7 # Number of days until environment is considered stale

# Initialize conda if not already initialized
conda_path=$($SHELL -ic 'conda info --base')
if [ ! -d "$conda_path" ]; then
  echo
  echo "ERROR: Conda path $conda_path does not exist"
  exit 1
fi
if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
  echo
  echo "Initializing conda from $conda_path"
  source "$conda_path/etc/profile.d/conda.sh"
else
  echo
  echo "ERROR: Unable to find conda in expected locations"
  exit 1
fi

# Reset OPTIND so help can be invoked multiple times per shell session.
OPTIND=1
Help()
{ 
   # Display Help
   echo
   echo "Script to automatically create and validate conda environments."
   echo
   echo "Syntax: source environment.sh [-h|t|v|s|f|l]"
   echo "options:"
   echo "h     Print this Help."
   echo "t     Type of conda environment. Either 'simulation' (default) or 'artifact'."
   echo "s     Use shared environment (venv overlay). Recommended for cluster development."
   echo "f     Force creation of a new environment."
   echo "l     Install git lfs."
}

# Process input options
while getopts ":hsflt:" option; do
   case $option in
      h) # display help
         Help
         exit 0;;
      t) # Type of conda environment to build
         env_type=$OPTARG;;
      s) # Use shared environment
         use_shared="yes";;
      f) # Force creation of a new environment
         make_new="yes";;
      l) # Install git lfs
         install_git_lfs="yes";;
     \?) # Invalid option
         echo
         echo "ERROR: Invalid option"
         exit 1;;
   esac
done

# Parse environment name
env_name=$(basename "`pwd`")
env_name+="_$env_type"
branch_name=$(git rev-parse --abbrev-ref HEAD)
# Determine which requirements.txt to install from
if [ $env_type == 'simulation' ]; then
  install_file="requirements.txt"
elif [ $env_type == 'artifact' ]; then
  install_file="artifact_requirements.txt"
else
  echo
  echo "Invalid environment type. Valid argument types are 'simulation' and 'artifact'."
  exit 1 
fi

# Pull repo to get latest changes from remote if remote exists
set +e # Do not exit on error for git ls-remote
git ls-remote --exit-code --heads origin $branch_name >/dev/null 2>&1
exit_code=$?
set -e # Re-enable exit on error
if [[ $exit_code == '0' ]]; then
  git fetch --all
  echo
  echo "Git branch '$branch_name' exists in the remote repository; pulling latest changes"
  git pull origin $branch_name
fi

# Check if environment exists already
if [[ $use_shared == 'yes' ]]; then
  # For shared environments, check if venv exists
  venv_path=".venv/$env_name"
  if [[ -d $venv_path ]]; then
    echo
    echo "Virtual environment found at $venv_path"
    env_exists="yes"
    # For venvs, we don't have creation time in the same way, so always check if make_new is set
    if [[ $make_new == 'yes' ]]; then
      echo "Force flag set; will recreate environment"
      create_env="yes"
    else
      echo "Environment exists; no staleness check for venvs (use -f to force rebuild)"
      create_env="no"
    fi
  else
    echo
    echo "Virtual environment $env_name does not exist at $venv_path"
    create_env="yes"
    env_exists="no"
  fi
else
  # For regular conda environments, check with conda
  env_info=$(conda info --envs | grep $env_name | head -n 1)
  if [[ $env_info == '' ]]; then
    # No environment exists with this name
    echo
    echo "Environment $env_name does not exist"
    create_env="yes"
    env_exists="no"
  elif [[ $make_new == 'yes' ]]; then
    # User has requested to make a new environment
    echo
    echo "Making a new environment"
    create_env="yes"
    env_exists="yes"
  else
    env_exists="yes"
    conda activate $env_name
    # Check if existing environment needs to be recreated
    echo
    echo "Existing environment found for $env_name"
    expiration_time=$(date -d "$days_until_stale days ago" +%s)
    creation_time="$(head -n1 $CONDA_PREFIX/conda-meta/history)"
    creation_time=$(echo $creation_time | sed -e 's/^==>\ //g' -e 's/\ <==//g')
    creation_time="$(date -d "$creation_time" +%s)"
    requirements_modification_time="$(date -r $install_file +%s)"
    # Check if existing environment is older than a week or if environment was built 
    # before last modification to requirements file. If so, mark for recreation.
    if [[ $creation_time < $expiration_time ]] || [[ $creation_time < $requirements_modification_time ]]; then
      echo
      echo "Environment is stale; deleting and remaking"
      create_env="yes"
    else
      echo
      echo "Environment is up to date; no action needed"
    fi
  fi
fi

if [[ $create_env == 'yes' ]]; then
  # Build force flag for make commands
  force_flag=""
  if [[ $create_env == 'yes' ]] && [[ $env_exists == 'yes' ]]; then
    force_flag="force=yes"
  fi
  
  # Build lfs flag for make commands
  lfs_flag=""
  if [[ $install_git_lfs == 'yes' ]]; then
    lfs_flag="lfs=yes"
  fi
  
  echo
  if [[ $use_shared == 'yes' ]]; then
    # Use build-shared-env target
    echo "Creating shared virtual environment using 'make build-shared-env'"
    make build-shared-env type=$env_type $force_flag
    
    # Activate the venv
    echo
    echo "Activating virtual environment"
    source .venv/$env_name/bin/activate
    
    # Install git-lfs if requested (venv inherits from base, but we may want to configure it)
    if [[ $install_git_lfs == 'yes' ]]; then
      echo "Configuring git-lfs"
      git lfs install
    fi
  else
    # Use build-env target
    echo "Creating conda environment using 'make build-env'"
    make build-env type=$env_type name=$env_name $force_flag $lfs_flag
    
    # Activate the conda environment
    echo
    echo "Activating conda environment"
    conda activate $env_name
  fi
else
  echo
  echo "Existing environment validated"
  
  # Activate the appropriate environment
  if [[ $use_shared == 'yes' ]]; then
    if [[ -d .venv/$env_name ]]; then
      source .venv/$env_name/bin/activate
    fi
  else
    # Only activate if not already active
    if [[ "$CONDA_DEFAULT_ENV" != "$env_name" ]]; then
      conda activate $env_name
    fi
  fi
fi

echo
echo "*** FINISHED ***"
echo
if [[ $use_shared == 'yes' ]]; then
  echo "Virtual environment is active: $VIRTUAL_ENV"
else
  echo "Conda environment is active: $CONDA_DEFAULT_ENV"
fi
