#!/bin/bash

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
   echo "l     Install git lfs (only applies when creating a new conda environment)."
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

# Pull repo to get latest changes from remote if remote exists
git ls-remote --exit-code --heads origin $branch_name >/dev/null 2>&1
exit_code=$?
if [[ $exit_code == '0' ]]; then
  git fetch --all
  echo
  echo "Git branch '$branch_name' exists in the remote repository; pulling latest changes"
  git pull origin $branch_name
fi

if [[ $use_shared == 'yes' ]]; then
  # Shared environment (venv overlay)
  # Always call make - it handles existence checks and displays appropriate messages
  force_flag=""
  if [[ $make_new == 'yes' ]]; then
    force_flag="force=yes"
  fi
  
  make build-shared-env type=$env_type $force_flag
  
  # Activate the venv
  source .venv/$env_name/bin/activate

else
  # Conda environment
  # Check staleness and only call make if needed
  force_flag=""
  create_env="no"
  lfs_flag=""

  if [[ $install_git_lfs == 'yes' ]]; then
    lfs_flag="lfs=yes"
  fi
  
  if [[ $make_new == 'yes' ]]; then
    # User explicitly requested force rebuild
    force_flag="force=yes"
    create_env="yes"
  else
    # Check if environment exists and if stale
    env_info=$(conda info --envs | grep $env_name | head -n 1)
    if [[ $env_info != '' ]]; then
      # Environment exists, check staleness
      conda activate $env_name
      expiration_time=$(date -d "$days_until_stale days ago" +%s)
      creation_time="$(head -n1 $CONDA_PREFIX/conda-meta/history)"
      creation_time=$(echo $creation_time | sed -e 's/^==>\ //g' -e 's/\ <==//g')
      creation_time="$(date -d "$creation_time" +%s)"
      if [[ $creation_time < $expiration_time ]]; then
        # Environment is stale
        force_flag="force=yes"
        create_env="yes"
      fi
    else
      # Environment doesn't exist
      create_env="yes"
    fi
  fi
  
  if [[ $create_env == 'yes' ]]; then
    make build-env type=$env_type name=$env_name $force_flag $lfs_flag
    conda activate $env_name
  else
    # Environment is up to date, just activate
    [[ "$CONDA_DEFAULT_ENV" != "$env_name" ]] && conda activate $env_name || true
  fi
fi