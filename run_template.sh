#!/bin/bash
set -e

#########################################
# Your experiment file for submission   #
#########################################

export EXPERIMENT_DEFAULT="experiments/impala_baseline.yaml"
export EXPERIMENT="experiments/AGENT.yaml"
export EXPERIMENT=${EXPERIMENT:-$EXPERIMENT_DEFAULT}

if [[ -z $AICROWD_IS_GRADING ]]; then
  ##########################################################################
  # This section contains commands you would like to run, when running     #
  # the codebase on your machines. During evaluation AICROWD_IS_GRADING    #
  # variable is set, due to which this block will be skipped.              #
  ##########################################################################

  export OUTPUTS_DIR=./outputs
  export RAY_MEMORY_LIMIT=60129542144
  export RAY_CPUS=12
  export RAY_STORE_MEMORY=30000000000

  # Cleaning output directory between multiple runs
  rm -rf ${OUTPUTS_DIR}
  mkdir ${OUTPUTS_DIR}
fi


export VALID_RUN=false
print_banner() {
cat << BANNER
           _____                          _ 
     /\   |_   _|                        | |
    /  \    | |  ___ _ __ _____      ____| |
   / /\ \   | | / __| '__/ _ \ \ /\ / / _  |
  / ____ \ _| || (__| | | (_) \ V  V / (_| |
 /_/    \_\_____\___|_|  \___/ \_/\_/ \__,_|
 
BANNER
}

print_usage() {
cat << USAGE
Available Parameters

--train: for training your agent
--rollout: for agent rollout

Important Environment Variables
EXPERIMENT: path to experiment file you want to use, default: $EXPERIMENT_DEFAULT
CHECKPOINT: (for rollout) path to checkpoint directory, tries to detect automatically otherwise, assumes data present at ~/ray_results.
EPISODES: (for rollout) number of episodes, default: 5
USAGE
}

detect_latest_checkpoint() {
#   export DIRECTORY="$HOME/ray_results"
#   export EXPERIMENT_NAME=$(cat $EXPERIMENT | grep -Ev '^ |^#' | head -n1 | awk -F':' '{print $1}')
#   export LATEST_EXECUTION_DIRECTORY=$(ls -trd $DIRECTORY/$EXPERIMENT_NAME/*/ | tail -n 1)
#   if [ -z $LATEST_EXECUTION_DIRECTORY ]; then
#     echo "Cannot find checkpoint in ~/ray_results, have you run training phase yet?"
#     exit;
#   fi
#   export LATEST_CHECKPOINT_DIRECTORY=$(ls -trd ${LATEST_EXECUTION_DIRECTORY%/}/*/ | tail -n 1)
  export LATEST_EXECUTION_DIRECTORY=~/ray_results/agent-heist/SACIQNAgent_my_procgen_0_2020-11-04_18-19-31orrh6aow
#   export LATEST_EXECUTION_DIRECTORY=~/ray_results/bigfish/saciqn-deter_stoch=True-belief={False}-twin_q=False
  export LATEST_CHECKPOINT_DIRECTORY=$(ls -trd ${LATEST_EXECUTION_DIRECTORY%/}/*/ | tail -n 1)
  export CHECKPOINT=$(find ${LATEST_CHECKPOINT_DIRECTORY%/} | grep "tune_metadata" | head -n1 | awk -F'\.tune_metadata' '{print $1}')
}

print_banner

if [[ " $1 " =~ " --train " ]]; then
  export VALID_RUN=true
  echo "Executing: python train.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}"
  python train.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000} "${@:2}"
  STATUS_CODE=$?
fi


if [[ " $1 " =~ " --rollout " ]]; then
  export VALID_RUN=true
  export ROLLOUT_RUN=$(cat $EXPERIMENT | grep '  run:' | awk '{print $2}')
  export ROLLOUT_ENV=$(cat $EXPERIMENT | grep '  env:' | awk '{print $2}')

  if [ -z $CHECKPOINT ]; then
    detect_latest_checkpoint
  fi
  echo "Rollout with checkpoint: $CHECKPOINT"
  echo "Executing: python ./rollout.py $CHECKPOINT --episodes ${EPISODES:-100} --run $ROLLOUT_RUN --env $ROLLOUT_ENV"
  python ./rollout.py $CHECKPOINT --episodes ${EPISODES:-100} --run $ROLLOUT_RUN --env $ROLLOUT_ENV -n 3 "${@:2}"
  STATUS_CODE=$?
fi


if [ "$VALID_RUN" = false ] ; then
    print_usage
    STATUS_CODE=1
fi
