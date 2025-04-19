#!/usr/bin/env bash
# Script to launch both the ROS bridge and Python 3 training

# ensure workspace is sourced even under sudo or cron
CATKIN_WS=$(realpath "$(dirname "$0")/../../..")
# if this file isn't executable or you use sudo, force it:
source "$CATKIN_WS/devel/setup.bash"
export PYTHONPATH="$CATKIN_WS/devel/lib/python2.7/dist-packages:$PYTHONPATH"

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== JetHexa RL Training Launcher ===${NC}"
# --- START: Forceful Cleanup ---
echo -e "${YELLOW}Attempting to kill previous potentially conflicting processes...${NC}"
pkill -f gzserver > /dev/null 2>&1
pkill -f gzclient > /dev/null 2>&1
pkill -f rosmaster > /dev/null 2>&1
pkill -f gym_bridge_ros.py > /dev/null 2>&1
pkill -f train_ppo.py > /dev/null 2>&1
sleep 1 # Give processes a moment to die
echo -e "${YELLOW}Cleanup attempt finished.${NC}"
# --- END: Forceful Cleanup ---


# Set paths
CATKIN_WS=$(realpath "$(dirname "$0")/../../..")
source $CATKIN_WS/devel/setup.bash


echo "This script will start both the Gazebo simulation (Python 2) and training (Python 3)"
echo

# Default timesteps
DEFAULT_TIMESTEPS=1000000 # Keep default timesteps

# Check for script executability
if [ ! -x "$(realpath "$0")" ]; then
    echo -e "${YELLOW}Making this script executable...${NC}"
    chmod +x "$(realpath "$0")"
fi

if [ ! -x "$CATKIN_WS/src/jethexa_rl/scripts/gym_bridge_ros.py" ]; then
    echo -e "${YELLOW}Making gym_bridge_ros.py executable...${NC}"
    chmod +x "$CATKIN_WS/src/jethexa_rl/scripts/gym_bridge_ros.py"
fi

if [ ! -x "$CATKIN_WS/src/jethexa_rl/scripts/train_ppo.py" ]; then
    echo -e "${YELLOW}Making train_ppo.py executable...${NC}"
    chmod +x "$CATKIN_WS/src/jethexa_rl/scripts/train_ppo.py"
fi

# Parse arguments - Separate launch args from training args
LAUNCH_ARGS=""
TRAIN_ARGS_LIST=() # Use an array to handle spaces in paths correctly
USER_PROVIDED_LOAD_MODEL=false
USER_PROVIDED_LOAD_VECNORM=false
USER_PROVIDED_TIMESTEPS=false
USER_PROVIDED_CURRICULUM_FLAG=false # Track if --curriculum or --no-curriculum is set
START_FRESH=false # <<< ADDED: Flag to force starting new training

while [[ $# -gt 0 ]]; do
    case "$1" in
        gui:=*|headless:=*|paused:=*|world:=*) # Known launch args
            LAUNCH_ARGS="$LAUNCH_ARGS $1"
            shift # past argument
            ;;
        --gui) # Special handling for simple --gui flag
            LAUNCH_ARGS="$LAUNCH_ARGS gui:=true"
            shift # past argument
            ;;
        --load-model)
            if [[ -n "$2" ]]; then
                TRAIN_ARGS_LIST+=("--load-model" "$2")
                USER_PROVIDED_LOAD_MODEL=true
                shift # past argument
                shift # past value
            else
                echo "${RED}Error: --load-model requires a path argument.${NC}" >&2; exit 1
            fi
            ;;
        --load-vecnormalize)
            if [[ -n "$2" ]]; then
                TRAIN_ARGS_LIST+=("--load-vecnormalize" "$2")
                USER_PROVIDED_LOAD_VECNORM=true
                shift # past argument
                shift # past value
            else
                echo "${RED}Error: --load-vecnormalize requires a path argument.${NC}" >&2; exit 1
            fi
            ;;
         --timesteps)
             if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                 TRAIN_ARGS_LIST+=("--timesteps" "$2")
                 USER_PROVIDED_TIMESTEPS=true
                 shift # past argument
                 shift # past value
             else
                 echo "${RED}Error: --timesteps requires a numeric argument.${NC}" >&2; exit 1
             fi
             ;;
         --curriculum)
             USER_PROVIDED_CURRICULUM_FLAG=true # Mark that user specified something
             # We'll handle adding --curriculum later if needed
             shift # past argument
             ;;
         --no-curriculum) # Option to explicitly disable curriculum
             # Do not add --curriculum to args list
             USER_PROVIDED_CURRICULUM_FLAG=true # Mark that user specified something
             TRAIN_ARGS_LIST+=("--no-curriculum") # Pass this through
             shift # past argument
             ;;
        --new) # <<< ADDED: Flag to explicitly start fresh
             START_FRESH=true
             shift # past argument
             ;;
        *) # Unknown option - assume it's for training script or a typo
           echo "${YELLOW}Warning: Unknown argument '$1' passed to run_training.sh. Assuming it's for train_ppo.py.${NC}"
           TRAIN_ARGS_LIST+=("$1")
           shift # past argument
           ;;
    esac
done

# --- Handle --new flag: Remove loading args if present ---
if [ "$START_FRESH" = true ]; then
    echo -e "${YELLOW}--new flag detected. Starting fresh training. Ignoring any --load-model or --load-vecnormalize arguments.${NC}"
    # Filter out load arguments
    TEMP_ARGS_LIST=()
    skip_next=false
    for item in "${TRAIN_ARGS_LIST[@]}"; do
        if [ "$skip_next" = true ]; then
            skip_next=false # Skip the value argument
            continue
        fi
        if [[ "$item" == "--load-model" || "$item" == "--load-vecnormalize" ]]; then
            skip_next=true # Mark the next item (the path) to be skipped
            continue
        fi
        TEMP_ARGS_LIST+=("$item")
    done
    TRAIN_ARGS_LIST=("${TEMP_ARGS_LIST[@]}")
fi
# --- End --new handling ---

if [ "$USER_PROVIDED_TIMESTEPS" = false ]; then
    echo -e "${YELLOW}Using default timesteps: $DEFAULT_TIMESTEPS${NC}"
    TRAIN_ARGS_LIST+=("--timesteps" "$DEFAULT_TIMESTEPS")
fi

# Default to enabling curriculum unless user explicitly set --no-curriculum
if [ "$USER_PROVIDED_CURRICULUM_FLAG" = false ]; then
     TRAIN_ARGS_LIST+=("--curriculum")
     echo -e "${YELLOW}No curriculum flag specified, defaulting to --curriculum.${NC}"
fi

# --- ADD Default loading arguments IF --new is NOT used AND user didn't specify them ---
LATEST_MODEL_PATH="$CATKIN_WS/ppo_jethexa_3004416_final.zip" # Updated to 2M step model
LATEST_VECNORM_PATH="$CATKIN_WS/vec_normalize_3004416_final.pkl" # Updated to 2M step vecnormalize

if [ "$START_FRESH" = false ]; then
    if [ "$USER_PROVIDED_LOAD_MODEL" = false ]; then
        if [ -f "$LATEST_MODEL_PATH" ]; then
            echo -e "${YELLOW}No model specified and --new not used. Defaulting to load latest model: $LATEST_MODEL_PATH${NC}"
            TRAIN_ARGS_LIST+=("--load-model" "$LATEST_MODEL_PATH")
        else
            echo -e "${RED}Warning: Default latest model path not found, cannot load: '$LATEST_MODEL_PATH'${NC}"
        fi
    fi
    if [ "$USER_PROVIDED_LOAD_VECNORM" = false ]; then
        if [ -f "$LATEST_VECNORM_PATH" ]; then
            echo -e "${YELLOW}No vecnormalize specified and --new not used. Defaulting to load latest vecnormalize: $LATEST_VECNORM_PATH${NC}"
            TRAIN_ARGS_LIST+=("--load-vecnormalize" "$LATEST_VECNORM_PATH")
        else
            echo -e "${RED}Warning: Default latest vecnormalize path not found, cannot load: '$LATEST_VECNORM_PATH'${NC}"
        fi
    fi
else
     echo -e "${YELLOW}--new flag specified, skipping default model loading.${NC}"
fi
# --- END Default Loading --- 

# <<< ADD Explicit Learning Rate for Fine-tuning >>>
TRAIN_ARGS_LIST+=("--learning-rate" "3e-5")
echo -e "${YELLOW}Explicitly setting learning rate for fine-tuning: 3e-5${NC}"
# <<< END ADD >>>

# Convert array back to string for logging, properly quoted
TRAIN_ARGS=$(printf "'%s' " "${TRAIN_ARGS_LIST[@]}")

echo -e "${YELLOW}ROS Launch Args: $LAUNCH_ARGS${NC}"
echo -e "${YELLOW}Training Script Args: $TRAIN_ARGS${NC}"

# Initialize PIDs
GAZEBO_PID=""
TRAIN_PID=""

# --- MODIFIED Cleanup Function ---
function cleanup {
  echo -e "${YELLOW}Caught SIGINT/SIGTERM, shutting down processes...${NC}"
  # Send SIGINT first for graceful shutdown attempt to process groups
  if [ -n "$GAZEBO_PID" ] && ps -p $GAZEBO_PID > /dev/null; then
      echo "Sending SIGINT to roslaunch process group (-$GAZEBO_PID)..."
      kill -INT -$GAZEBO_PID 2>/dev/null
  fi
  if [ -n "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null; then
       echo "Sending SIGINT to training process group (-$TRAIN_PID)..."
       kill -INT -$TRAIN_PID 2>/dev/null
       # Consider direct PID kill as fallback: kill -INT $TRAIN_PID 2>/dev/null
  fi

  # Wait a moment for graceful shutdown
  sleep 2

  # Force kill (SIGKILL) if they are still running
  echo "Checking if processes need force kill..."
  if [ -n "$GAZEBO_PID" ] && ps -p $GAZEBO_PID > /dev/null; then
      echo "Force killing roslaunch process group (-$GAZEBO_PID)..."
      kill -KILL -$GAZEBO_PID 2>/dev/null
  fi
  if [ -n "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null; then
       echo "Force killing training process group (-$TRAIN_PID)..."
       kill -KILL -$TRAIN_PID 2>/dev/null
       # Consider direct PID kill as fallback: kill -KILL $TRAIN_PID 2>/dev/null
  fi

  # Additional forceful cleanup for common ROS/Gazebo processes
  echo "Performing additional cleanup (pkill)..."
  pkill -SIGKILL -f gzserver > /dev/null 2>&1
  pkill -SIGKILL -f gzclient > /dev/null 2>&1
  pkill -SIGKILL -f gym_bridge_ros.py > /dev/null 2>&1
  pkill -SIGKILL -f train_ppo.py > /dev/null 2>&1 # Redundant but safe
  pkill -SIGKILL -f rosmaster > /dev/null 2>&1

  echo -e "${GREEN}Shutdown complete.${NC}"
  exit 0 # Exit script after cleanup
}
# --- END MODIFIED Cleanup ---

trap cleanup SIGINT SIGTERM

# Function to run ROS bridge
run_gazebo() {
    echo -e "${GREEN}Starting Gazebo and ROS bridge...${NC}"
    # --- ADDED: Prevent Gazebo online check causing SSL crash ---
    export IGN_FUEL_OFFLINE=1
    echo -e "${YELLOW}Setting IGN_FUEL_OFFLINE=1 to prevent potential SSL connection errors.${NC}"
    # -----------------------------------------------------------
    # Launch gazebo and bridge in background, passing only launch args
    roslaunch jethexa_rl train.launch $LAUNCH_ARGS &
    GAZEBO_PID=$! # Capture roslaunch PID
    if [ -z "$GAZEBO_PID" ]; then
        echo -e "${RED}Failed to capture roslaunch PID.${NC}"
        return 1
    fi
    echo -e "${GREEN}Gazebo (roslaunch) started with PID: $GAZEBO_PID${NC}" # Clarified PID source
    
    # Wait for Gazebo to initialize
    echo -e "${YELLOW}Waiting up to 60 seconds for Gazebo and Bridge node (/gym_interface)...${NC}"
    NODE_CHECK_ATTEMPTS=10 # 10 * 2s = 20s
    NODE_FOUND=false
    for (( i=1; i<=NODE_CHECK_ATTEMPTS; i++ )); do
        # Check if roslaunch process itself is still running
        if ! ps -p $GAZEBO_PID > /dev/null; then
             echo -e "${RED}roslaunch process (PID: $GAZEBO_PID) exited prematurely!${NC}"
             return 1 # roslaunch died, can't continue
        fi

        if rosnode list 2>/dev/null | grep -q "/gym_interface"; then
            NODE_FOUND=true
            echo -e "${GREEN}/gym_interface node found!${NC}"
            break
        fi
        echo -e "${YELLOW}  Attempt $i/$NODE_CHECK_ATTEMPTS: /gym_interface node not found yet, waiting 2 seconds...${NC}"
        sleep 2
    done

    if [ "$NODE_FOUND" = false ]; then
        echo -e "${RED}/gym_interface node not found after $NODE_CHECK_ATTEMPTS attempts - ROS bridge may not be running properly${NC}"
        echo -e "${YELLOW}Available nodes:${NC}"
        rosnode list 2>/dev/null
        echo -e "${YELLOW}Check the roslaunch output above for errors in gym_bridge_ros.py initialization.${NC}"
        return 1 # Return error code
    fi
    return 0 # Success
}

# --- MODIFIED Function to run training ---
run_training() {
    echo -e "${GREEN}Starting RL training in background with args: $TRAIN_ARGS ${NC}"
    cd "$CATKIN_WS" || { echo "${RED}Failed to cd to $CATKIN_WS${NC}"; return 1; } # Ensure cd succeeds
    # Pass train args array to the Python script and run in BACKGROUND
    python3 "$CATKIN_WS/src/jethexa_rl/scripts/train_ppo.py" "${TRAIN_ARGS_LIST[@]}" &
    TRAIN_PID=$! # Capture the PID of the python3 process
    if [ -z "$TRAIN_PID" ]; then
        echo -e "${RED}Failed to capture training script PID.${NC}"
        return 1
    fi
    echo -e "${GREEN}Training script started with PID: $TRAIN_PID${NC}"
    return 0
}
# --- END MODIFIED ---

# --- Main Execution Flow ---
run_gazebo
GAZEBO_EXIT_CODE=$?

if [ $GAZEBO_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Gazebo and ROS bridge appear to be running${NC}"
    run_training
    TRAINING_START_EXIT_CODE=$?

    if [ $TRAINING_START_EXIT_CODE -ne 0 ] || [ -z "$TRAIN_PID" ]; then
        echo -e "${RED}Failed to start training script (Exit code: $TRAINING_START_EXIT_CODE). Cleaning up...${NC}"
        # Cleanup is handled by trap on exit
        exit 1
    fi

else
    echo -e "${RED}Failed to start Gazebo and ROS bridge properly (Exit code: $GAZEBO_EXIT_CODE). Check logs.${NC}"
    # Cleanup is handled by trap on exit
    exit 1
fi

# --- MODIFIED Wait Logic ---
# Wait specifically for the training process PID
echo -e "${YELLOW}Training running (PID: $TRAIN_PID). Waiting for it to complete or for Ctrl+C...${NC}"
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?
echo -e "${YELLOW}Training process (PID: $TRAIN_PID) exited with code $TRAIN_EXIT_CODE.${NC}"

# --- End MODIFIED ---

# Final cleanup is handled by the trap on exit (normal or signaled)
echo -e "${YELLOW}Script finished or training completed. Cleanup will be handled by exit trap.${NC}"

# --- REMOVED old wait logic for GAZEBO_PID ---
# ...
# --- END REMOVAL --- 