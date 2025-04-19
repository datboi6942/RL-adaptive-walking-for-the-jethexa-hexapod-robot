#!/usr/bin/env bash
# Script to launch both the ROS bridge and Python 3 training

# ensure workspace is sourced even under sudo or cron
CATKIN_WS=$(realpath "$(dirname "$0")/../../..")
# if this file isn't executable or you use sudo, force it:
source "$CATKIN_WS/devel/setup.bash"
export PYTHONPATH="$CATKIN_WS/devel/lib/python2.7/dist-packages:$PYTHONPATH"

# --- START: Forceful Cleanup ---
echo -e "${YELLOW}Attempting to kill previous processes...${NC}"
pkill -f gzserver
pkill -f gzclient
pkill -f rosmaster
pkill -f gym_bridge_ros.py
pkill -f train_ppo.py
sleep 2 # Give processes time to die
echo -e "${YELLOW}Cleanup attempt finished.${NC}"
# --- END: Forceful Cleanup ---


# Set paths
CATKIN_WS=$(realpath "$(dirname "$0")/../../..")
source $CATKIN_WS/devel/setup.bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== JetHexa RL Training Launcher ===${NC}"
echo "This script will start both the Gazebo simulation (Python 2) and training (Python 3)"
echo

# --- REMOVE Default paths - Loading requires explicit arguments or --new for fresh start ---
# Default paths for fine-tuning from ~143k steps
# DEFAULT_MODEL_PATH="/catkin_ws/models/jethexa_ppo_20250416_214749/ppo_jethexa_143360_steps.zip"
# DEFAULT_VECNORM_PATH="/catkin_ws/models/jethexa_ppo_20250416_214749/vec_normalize_143360_steps.pkl"
# Use the correct path within /catkin_ws, including /src/
# DEFAULT_MODEL_PATH="$CATKIN_WS/ppo_jethexa_280000_steps.zip" # <<< UPDATED: Load from 280k steps
# DEFAULT_VECNORM_PATH="$CATKIN_WS/ppo_jethexa_280000_steps_vecnormalize.pkl" # <<< UPDATED: Load from 280k steps
DEFAULT_TIMESTEPS=1000000 # Keep default timesteps

# --- Debug path check (Removed references to default model paths) ---
echo "DEBUG: CATKIN_WS is set to: $CATKIN_WS"
# echo "DEBUG: Checking Model Path: $DEFAULT_MODEL_PATH"
# ls -l "$DEFAULT_MODEL_PATH"
# echo "DEBUG: Checking VecNorm Path: $DEFAULT_VECNORM_PATH"
# ls -l "$DEFAULT_VECNORM_PATH"
# --- End Debug --- 

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

# --- REMOVE Default Loading Logic Entirely ---
# Add default loading args ONLY if user didn't provide them
# --- RE-ENABLE default loading for resuming training ---
#  echo "DEBUG: USER_PROVIDED_LOAD_MODEL=$USER_PROVIDED_LOAD_MODEL"
#  echo "DEBUG: USER_PROVIDED_LOAD_VECNORM=$USER_PROVIDED_LOAD_VECNORM"
#  if [ "$USER_PROVIDED_LOAD_MODEL" = false ]; then
#      if [ -f "$DEFAULT_MODEL_PATH" ]; then
#          echo -e "${YELLOW}Using default model path: $DEFAULT_MODEL_PATH${NC}"
#          TRAIN_ARGS_LIST+=("--load-model" "$DEFAULT_MODEL_PATH")
#      else
#          # Adjusted Error Message
#          echo -e "${RED}Default model path not found, not adding --load-model: '$DEFAULT_MODEL_PATH' ${NC}"
#      fi
#  fi
#  if [ "$USER_PROVIDED_LOAD_VECNORM" = false ]; then
#      if [ -f "$DEFAULT_VECNORM_PATH" ]; then
#          echo -e "${YELLOW}Using default VecNormalize path: $DEFAULT_VECNORM_PATH${NC}"
#          TRAIN_ARGS_LIST+=("--load-vecnormalize" "$DEFAULT_VECNORM_PATH")
#      else
#          # Adjusted Error Message
#          echo -e "${RED}Default VecNormalize path not found, not adding --load-vecnormalize: '$DEFAULT_VECNORM_PATH' ${NC}"
#      fi
#  fi
# --- End Re-enable ---
# --- END REMOVAL ---

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

# Convert array back to string for logging, properly quoted
TRAIN_ARGS=$(printf "'%s' " "${TRAIN_ARGS_LIST[@]}")

echo -e "${YELLOW}ROS Launch Args: $LAUNCH_ARGS${NC}"
echo -e "${YELLOW}Training Script Args: $TRAIN_ARGS${NC}"

# Function to handle SIGINT
cleanup() {
    echo -e "\n${YELLOW}Shutting down processes...${NC}"
    
    # Kill Gazebo and ROS nodes
    if [ -n "$GAZEBO_PID" ]; then
        echo -e "${YELLOW}Stopping Gazebo (roslaunch process)...${NC}"
        kill -INT $GAZEBO_PID 2>/dev/null
    fi
    
    # Kill roscore explicitly IF IT'S RUNNING, regardless of who started it
    if pgrep -x "rosmaster" > /dev/null; then
        echo -e "${YELLOW}Stopping rosmaster...${NC}"
        pkill -f rosmaster
    fi
    
    # Add specific kills for nodes known to linger
    echo -e "${YELLOW}Performing additional cleanup...${NC}"
    pkill -f gzserver
    pkill -f gzclient
    pkill -f gym_bridge_ros.py
    pkill -f train_ppo.py

    echo -e "${GREEN}Cleanup complete${NC}"
    exit 0
}

# Set up signal handling
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
    GAZEBO_PID=$!
    echo -e "${GREEN}Gazebo started with PID: $GAZEBO_PID${NC}"
    
    # Wait for Gazebo to initialize
    echo -e "${YELLOW}Waiting for Gazebo and Bridge to initialize (30 seconds)...${NC}"
    sleep 30
    
    # Check that the bridge node is running
    echo -e "${YELLOW}Checking if ROS bridge node (/gym_interface) is running...${NC}"
    # Use rosnode list to check for the node, allow some retries
    NODE_CHECK_ATTEMPTS=5
    NODE_FOUND=false
    for (( i=1; i<=NODE_CHECK_ATTEMPTS; i++ )); do
        if rosnode list 2>/dev/null | grep -q "/gym_interface"; then
            NODE_FOUND=true
            break
        fi
        echo -e "${YELLOW}  Attempt $i/$NODE_CHECK_ATTEMPTS: /gym_interface node not found yet, waiting 2 seconds...${NC}"
        sleep 2
    done

    if [ "$NODE_FOUND" = true ]; then
        echo -e "${GREEN}/gym_interface node found - ROS bridge appears to be running${NC}"
        return 0
    else
        echo -e "${RED}/gym_interface node not found after $NODE_CHECK_ATTEMPTS attempts - ROS bridge may not be running properly${NC}"
        echo -e "${YELLOW}Available nodes:${NC}"
        rosnode list 2>/dev/null
        echo -e "${YELLOW}Check the roslaunch output above for errors in gym_bridge_ros.py initialization.${NC}"
        return 1
    fi
}

# Function to run training
run_training() {
    echo -e "${GREEN}Starting RL training with args: $TRAIN_ARGS ${NC}"
    cd "$CATKIN_WS" || exit 1 # Ensure cd succeeds
    # Pass train args array to the Python script
    python3 "$CATKIN_WS/src/jethexa_rl/scripts/train_ppo.py" "${TRAIN_ARGS_LIST[@]}"
    TRAINING_EXIT=$?

    if [ $TRAINING_EXIT -ne 0 ]; then
        echo -e "${RED}Training exited with code $TRAINING_EXIT${NC}"
    else
        echo -e "${GREEN}Training completed successfully${NC}"
    fi
}

# Run the processes in sequence
run_gazebo
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Gazebo and ROS bridge are running${NC}"
    echo -e "${GREEN}Starting training in 5 seconds...${NC}"
    sleep 5
    run_training
else
    echo -e "${RED}Failed to start Gazebo and ROS bridge properly${NC}"
    echo -e "${YELLOW}Check the output above for errors${NC}"
    echo -e "${YELLOW}You can try running these commands manually:${NC}"
    echo -e "  ${GREEN}Terminal 1:${NC} roslaunch jethexa_rl train.launch"
    echo -e "  ${GREEN}Terminal 2:${NC} cd $CATKIN_WS && python3 $CATKIN_WS/src/jethexa_rl/scripts/train_ppo.py $TRAIN_ARGS"
fi

# Wait for user to terminate with Ctrl+C
echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"
wait $GAZEBO_PID

# Clean up when done
cleanup 