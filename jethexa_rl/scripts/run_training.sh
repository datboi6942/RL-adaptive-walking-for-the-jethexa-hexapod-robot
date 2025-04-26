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
# --- START: Forceful Cleanup (More Robust) ---
echo -e "${YELLOW}Attempting to kill previous potentially conflicting processes...${NC}"
pkill -f gzserver > /dev/null 2>&1
pkill -f gzclient > /dev/null 2>&1
pkill -f rosmaster > /dev/null 2>&1
pkill -f roscore > /dev/null 2>&1 # <-- Added roscore
pkill -f gym_bridge_ros.py > /dev/null 2>&1
pkill -f train_ppo.py > /dev/null 2>&1
pkill -f cpg_service_node.py > /dev/null 2>&1 # <-- Added CPG service node
# <<< ADDED: Kill nodelet manager if it gets stuck >>>
pkill -f 'nodelet manager' > /dev/null 2>&1
sleep 2 # <<< Increased sleep duration slightly >>>
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
DEBUG_FLAG=false # <<< ADDED: Flag to track --debug
USER_PROVIDED_LR_VALUE=""
DOMAIN_RANDOMIZATION="false" # Default to false if not specified

# <<< ADDED: Loop to parse arguments >>>
while [[ $# -gt 0 ]]; do
    case "$1" in
        # --- UPDATED: Handle gui:=false and --gui=false specifically ---
        gui:=true|--gui) # Handle gui:=true or --gui flag
            LAUNCH_ARGS="$LAUNCH_ARGS gui:=true"
            shift # past argument
            ;;
        gui:=false|--gui=false) # Handle gui:=false or --gui=false flag
            LAUNCH_ARGS="$LAUNCH_ARGS gui:=false"
            shift # past argument
            ;;
        # --- END UPDATED ---
        headless:=*|paused:=*|world:=*) # Other known launch args
            LAUNCH_ARGS="$LAUNCH_ARGS $1"
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
        --learning-rate)
             if [[ -n "$2" ]]; then
                  TRAIN_ARGS_LIST+=("--learning-rate" "$2")
                  USER_PROVIDED_LR_VALUE="$2" # Store the value
                  shift # past argument
                  shift # past value
             else
                  echo "${RED}Error: --learning-rate requires a value.${NC}" >&2; exit 1
             fi
             ;;
         --curriculum)
             USER_PROVIDED_CURRICULUM_FLAG=true # Mark that user specified something
             # We'll handle adding --curriculum later if needed
             shift # past argument
             ;;
         --no-curriculum)
              USER_PROVIDED_CURRICULUM_FLAG=true # Mark that user specified something
              # Flag that curriculum should NOT be added by default
              # No specific argument added to TRAIN_ARGS_LIST for this
              shift # past argument
              ;;
        --new)
             START_FRESH=true
             shift # past argument
             ;;
        --debug) # <<< ADDED: Handle --debug flag >>>
             DEBUG_FLAG=true
             shift # past argument
             ;;
        --dr)
            DOMAIN_RANDOMIZATION="true"
            shift # past argument
            ;;
        --no-dr)
            DOMAIN_RANDOMIZATION="false"
            shift # past argument
            ;;
        *) # Unknown option - assume it's for train_ppo.py
            # echo -e "${YELLOW}Warning: Unknown argument '$1' passed to run_training.sh. Assuming it's for train_ppo.py.${NC}"
            TRAIN_ARGS_LIST+=("$1") # Add unknown arg to training args
            shift # past argument
            ;;
    esac
done
# <<< END Argument Parsing Loop >>>

# Default to curriculum if user didn't specify --curriculum OR --no-curriculum
if [ "$USER_PROVIDED_CURRICULUM_FLAG" = false ]; then
    echo -e "${YELLOW}No curriculum flag specified, defaulting to --curriculum.${NC}"
    TRAIN_ARGS_LIST+=("--curriculum")
# Else: user explicitly specified --curriculum or --no-curriculum, respect their choice
# (If --curriculum, it was handled above. If --no-curriculum, nothing needs to be added.)
fi

# <<< MODIFIED: Add --debug to training args if flag was set >>>
if [ "$DEBUG_FLAG" = true ]; then
    echo -e "${YELLOW}Debug mode enabled for train_ppo.py.${NC}"
    TRAIN_ARGS_LIST+=("--debug") # Pass the flag to the python script
fi

# Add domain_randomization arg to LAUNCH_ARGS
LAUNCH_ARGS="$LAUNCH_ARGS domain_randomization:=$DOMAIN_RANDOMIZATION"

# --- Modified Loading Logic ---
MODEL_PATH_TO_USE="" # Variable to store the final model path selected

# Determine which model path to use
if [ "$START_FRESH" = false ]; then
    if [ "$USER_PROVIDED_LOAD_MODEL" = false ]; then
        # Find the latest model if user didn't specify one
        echo -e "${YELLOW}Searching for the latest model file...${NC}"
        # Look in /catkin_ws first, then in the models directory
        LATEST_MODEL_PATH=$(ls -t /catkin_ws/ppo_jethexa_*.zip 2>/dev/null | head -n 1)
        if [ -z "$LATEST_MODEL_PATH" ]; then
            MODEL_SEARCH_DIR="/catkin_ws/models" # Using absolute path defined earlier
            if [ -d "$MODEL_SEARCH_DIR" ]; then
                 LATEST_MODEL_PATH=$(ls -t ${MODEL_SEARCH_DIR}/jethexa_ppo_*/ppo_jethexa_*_steps.zip 2>/dev/null | head -n 1)
            fi
        fi

        if [ -n "$LATEST_MODEL_PATH" ] && [ -f "$LATEST_MODEL_PATH" ]; then
            echo -e "${YELLOW}No model specified and --new not used. Defaulting to load latest model: $LATEST_MODEL_PATH${NC}"
            TRAIN_ARGS_LIST+=("--load-model" "$LATEST_MODEL_PATH")
            MODEL_PATH_TO_USE=$LATEST_MODEL_PATH
        else
            echo -e "${RED}Warning: No previous model specified or found. Starting training from scratch.${NC}"
            MODEL_PATH_TO_USE=""
            START_FRESH=true # Force START_FRESH if no model found
        fi
    else
        # User provided --load-model, find the path they specified in TRAIN_ARGS_LIST
        for i in "${!TRAIN_ARGS_LIST[@]}"; do
           if [[ "${TRAIN_ARGS_LIST[$i]}" == "--load-model" ]]; then
               MODEL_PATH_TO_USE="${TRAIN_ARGS_LIST[i+1]}"
               if [ ! -f "$MODEL_PATH_TO_USE" ]; then
                   echo -e "${RED}Error: Specified model file not found: '$MODEL_PATH_TO_USE'${NC}" >&2; exit 1
               fi
               echo -e "${YELLOW}Using user-specified model: $MODEL_PATH_TO_USE${NC}"
               break
           fi
        done
        if [ -z "$MODEL_PATH_TO_USE" ] && [ "$USER_PROVIDED_LOAD_MODEL" = true ]; then
            echo -e "${RED}Error: --load-model specified but path could not be determined.${NC}" >&2; exit 1
        fi
    fi

    # Now, attempt to find matching VecNormalize ONLY if user didn't specify one
    if [ "$USER_PROVIDED_LOAD_VECNORM" = false ]; then
        if [ -n "$MODEL_PATH_TO_USE" ]; then # Check if we have a model path to work with
            # <<< IMPROVED LOGIC TO EXTRACT TIMESTEP/SUFFIX >>>
            MODEL_FILENAME=$(basename "$MODEL_PATH_TO_USE" .zip) # Get filename without dir/extension
            # Try to extract number sequence (timestep or final count) and any suffix
            # Example: ppo_jethexa_3004416_final -> extract 3004416_final
            # Example: ppo_jethexa_420000_steps -> extract 420000_steps
            STEPS_SUFFIX=$(echo "$MODEL_FILENAME" | grep -o '[0-9]\+_[^_]*$')

            MATCHING_VECNORM_PATH=""
            if [ -n "$STEPS_SUFFIX" ]; then
                # Construct the expected VecNormalize path based on the extracted part
                EXPECTED_VECNORM_FILENAME="ppo_jethexa_${STEPS_SUFFIX}_vecnormalize.pkl"
                # Look for it in the same directory as the model OR in /catkin_ws
                MODEL_DIR=$(dirname "$MODEL_PATH_TO_USE")
                if [ -f "${MODEL_DIR}/${EXPECTED_VECNORM_FILENAME}" ]; then
                    MATCHING_VECNORM_PATH="${MODEL_DIR}/${EXPECTED_VECNORM_FILENAME}"
                elif [ -f "/catkin_ws/${EXPECTED_VECNORM_FILENAME}" ]; then
                    MATCHING_VECNORM_PATH="/catkin_ws/${EXPECTED_VECNORM_FILENAME}"
                fi
            fi
            # <<< END IMPROVED LOGIC >>>

            if [ -n "$MATCHING_VECNORM_PATH" ]; then
                echo -e "${YELLOW}No vecnormalize specified. Found matching vecnormalize for model: $MATCHING_VECNORM_PATH${NC}"
                TRAIN_ARGS_LIST+=("--load-vecnormalize" "$MATCHING_VECNORM_PATH")
            else
                echo -e "${RED}Warning: No explicit vecnormalize specified, and could not find matching vecnormalize file for model '$MODEL_PATH_TO_USE' (tried pattern ppo_jethexa_${STEPS_SUFFIX}_vecnormalize.pkl in model dir and /catkin_ws). Training will use fresh normalization statistics.${NC}"
            fi
        else
             # No model path was determined
             echo -e "${YELLOW}No model path specified or found, skipping VecNormalize loading.${NC}"
        fi
    else
         echo -e "${YELLOW}Using user-specified VecNormalize path.${NC}"
    fi
else
     echo -e "${YELLOW}--new flag specified, skipping default model and VecNormalize loading.${NC}"
fi
# --- END Modified Loading Logic ---

# <<< REVISED: Learning Rate Logic >>>
# Decide which learning rate to USE based on whether we are fine-tuning and if user provided one
if [ -n "$MODEL_PATH_TO_USE" ]; then # We are fine-tuning
    if [ -n "$USER_PROVIDED_LR_VALUE" ]; then
        echo -e "${YELLOW}Fine-tuning: Using user-provided learning rate: $USER_PROVIDED_LR_VALUE${NC}"
        # The value is already in TRAIN_ARGS_LIST from parsing loop
    else
        echo -e "${YELLOW}Fine-tuning: No LR provided, defaulting to 1e-5.${NC}"
        TRAIN_ARGS_LIST+=("--learning-rate" "1e-5") # Add the default fine-tuning LR
    fi
else # Starting new training
    if [ -n "$USER_PROVIDED_LR_VALUE" ]; then
        echo -e "${YELLOW}New training: Using user-provided learning rate: $USER_PROVIDED_LR_VALUE${NC}"
        # The value is already in TRAIN_ARGS_LIST
    else
        echo -e "${YELLOW}New training: No LR provided, using default from train_ppo.py.${NC}"
        # Do NOT add --learning-rate, let the Python script use its default
    fi
fi
# <<< END REVISED LR Logic >>>

# Convert array back to string for logging, properly quoted
TRAIN_ARGS=$(printf "'%s' " "${TRAIN_ARGS_LIST[@]}")

echo -e "${YELLOW}ROS Launch Args: $LAUNCH_ARGS${NC}"
echo -e "${YELLOW}Training Script Args: $TRAIN_ARGS${NC}"

# Initialize PIDs
GAZEBO_PID=""
TRAIN_PID=""

# --- MODIFIED Cleanup Function ---
cleanup() {
    echo -e "\n${RED}Caught signal, cleaning up...${NC}"
    # Kill the training script first
    if [ -n "$TRAIN_PID" ]; then
        echo "Stopping Python 3 Training (PID: $TRAIN_PID)..."
        kill -SIGINT $TRAIN_PID # Send SIGINT first for graceful shutdown
        sleep 5 # <<< INCREASED from 2 >>>
        kill -SIGKILL $TRAIN_PID > /dev/null 2>&1 # Force kill if still running
    fi
    # Kill the roslaunch process (which manages Gazebo and bridge)
    if [ -n "$GAZEBO_PID" ]; then
        echo "Stopping Gazebo & Bridge (roslaunch PID: $GAZEBO_PID)..."
        kill -SIGINT $GAZEBO_PID # Send SIGINT first for graceful shutdown
        sleep 5 # <<< INCREASED from 2 >>>
        kill -SIGKILL $GAZEBO_PID > /dev/null 2>&1 # Force kill if still running
    fi
    # Final redundant cleanup
    echo "Performing final process cleanup..."
    pkill -f gzserver > /dev/null 2>&1
    pkill -f gzclient > /dev/null 2>&1
    pkill -f rosmaster > /dev/null 2>&1
    pkill -f roscore > /dev/null 2>&1
    pkill -f gym_bridge_ros.py > /dev/null 2>&1
    pkill -f train_ppo.py > /dev/null 2>&1
    pkill -f cpg_service_node.py > /dev/null 2>&1
    pkill -f 'nodelet manager' > /dev/null 2>&1

    echo -e "${GREEN}Cleanup finished.${NC}"
    exit 0
}
# --- END MODIFIED Cleanup ---

# Trap signals to call cleanup function
trap cleanup SIGINT SIGTERM

# Start Gazebo and ROS bridge in the background
echo "Starting Gazebo and ROS bridge..."
export IGN_FUEL_OFFLINE=1 # Prevent potential SSL connection errors

# Execute roslaunch in the background, passing LAUNCH_ARGS
# <<< MODIFIED: Use LAUNCH_ARGS correctly >>>
roslaunch jethexa_rl train.launch $LAUNCH_ARGS &
GAZEBO_PID=$! # Get the PID of the roslaunch process
echo "Gazebo (roslaunch) started with PID: $GAZEBO_PID"

# Wait for Gazebo and the bridge node to be ready
echo "Waiting up to 60 seconds for Gazebo and Bridge node (/gym_interface)..."
WAIT_COUNT=0
MAX_WAIT=30 # Wait up to 60 seconds (30 checks * 2 seconds)
BRIDGE_READY=false

# <<< ADDED: Check for Gazebo PID existence before waiting >>>
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
     # Check if the roslaunch process is still running
     if ! kill -0 $GAZEBO_PID > /dev/null 2>&1; then
          echo -e "${RED}roslaunch process (PID: $GAZEBO_PID) exited prematurely!${NC}"
          # Try to get exit code if possible (might not work reliably)
          wait $GAZEBO_PID
          EXIT_CODE=$?
          echo "Failed to start Gazebo and ROS bridge properly (Exit code: $EXIT_CODE). Check logs."
          exit 1
     fi

     # Check if the bridge node is listed by rostopic
     if rostopic list | grep -q '/jethexa_rl/observation'; then # Check for a topic published by the bridge
          echo -e "${GREEN}Bridge node seems ready (found topic).${NC}"
          BRIDGE_READY=true
          break
     fi
     # <<< ADDED: Check for /clock topic as a sign Gazebo is running >>>
     if rostopic list | grep -q '/clock'; then
          echo "  Attempt $((WAIT_COUNT + 1))/10: Gazebo seems running (found /clock), but bridge node not fully ready yet..."
     else
         echo "  Attempt $((WAIT_COUNT + 1))/10: Gazebo or Bridge node not found yet, waiting 2 seconds..."
     fi
     sleep 2
     WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ "$BRIDGE_READY" = false ]; then
    echo -e "${RED}Timeout: Gazebo or Bridge node did not become ready within 60 seconds.${NC}"
    cleanup # Clean up processes
    exit 1
fi

# <<< ADDED: Short delay to ensure ROS master is fully ready >>>
sleep 5

echo -e "\n${GREEN}DEBUG: About to launch Python 3 script...${NC}"

# Start Python 3 training script
echo -e "\n${GREEN}Starting Python 3 Training Script...${NC}"
# <<< MODIFIED: Run in background and capture PID >>>
python3 "$CATKIN_WS/src/jethexa_rl/scripts/train_ppo.py" "${TRAIN_ARGS_LIST[@]}" &
TRAIN_PID=$! # Get the PID of the Python 3 script
echo "Python 3 Training started with PID: $TRAIN_PID"

# <<< ADDED: Wait for both background processes to finish >>>
# The script will now pause here until either Gazebo/Bridge (roslaunch) or the
# Python training script exits, or until Ctrl+C is pressed (which triggers cleanup).
echo "Waiting for Gazebo (PID: $GAZEBO_PID) or Training (PID: $TRAIN_PID) to exit..."
wait $GAZEBO_PID $TRAIN_PID
PYTHON3_EXIT_CODE=$? # Capture exit code of whichever process finished first (or wait itself)
echo -e "${YELLOW}A background process exited (Code: $PYTHON3_EXIT_CODE). Initiating cleanup...${NC}"

# Final cleanup (will be called if wait finishes OR if Ctrl+C is caught by trap)
cleanup

# Explicit exit at the end (cleanup function also has exit)
echo "Script finished."
exit $PYTHON3_EXIT_CODE 