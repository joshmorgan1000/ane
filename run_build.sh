#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${SCRIPT_DIR}/.build.lock"
LOG_PATH_FILE="${SCRIPT_DIR}/.build.log.path"
# Function to follow log output, handling log file moves
follow_build_log() {
    local last_log_path=""
    local tail_pid=""
    echo -e "\033[1;33m=== Joining in-progress build ===\033[0m"
    echo -e "\033[0;36mAnother build is running (PID: $(cat "$LOCK_FILE")). Following log output...\033[0m"
    echo -e "\033[0;36mPress Ctrl+C to stop following (build will continue in background)\033[0m\n"
    # Follow the log, checking periodically if the path has changed
    while [[ -f "$LOCK_FILE" ]]; do
        if [[ -f "$LOG_PATH_FILE" ]]; then
            current_log_path=$(cat "$LOG_PATH_FILE" 2>/dev/null) || {
                echo -e "\033[1;33mWarning: Log path file exists but could not be read: $LOG_PATH_FILE\033[0m"
                current_log_path=""
            }
            # If log path changed, restart tail
            if [[ "$current_log_path" != "$last_log_path" && -n "$current_log_path" && -f "$current_log_path" ]]; then
                # Kill previous tail if running
                if [[ -n "$tail_pid" ]]; then
                    kill "$tail_pid" 2>/dev/null || true
                    wait "$tail_pid" 2>/dev/null || true
                fi
                last_log_path="$current_log_path"
                # Start new tail in background
                tail -f "$current_log_path" 2>/dev/null &
                tail_pid=$!
            fi
        fi
        # Check every second
        sleep 1
    done
    # Build finished - kill tail and show final status
    if [[ -n "$tail_pid" ]]; then
        kill "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true
    fi
    echo -e "\n\033[1;32m=== Build completed ===\033[0m"
    exit 0
}
# Function to update the log path file
update_log_path() {
    echo "$1" > "$LOG_PATH_FILE"
}
# Function to clean up lock on exit
cleanup_lock() {
    rm -f "$LOCK_FILE" "$LOG_PATH_FILE" 2>/dev/null || true
}
# Check for existing build
if [[ -f "$LOCK_FILE" ]]; then
    existing_pid=$(cat "$LOCK_FILE" 2>/dev/null) || {
        echo -e "\033[1;33mWarning: Lock file exists but could not be read: $LOCK_FILE\033[0m"
        existing_pid=""
    }
    # Check if the process is still running
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        # Process is running - join the log output
        follow_build_log
    else
        # Stale lock file - remove it
        echo -e "\033[1;33mRemoving stale build lock (PID $existing_pid no longer running)\033[0m"
        rm -f "$LOCK_FILE" "$LOG_PATH_FILE" 2>/dev/null || true
    fi
fi
# Create lock file with our PID
echo $$ > "$LOCK_FILE"
# Error handler to copy logs back even on failure
cleanup_on_error() {
    # Clean up lock files
    cleanup_lock
    if [[ -d "${BUILD_TEMP}" ]] && [[ -n "${BUILD_DIR}" ]]; then
        mkdir -p "${BUILD_DIR}"
        if [[ -f "${BUILD_TEMP}/build_libane.log" ]]; then
            cp "${BUILD_TEMP}/build_libane.log" "${BUILD_DIR}/" 2>/dev/null || true
        fi
        for error_log in "${BUILD_TEMP}"/*_errors.log; do
            if [[ -f "$error_log" ]]; then
                cp "$error_log" "${BUILD_DIR}/" 2>/dev/null || true
            fi
        done
    fi
}
trap cleanup_on_error ERR EXIT
# Colors for output
RED='\033[1;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BLUE='\033[0;34m'
BRIGHT_CYAN='\033[1;96m'
BRIGHT_PINK='\033[1;95m'
BRIGHT_RED='\033[1;91m'
BRIGHT_GREEN='\033[1;92m'
WHITE='\033[1;97m'
LIGHT_GREY='\033[0;37m'
MEDIUM_GREY='\033[0;90m'
NC='\033[0m' # No Color
IS_CLAUDE=OFF
echo -e "${GREEN}=== libane Build Script ===${NC}"
# Detect OS
OS="unknown"
PKG_MANAGER=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Detected: macOS"
    if command -v brew &> /dev/null; then
        PKG_MANAGER="brew"
    else
        echo -e "${YELLOW}Warning: Homebrew not found. Install from https://brew.sh${NC}"
    fi
else
    echo -e "${RED}Unsupported OS: $OSTYPE. This build script only supports macOS.${NC}"
    exit 1
fi
# Make sure we are in the project root directory (the same directory as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "${SCRIPT_DIR}/src" ]]; then
    echo -e "${RED}Error: This script must be run from the project root directory.${NC}"
    exit 1
fi
# Clean build directories and node_modules if --clean was specified
export BUILD_DIR="${SCRIPT_DIR}/build"
if [[ -d "${BUILD_DIR}" ]]; then
    rm -rf "${BUILD_DIR}"
fi
# Create build directory for C++ build (no-op if already exists)
mkdir -p "${BUILD_DIR}"
# Initialize build log
LOG_FILE="${BUILD_DIR}/build_libane.log"
# Rotate old build log if it exists
if [[ -f "${LOG_FILE}" ]]; then
    # Use the file's modification timestamp, not current time
    TIMESTAMP=$(date -r "${LOG_FILE}" +%Y%m%d_%H%M%S)
    mv "${LOG_FILE}" "${BUILD_DIR}/build_libane_${TIMESTAMP}.log"
fi
echo "=== libane Build Log ===" > "${LOG_FILE}"
echo "Build started: $(date)" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"
# Update log path for any "joiners"
update_log_path "${LOG_FILE}"
# Now cd to build directory for C++ build
cd ${BUILD_DIR}
# Use make on Linux (ninja has issues), ninja on macOS
if [[ "$OS" == "linux" ]]; then
    echo -e "${YELLOW}Using make for Linux build${NC}"
    CMAKE_ARGS+=(-G "Unix Makefiles")
    BUILD_COMMAND="make"
elif command -v ninja &> /dev/null; then
    CMAKE_ARGS+=(-G Ninja)
    BUILD_COMMAND="ninja"
else
    echo -e "${YELLOW}WARNING: ninja not found. Trying make instead${NC}"
    CMAKE_ARGS+=(-G "Unix Makefiles")
    BUILD_COMMAND="make"
fi
middle=''
right=''
# Add build type
CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=${BUILD_TYPE})
# GPU neural math backend (auto-detects by default, override with --vulkan/--cuda/--cpu-only)
if [[ -n "$GPU_BACKEND" ]]; then
    CMAKE_ARGS+=(-DLIBANE_GPU_BACKEND=${GPU_BACKEND})
fi
# Run CMake
if [[ "$VERBOSE" == "ON" ]]; then
    echo -e "${YELLOW}Configuring with CMake in verbose mode...${NC}"
    cmake "${CMAKE_ARGS[@]}" ..
else
    printf "${YELLOW}Configuring with CMake...${NC}\n"
    echo "=== CMake Configuration ===" >> "${LOG_FILE}"
    CYAN_LINES=0
    PURPLE_LINES=0
    PINK_LINES=0
    MFLD_LINES=0
    GREY_LINES=0
    GREY_LINES_AGAIN=0
    DARK_BLUE_LINES=0
    cmake "${CMAKE_ARGS[@]}" .. 2>&1 | while IFS= read -r line; do
        if [[ "$line" == *"______________________________________________________" ]]; then
            printf "${PURPLE}${line}${NC}\n"
            CYAN_LINES=3
        elif [[ "$line" == *"┌───"* ]]; then
            printf "%s\n" "$line"
            MFLD_LINES=4
        elif [[ "$line" == *"-------------------------------------------------------"* ]]; then
            if [[ $GREY_LINES -eq 0 ]]; then
                printf "${MEDIUM_GREY}${line}${NC}\n"
                if [[ $GREY_LINES_AGAIN -eq 1 ]]; then
                    GREY_LINES_AGAIN=0
                else
                    GREY_LINES=2
                fi
            fi
        elif [[ $GREY_LINES -gt 0 ]]; then
            printf "${WHITE}${line}${NC}\n"
            GREY_LINES=$((GREY_LINES - 1))
            if [[ $GREY_LINES -eq 0 ]]; then
                GREY_LINES_AGAIN=1
            else
                DARK_BLUE_LINES=1
            fi
        elif [[ $DARK_BLUE_LINES -gt 0 ]]; then
            printf "${MEDIUM_GREY}${line}${NC}\n"
            DARK_BLUE_LINES=$((DARK_BLUE_LINES - 1))
        elif [[ $MFLD_LINES -gt 0 ]]; then
            printf "%s${BRIGHT_CYAN}%s${NC}%s\n" "${line:0:4}" "${line:4:51}" "${line:55}"
            MFLD_LINES=$((MFLD_LINES - 1))
            if [[ $MFLD_LINES -eq 0 ]]; then
                SIMP_LINES=1
            fi
        elif [[ $SIMP_LINES -gt 0 ]]; then
            printf "%s${WHITE}%s${BRIGHT_CYAN}%s${NC}%s\n" "${line:0:4}" "${line:4:36}" "${line:40:15}" "${line:55}"
            SIMP_LINES=$((SIMP_LINES - 1))
        elif [[ $CYAN_LINES -gt 0 ]]; then
            printf "${CYAN}${line}${NC}\n"
            CYAN_LINES=$((CYAN_LINES - 1))
            if [[ $CYAN_LINES -eq 0 ]]; then
                PURPLE_LINES=1
            fi
        elif [[ $PURPLE_LINES -gt 0 ]]; then
            printf "${PURPLE}${line}${NC}\n"
            PURPLE_LINES=$((PURPLE_LINES - 1))
            PINK_LINES=1
        elif [[ $PINK_LINES -gt 0 ]]; then
            printf "${BLUE}${line}${NC}\n"
            PINK_LINES=$((PINK_LINES - 1))
        else
            echo "$line"
        fi
        echo "$line" >> "${LOG_FILE}"
        # If the line contains the word "error" or "failed", print it in red
        if [[ "$line" == *"error"* || "$line" == *"failed"* ]]; then
            echo -e "${RED}$line${NC}"
        fi
    done
fi
if grep -q "errors occurred!" "${LOG_FILE}"; then
    echo -e "\n${RED}=== CMake Configuration Failed! ===${NC}"
    echo "  Build type: ${BUILD_TYPE}"
    echo -e "${NC}For more details, check the build log: ${LOG_FILE}"
    exit 1
else
    echo -e "\n${GREEN}=== CMake Configuration Complete ===${NC}"
    echo "  Build type: ${BUILD_TYPE}"
fi
# Build command
set +e  # Disable exit on error temporarily
printf "${YELLOW}Building libane.a...${NC}"
echo "" >> "${LOG_FILE}"
echo "=== C++ Build (${BUILD_COMMAND}) ===" >> "${LOG_FILE}"
PRINT_ERROR_ON=OFF
if [[ "$VERBOSE" == "ON" ]]; then
    ${BUILD_COMMAND} 2>&1 | tee -a "${LOG_FILE}"
else
    ${BUILD_COMMAND} 2>&1 | while IFS= read -r line; do
        # Change the color to red for errors
        if [[ "$line" == "FAILED:"* ]]; then
            PRINT_ERROR_ON=ON
            echo " " >> "${BUILD_DIR}/build_libane_errors.log"
        elif [[ "$line" == *"linker command failed"* ]]; then
            PRINT_ERROR_ON=OFF
        elif [[ "$line" == "[\d]+ error[s]* generated[\.]"* ]]; then
            PRINT_ERROR_ON=OFF
        elif [[ "$line" == *"error[s]* generated"* ]]; then
            PRINT_ERROR_ON=OFF
        elif [[ "$line" == "[\[][\d]+/[\d]+[\]]"* ]]; then
            PRINT_ERROR_ON=OFF
        fi
        if [[ "$PRINT_ERROR_ON" == "ON" ]]; then
            echo "$line" >> "${BUILD_DIR}/build_libane_errors.log"
        else
            if [[ "$IS_CLAUDE" == "OFF" ]]; then
                printf "."
            fi
        fi
        echo "$line" >> "${LOG_FILE}"
    done
fi
set -e  # Re-enable exit on error
# Check if the error log file exists and has content
if [[ -s "${BUILD_DIR}/build_libane_errors.log" ]]; then
    echo -e "\n${RED}=== Build Failed! ===${NC}"
    echo "  Build directory: $(pwd)"
    echo "  Build type: ${BUILD_TYPE}"
    echo -e "${NC}For more details, check the build log: ${LOG_FILE}"
    echo " "
    echo -e "${RED}Errors encountered during build:${NC}"
    cat "${BUILD_DIR}/build_libane_errors.log"
    exit 1
fi
echo -e "${BRIGHT_GREEN}done!${NC}"
set -e
echo -e "\n ${CYAN}     ⣇⡀ ⡀⢀ ⠄ ⡇ ⢀⣸   ⢀⣀ ⡀⢀ ⢀⣀ ⢀⣀ ⢀⡀ ⢀⣀ ⢀⣀ ⣰⡁ ⡀⢀ ⡇${NC}"
echo -e " ${CYAN}     ⠧⠜ ⠣⠼ ⠇ ⠣ ⠣⠼   ⠭⠕ ⠣⠼ ⠣⠤ ⠣⠤ ⠣⠭ ⠭⠕ ⠭⠕ ⢸  ⠣⠼ ⠣${NC}"
echo -e " "
echo -e "\n  Build directory: $(pwd)"
echo -e "  Build type:${CYAN} ${BUILD_TYPE} ${NC}"
echo -e " "
# Clean up lock files on successful exit
cleanup_lock
exit 0
