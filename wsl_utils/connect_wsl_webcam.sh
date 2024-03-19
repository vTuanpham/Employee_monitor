#!/bin/bash

# Script to download, unzip, and run a Windows executable from WSL with error handling

# Function to download and extract cam2ip
download_and_extract_cam2ip() {
    local download_url="https://github.com/gen2brain/cam2ip/releases/download/1.6/cam2ip-1.6-64bit-cv2.zip"
    local target_directory="wsl_utils"
    local zip_file="${target_directory}/cam2ip-1.6-64bit-cv2.zip"
    local executable_file="${target_directory}/cam2ip-1.6-64bit-cv/cam2ip.exe"

    # Check if executable file already exists
    if [ -f "$executable_file" ]; then
        echo "Executable file already exists: $executable_file"
        return 1
    fi

    # Check if zip file already exists
    if [ -f "$zip_file" ]; then
        echo "Zip file already exists: $zip_file"
        return 1
    fi

    # Check if target directory exists, create if not
    if [ ! -d "$target_directory" ]; then
        mkdir -p "$target_directory"
        echo "Created directory: $target_directory"
    fi

    # Download zip file
    wget "$download_url" -O "$zip_file"
    echo "Downloaded: $zip_file"

    # Unzip the downloaded file
    unzip "$zip_file" -d "$target_directory"
    echo "Extracted to: $target_directory"

    # Remove the zip file
    rm "$zip_file"
    echo "Removed zip file: $zip_file"
}

# Function to run a Windows executable from WSL
# Usage: run_win_exe /path/to/your/executable
run_win_exe() {
    # Check for executable path argument
    if [ "$#" -ne 1 ]; then
        echo "Usage: run_win_exe /path/to/your/executable"
        return 1
    fi

    local WSL_EXECUTABLE_PATH="$1"

    # Check if the file exists and is a regular file
    if [ ! -f "$WSL_EXECUTABLE_PATH" ]; then
        echo "Error: The file does not exist or is not a regular file."
        return 1
    fi

    local WINDOWS_TEMP_DIR="/mnt/c/Windows/Temp"
    local EXECUTABLE_FILENAME=$(basename "$WSL_EXECUTABLE_PATH")

    # Copy the executable to the temporary directory in Windows
    cp "$WSL_EXECUTABLE_PATH" "$WINDOWS_TEMP_DIR/$EXECUTABLE_FILENAME"
    echo "Copied $EXECUTABLE_FILENAME to Windows temporary directory."

    # Function to clean up and kill the Windows process
    cleanup() {
        echo "Cleaning up and terminating the process..."
        /mnt/c/Windows/System32/cmd.exe /c taskkill /IM "$EXECUTABLE_FILENAME" /F
        rm -f "$WINDOWS_TEMP_DIR/$EXECUTABLE_FILENAME"
        echo "Cleanup complete."
    }

    # Trap SIGINT and call cleanup
    trap cleanup SIGINT

    local WINDOWS_FORMAT_TEMP_DIR=$(echo "$WINDOWS_TEMP_DIR" | sed -e 's/^\/mnt\/\([a-z]\)\//\1:\\/' -e 's/\//\\/g')
    echo Windows format temp dir: $WINDOWS_FORMAT_TEMP_DIR
    /mnt/c/Windows/System32/cmd.exe /c "$WINDOWS_FORMAT_TEMP_DIR\\$EXECUTABLE_FILENAME" &

    wait $!
    cleanup
    trap - SIGINT
}

# Main script execution
download_and_extract_cam2ip

# Get the IP address of the host from /etc/resolv.conf
nameserver=$(awk '/nameserver/{print $2; exit}' /etc/resolv.conf)
echo "Name Server: $nameserver"

# To watch the video stream, adjust the following line with the correct IP and port
ffplay -i http://$nameserver:56000/mjpeg & # The & denote that this function will run in multithread
echo "Video stream started."
echo "The api is available at http://$nameserver:56000/mjpeg"
echo "This can be used directly with cv2 via cv2.VideoCapture('http://$nameserver:56000/mjpeg')"

# Example usage of running the executable
# Uncomment the line below to run with an example executable path
run_win_exe wsl_utils/cam2ip-1.6-64bit-cv/cam2ip.exe 


# To run the script, type in the wsl terminal:
# ```
# bash connect_webcam.sh
# ```