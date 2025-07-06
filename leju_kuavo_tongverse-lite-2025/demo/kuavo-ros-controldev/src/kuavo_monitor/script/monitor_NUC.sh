#!/bin/bash

# Set CPU usage threshold, in percentage
CPU_USAGE_THRESHOLD=80

# Set temperature threshold, in degrees Celsius
TEMP_THRESHOLD=90

# Set monitoring interval, in seconds
INTERVAL=1

# Current date, for output file name
current_date=$(date +"%Y-%m-%d_%H-%M-%S")
# Output file path
OUTPUT_FILE="/tmp/monitor_cpu_$current_date.txt"

# Clear or create output file
echo "CPU Usage and Temperature Monitoring Log" > $OUTPUT_FILE
echo "Timestamp, CPU Usage (%), CPU Temperature (°C)" >> $OUTPUT_FILE

echo "Starting to monitor CPU usage and temperature..."

# Infinite loop monitoring
while true; do
    # Get the average CPU usage
    CPU_USAGE=$(mpstat 1 1 | tail -n 1 | awk '{print $3 + $4}')
    
    # Check if successfully retrieved CPU usage rate
    if [[ -z "$CPU_USAGE" ]]; then
        echo "Cannot retrieve CPU usage rate"
        exit 1
    fi

    # Convert CPU usage to numeric value and remove decimal points
    CPU_USAGE_NUMERIC=$(printf "%.0f" "$CPU_USAGE")  # Round to nearest integer

    # Get CPU temperature (assuming the temperature of the first core)
    CPU_TEMP=$(sensors | grep 'Core 0' | awk '{print $3}' | tr -d '+°C')

    # Check if successfully retrieved temperature value
    if [[ -z "$CPU_TEMP" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'), Cannot retrieve CPU temperature, error" >> $OUTPUT_FILE
        sleep $INTERVAL
        continue
    fi

    # Convert temperature value to numeric
    CPU_TEMP_NUMERIC=$(printf "%.0f" "$CPU_TEMP")  # Remove decimal points, round to nearest integer

    # Get current timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Record to output file
    echo "$TIMESTAMP, $CPU_USAGE_NUMERIC, $CPU_TEMP_NUMERIC" >> $OUTPUT_FILE

    # Compare CPU usage with threshold
    if [ "$CPU_USAGE_NUMERIC" -gt "$CPU_USAGE_THRESHOLD" ]; then
        WARNING_MESSAGE="$TIMESTAMP Warning: CPU usage is too high! Current usage is $CPU_USAGE%"
        echo "$WARNING_MESSAGE" | tee -a $OUTPUT_FILE
    fi

    # Compare CPU temperature with threshold
    if [ "$CPU_TEMP_NUMERIC" -gt "$TEMP_THRESHOLD" ]; then
        TEMP_WARNING="$TIMESTAMP Warning: CPU temperature is too high! Current temperature is $CPU_TEMP°C"
        echo "[kuavo_monitor] $TEMP_WARNING" | tee -a $OUTPUT_FILE
    else
        NORMAL_MESSAGE="[kuavo_monitor] $TIMESTAMP CPU usage is normal: $CPU_USAGE%, CPU temperature is normal: $CPU_TEMP°C"
        echo "$NORMAL_MESSAGE" | tee -a $OUTPUT_FILE
    fi

    # Wait for the specified interval
    sleep $INTERVAL
done
