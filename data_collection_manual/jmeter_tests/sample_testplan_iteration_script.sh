#!/bin/bash

# Path to JMeter executable
JMETER_PATH="/home/ubuntu/apache-jmeter-5.6.3/bin/jmeter"

# Path to your JMeter test plan
TEST_PLAN_PATH="/home/ubuntu/jmeter_tests/fd_real_workload_3.jmx"

# Number of times to run the test
RUNS=60

# Loop to run the test multiple times
for ((i=1; i<=RUNS; i++))
do
    echo "Running test iteration $i"
    $JMETER_PATH -n -t $TEST_PLAN_PATH -l "/home/ubuntu/jmeter_tests/results.jtl"
done
