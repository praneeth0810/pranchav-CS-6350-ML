#!/bin/bash

# Run Standard Perceptron
echo "Running Standard Perceptron..."
python3 StandardPerceptron.py
echo "Standard Perceptron completed."
echo "--------------------------------"

# Run Voted Perceptron
echo "Running Voted Perceptron..."
python3 VotedPerceptron.py
echo "Voted Perceptron completed."
echo "--------------------------------"

# Run Average Perceptron
echo "Running Average Perceptron..."
python3 AveragePerceptron.py
echo "Average Perceptron completed."
echo "--------------------------------"

echo "All Perceptron models have been run successfully."
