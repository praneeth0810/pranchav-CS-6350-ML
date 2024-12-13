#!/bin/bash

# Run all Python scripts sequentially
echo "Running forwbackpass.py..."
python3 forwbackpass.py

echo "Running sgd.py..."
python3 sgd.py

echo "Running neuralpytorch.py..."
python3 neuralpytorch.py

echo "Running MAP.py..."
python3 MAP.py

echo "Running MLE.py..."
python3 MLE.py


echo "All scripts executed successfully!"
