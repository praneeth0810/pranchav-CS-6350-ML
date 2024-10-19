 #!/bin/bash
echo "1 - Batch Gradient 2 - Stochastic Gradient 3 - Optimal Weight"
read -p 'Choice: ' choice

if [ "$choice" == 1 ]; then
    python3 BatchGradient.py
fi

if [ "$choice" == 2 ]; then
    python3 StochasticDescendant.py
fi

if [ "$choice" == 3 ]; then
    python3 optimal.py
fi