#!/bin/bash
echo "1 - adaboost\n2 - bagging\n3 - 100 bagged predictors\n4 - Random Forest\n5 - 100 Random Forests"
read -p 'Choice: ' choice

if [ "$choice" == 1 ]; then
    python3 adaboost.py
fi

if [ "$choice" == 2 ]; then
    python3 Bagging.py
fi

if [ "$choice" == 3 ]; then
    python3 bagging_100.py
fi

if [ "$choice" == 4 ]; then
    python3 randomforest.py
fi

if [ "$choice" == 5 ]; then
    python3 randomforest_100.py
fi