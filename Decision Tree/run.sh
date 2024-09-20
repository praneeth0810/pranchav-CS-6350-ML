#!/bin/sh

echo "Train and Test errors values for Car dataset"
python3 ID3Car.py

echo "Train and Test errors values for Car dataset for bank dataset -unknown as feature value"
python3 ID3B1.py

echo "Train and Test errors values for Car dataset for bank dataset -unknown as missing"
python3 ID3B2.py