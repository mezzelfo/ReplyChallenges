#! /bin/bash
echo compilo
g++ -Ofast main.cpp
echo 1_victoria_lake
time ./a.out 1_victoria_lake.txt
echo 2_himalayas
time ./a.out 2_himalayas.txt
echo 3_budapest
time ./a.out 3_budapest.txt
echo 4_manhattan
time ./a.out 4_manhattan.txt
echo 5_oceania
time ./a.out 5_oceania.txt