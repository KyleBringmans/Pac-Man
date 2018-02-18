# Pac-Man
http://ai.berkeley.edu

──▒▒▒▒▒────▄████▄─────                             
─▒─▄▒─▄▒──███▄█▀──────                             
─▒▒▒▒▒▒▒─▐████──█──█──                             
─▒▒▒▒▒▒▒──█████▄──────                             
─▒─▒─▒─▒───▀████▀─────                             

Honoursprogramma (2018)

Fun commands:

python gridworld.py -a value -i 100 -k 10

python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

python gridworld.py -a q -k 5 -m

python crawler.py

python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 

python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid 

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic 
