# Pac-Man
http://ai.berkeley.edu

──▒▒▒▒▒────▄████▄─────                             
─▒─▄▒─▄▒──███▄█▀──────                             
─▒▒▒▒▒▒▒─▐████──█──█──                             
─▒▒▒▒▒▒▒──█████▄──────                             
─▒─▒─▒─▒───▀████▀─────                             

Honoursprogramma (2018)

-----------------------------------------------------------------------------------------------------------------------------

**Fun commands:**


python gridworld.py -a value -i 100 -k 10

python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

python gridworld.py -a q -k 5 -m

python crawler.py

python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 

python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid 

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic 

-----------------------------------------------------------------------------------------------------------------------------
**A couple of nice runtime arguments:**


-x: amount of training episodes

-n: amount of total episodes (amount of played games = n-x)

-l: choose a grid to add (smallClassic, mediumClassic, smallGrid, mediumGrid, minimaxClassic, trickyClassic, trappedClassic)

-r: record games in files

-k: max nr of ghosts to use

--frameTime: slow down or speed up animation

--timeout: maximum calc time for pac-man
