# CS 512 - Data Mining Principles Research Project (Spring 2018)  
University of Illinois at Urbana-Champaign (UIUC)  

# Members  
Ahmed El-Kishky  
Chia-Wei, Chen (Jack Chen)  
Teng-Jie, You (Daniel You)  

# Pipeline
(1) put smaller.tsv into directory data/ <br>
(2) ./prepare_corpus.sh <br>
(3) ./generate_pickle.sh <br>
(4) python src/train.py # for more information use python src/train.py -h

# To-Do List
- [X] Revise the input arguments of separate modules to achieve better modularity
- [X] Merge some modules or modulize some frequently used methods
- [X] Remove LateX commmands in volcabulary list
- [X] Add scipts for processing the dataset
- [X] Pipelining the processes 
- [ ] Documentation on all methods
- [ ] Documentation on separate works 
- [ ] Use utf-8 encoding for future support multi-languages
- [ ] (TBD)


# Known Issues:
- [ ] Volcabulary list generation threading issue: Unknown cause to join threads

# Optimization on code
- [X] Bugs on displaying tqdm, consider to write progressbar manually
- [X] Better threading code structure
