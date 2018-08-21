# CS 512 - Data Mining Principles Research Project (Spring 2018)  
Department of Computer Science  
University of Illinois at Urbana-Champaign (UIUC)  

# Members  
Ahmed El-Kishky  
Chia-Wei, Chen (Jack Chen)  
Daniel You  

# Current Progress
(1) Tuning model for KBP dataset, use indicator as replacement of using another stream (mention)  
(2) Get some result on KBP dataset  

# Pipeline
(1) put smaller.tsv into directory data/ <br>
(2) pip install -r requirements.txt <br>
(3) ./prepare_corpus.sh <br>
(4) ./generate_pickle.sh <br>
(5) python src/train.py # for more information use python src/train.py -h  

# To-Do List
- [ ] Testing and tuning the model on KBP dataset  
- [ ] Incorporate subword embedding (mention = Average_Pooling on its corresponding subwords)  
- [ ] Write a supervisor module for easier and faster experiment  
- [ ] Clean up the workspace and make it more understandable  
- [ ] Documentation on all methods  
- [ ] Documentation on separate works   
- [ ] Use utf-8 encoding for future support multi-languages  
- [X] Revise the input arguments of separate modules to achieve better modularity  
- [X] Merge some modules or modulize some frequently used methods
- [X] Remove LateX commmands in volcabulary list
- [X] Add scipts for processing the dataset
- [X] Pipelining the processes 
- [ ] (TBD)


# Known Issues:
- [ ] Unstable training when applying same pretrained embedding (word2vec/FastText) on both mention and context (loss stuck in local minimum)  
- [ ] Volcabulary list generation threading issue: Unknown cause to join threads

# Optimization on code
- [ ] Parametrize some hyperparameters shared throughout the entire work  
- [X] Bugs on displaying tqdm, consider to write progressbar manually
- [X] Better threading code structure
