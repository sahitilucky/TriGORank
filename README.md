# TriGORank
This is the code repository of the TriGORANK model : A Gene Ontology Enriched Learning-to-Rank Framework for Trigenic Fitness Prediction.

**Description:** We developed machine learning models and learning to rank models for recommmending high-fitness triplet gene mutants for wet-lab experiments. We utilized gene ontology to obtain graph representations of triplet genes, gene similarities, gene intersection, path based features to add to the learning to rank model

### Paper

Labhishetty et al., In Proceedings of IEE BIBM 2021,_TriGORank: A Gene Ontology Enriched Learning-to-Rank Framework for Trigenic Fitness Prediction_.
https://doi.org/10.1109/bibm52615.2021.9669503


### Contents

`learning_to_rank_onto_features.py` : learning to rank model with added ontology based features .

`run_script.sh` : For training models and generating results with different top-k relevant and precision settings for training and testing.

### Research Usage

If you use our work in your research please cite:

```
@INPROCEEDINGS{9669503,
  author={Labhishetty, Sahiti and Lourentzou, Ismini and Volk, Michael Jeffrey and Mishra, Shekhar and Zhao, Huimin and Zhai, Chengxiang},
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={TriGORank: A Gene Ontology Enriched Learning-to-Rank Framework for Trigenic Fitness Prediction}, 
  year={2021},
  volume={},
  number={},
  pages={1841-1848},
  doi={10.1109/BIBM52615.2021.9669503}}
```

### License

By using this source code you agree to the license described in https://github.com/sahitilucky/TriGORank/blob/master/LICENSE



