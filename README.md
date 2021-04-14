TickIDNet

## 1) Installation
You can install the required packages with `conda` and `pip`
  
### Anaconda (recommended)
```bash
conda create --name venv python=3.6
conda activate venv
conda install tensorflow pillow numpy pandas
```

### Virtualenv
```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Making predictions
You can run TickIDNet on a batch of images with `predict.py` and the appropriate arguments in your terminal/console
  
```
python predict.py source dest model
```
where `source` is a path to a directory of images, `dest` is the path where you want to save the output CSV file, and `model` is the path to the saved model.

The script will internally crop any images in the directory (your files will remain unchanged) into a square along its shortest side and then resize it to the standard 224x224 input size. The tick will need to be near the center of the image and not, for example, on the upper edge of a portrait-style picture. See `\Sample Image Data` for examples. 

There is also a strong correlation between the relative size of the tick in the image and the networks accuracy. For better results try cropping the pictures closely around the tick. 

## 3) Sample Image Data
The images available in this repository were used in the evaluation of TickIDNet. The majority of images used in the development of TickIDNet, including these, were initially sourced from [iNaturalist](https://www.inaturalist.org/). The citations below reference the publically available image data from iNaturalist:

Ixodes scapularis Say, 1821 in GBIF Secretariat (2019). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2020-11-23.  
  
Dermacentor variabilis Say, 1821 in GBIF Secretariat (2019). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2020-11-23.  
  
Amblyomma americanum Linnaeus, 1758 in GBIF Secretariat (2019). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2020-11-23.
  
### File naming conventions
The files in `\Sample Image Data` are all named in the following way:
```
Genus_species_sex_lifestage_source_alive_feedstage_#.jpg
```
For example: `Dermacentor_variabilis_m_a_ta_unk_unfed_1.jpg`

The labels have the following categories:  
**Genus:** any  
**species:** any  
**sex:** male (m), female (f), unkown (unk)  
**lifestage:** adult (a), nymph (n), larvae (l), unkown (unk)  
**source:** Tick App (ta), WMEL Lab (MCEVBD), TickReport (tr), iNaturalist (iNat)   
**alive:** dead (dead), alive (live), unkown (unk)   
**fed:** fed (fed), unfed (unfed), unkown (unk)   
