# TickIDNet
Justen, Lennart, et al. “Identification of Public Submitted Tick Images: A Neural Network Approach.” PLOS ONE, vol. 16, no. 12, Dec. 2021, p. e0260622. PLoS Journals, https://doi.org/10.1371/journal.pone.0260622.

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

## 3) Model
The final model can be downloaded from [Google Drive](https://drive.google.com/file/d/124IfnT6rNLhmPr_3vH62jrK3OLL_fFrI/view?usp=drive_link). See the [docs](https://www.tensorflow.org/guide/keras/save_and_serialize) for more info on loading models.

## 4) Data
Sample images available in this repository were used in the evaluation of TickIDNet. The majority of images used in the development of TickIDNet, including those in the `\Sample Image Data` folder, were initially sourced from [iNaturalist](https://www.inaturalist.org/). The citations below reference the publically available image data from iNaturalist:

*Amblyomma americanum*:  
GBIF.org (16 July 2020) GBIF Occurrence Download https://doi.org/10.15468/dl.4gbcs6. 
  
*Dermacentor variabilis*:  
GBIF.org (16 July 2020) GBIF Occurrence Download https://doi.org/10.15468/dl.tyybke.  
  
*Ixodes scapularis*:  
GBIF.org (16 July 2020) GBIF Occurrence Download https://doi.org/10.15468/dl.sq29u5. 
  
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
