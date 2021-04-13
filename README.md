# TickIDNet

## Sample Image Data
The images available in this repository were used in the training of TickIDNet and can also be used to test the network. The majority of images used in the development of TickIDNet, including these, were initially sourced from [iNaturalist](https://www.inaturalist.org/). The citations below reference the publically available image data from iNaturalist:

Ixodes scapularis Say, 1821 in GBIF Secretariat (2019). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2020-11-23.  
  
Dermacentor variabilis Say, 1821 in GBIF Secretariat (2019). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2020-11-23.  
  
Amblyomma americanum Linnaeus, 1758 in GBIF Secretariat (2019). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2020-11-23.
  
# File naming conventions
The files in `\Sample Image Data` are all named according to the following standard:
```
Genus_species_sex_lifestage_source_alive_fed_#.jpg
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
