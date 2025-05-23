# Evaluation of Meso-scale Degree of Organization of Convection  <img src='https://21centuryweather.org.au/wp-content/uploads/Hackathon-Image-WCRP-Positive-1536x736.jpg' align="right" height="139" />

Observational studies show, the degree to which convection is in a more or less organized state on spatial scales of 100-200 km (mesoscale) is closely coupled with tropical-mean radiative fluxes ([Bony et al., 2020](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019AV000155)). Therefore, mesoscale DOC is likely an important convective feature for a realistic representation of the radiation budget. In this workshop, we calculate mesoscale DOC from model output and satellite observations, highlight model differences in the representation of DOC, and compare model DOC with observed DOC. The first two days will be spent calculating and contrasting different measures of DOC. On the third day, we collaborate with the large-scale environmental conditions group ([hk25-AusNode-LargeScaleP](https://github.com/21centuryweather/hk25-AusNode-LargeScaleP?tab=readme-ov-file)) to compare meso-scale DOC with environmental conditions, and on the final day, we summarize the key findings and tidy up the github repository for future work.



**Project leads** [name, affiliation, email, github username]  
[Philip Blackberg,      Monash University,              philip.blackberg@monash.edu,    [PBlackberg](https://github.com/PBlackberg?tab=repositories)]

**Project members** [name, affiliation, email, github username]  
[Greeshma Surendran,    University of New South Wales,  g.surendran@unsw.edu.au]  
[Alejandra Isaza,       University of New South Wales,  a.isaza@unsw.edu.au]  
[Yinglin Mu,            University of New South Wales,  yinglin.mu@unsw.edu.au]  
[Chris Chambers         The University of Melbourne,    cchambers@unimelb.edu.au]  

**Collaborators:** list here other collaborators to the project.  
[hk25-AusNode-LargeScaleP](https://github.com/21centuryweather/hk25-AusNode-LargeScaleP?tab=readme-ov-file)


**Data:** [name, directory on nci]
* [Unified Model: um_glm_n2560_RAL3](https://github.com/21centuryweather/hackathon-2025-australia-node/blob/main/available_simulations.md), /g/data/qx55/uk_node/glm.n2560_RAL3p3
* [ICON: icon_d3hp003](https://github.com/21centuryweather/hackathon-2025-australia-node/blob/main/available_simulations.md), /g/data/qx55/germany_node/d3hp003.zarr
* [IMERG: V07B](https://gpm.nasa.gov/data/imerg), /g/data/ia39/aus-ref-clim-data-nci/frogs/data/1DD_V1/IMERG_V07B_FC
* [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5), /g/data/rt52/era5/pressure-levels


## Contributing Guidelines
> The group will decide how to work as a team. This is only an example. 

This section outlines the guidelines to ensure everyone can work and collaborate. All project members have write access to this repository, to avoid overlapping and merge issues make sure you discuss the plan and any changes to existing code or analysis.

### Project organisation

All tasks and activities will be managed through GitHub Issues. While most discussions will take place face-to-face, it is important to document the main ideas and decisions on an issue. Issues will be assigned to one or more people and classified using labels. If you want to work on an issue, comment and make sure is assigned to you to avoid overlapping. If you find a problem in the code or a new task, you can open an issue. 

### How to collaborate

* **Main branch:** We want to keep things simple, if you are working on a notebook alone you can push changes to the main branch. Make sure to 1) only add and ccommit that file and nothing else, 2) pull from the remote repo and 3) push.

* **Working on a branch:** if you want to fix or propose a change to someone else code you will need to create a branch and open a pull request. Make sure you explain your suggestion in the pull request message. **This also applies to collaborators outside the project team.**

### Repository structure

This is how the project should look like but make sure to change the name `template-hackathon-project` to something meaningful. 

```bash
template-hackathon-project/
├── LICENCE
├── README.md
├── template_project_hackathon
│   ├── analysis.py
│   ├── __init__.py
│   └── read.py
└── tests
    ├── test_analysis.py
    └── test_read.py
```
* `template_hackathon_project/` this folder will include the code to analysis the data.
* `tests/` this folder contains test code that verifies that your code does what it should.

