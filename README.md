Project Organization

----------
	├── README.md         
	├── requirements.txt   <- To create environment and manage dependencies.
	│                       
        ├── notebooks
    │   ├── simulated_data.ipynb
        │
	├── nta                
	│   ├── __init__.py    
	│   │
	│   ├── data           <- Scripts to download and aggregate data.
	│   │   └── load_data.py
	│   │   └── simulations.py
        │   │
	│   ├── preprocessing  <- Scripts to preprocess raw data.
	│   │   └── signal_processing.py
	│   │   └── quality_control.py # TODO
	│   │   └── sync_data.py
	│   │
	│   ├── events         <- Scripts to analyze neural data around event times.
	│   │   └── align.py
	│   │   └── quantify.py
	│   │   └── predict_da.py # TODO
	│   │
	│   ├── features       <- Scripts to define behavior features for analysis and modeling.
	│   │   └── behavior_features.py
	│   │
	│   └── visualization  <- Scripts to visualize event-aligned neural data.
	│       └── avg_plots.py
	│       └── heatmaps.py
        │       └── peak_plots.py
	│       └── roc_curves.py # TODO
----------

