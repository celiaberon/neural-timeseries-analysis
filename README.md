Project Organization

├── README.md          │
├── requirements.txt   <- To create environment and manage dependencies.
│                       
├── nta                
│   ├── __init__.py    
│   │
│   ├── data           <- Scripts to download and aggregate data.
│   │   └── load_data.py
│   │
│   ├── preprocessing  <- Scripts to preprocess raw data.
│   │   └── signal_processing.py
│   │   └── quality_control.py # TODO
│   │   └── sync_data.py # TODO
│   │
│   ├── events         <- Scripts to analyze neural data around event times.
│   │   └── align.py
│   │   └── quantify.py # TODO
│   │   └── predict_da.py # TODO
│   │
│   ├── features       <- Scripts to define behavior features for analysis and modeling.
│   │   └── behavior_features.py
│   │
│   └── visualization  <- Scripts to visualize event-aligned neural data.
│       └── avg_plots.py
│       └── heatmaps.py
│       └── roc_curves.py

