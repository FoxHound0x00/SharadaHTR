# SharadaHTR
A handwritten text-recognition system for Sharada script

### Environment setup

1. Run the following command:
    ```python -m venv sharada_env```

2. Activate the environment
    ```source sharada_env/bin/activate```

3. Install the required libraries
    ```pip3 install -r requirements.txt```

### Data Preparation

1. Unzip the provided dataset in the ```data``` folder.

    ```tar -xf sharada.tar.gz```

2. Run the following to convert annotations into a usable format.

    ```python filter.py && python extract.py``` 
