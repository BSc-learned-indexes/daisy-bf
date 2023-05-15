# Benchmarking Learned Bloom Filters 
## About ℹ️

### Motivation
This project has been created as part of a Bachelor project at IT-University of Copenhagen in Spring 2023. It aims to provide an open and transparent way to benchmark various Learned Bloom Filters. This project contains implementation of the following Bloom Filters in `/bloom_filters`: 
- [Regular Bloom Filter](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwia6bOw9vb-AhULRvEDHfeEDLQQFnoECBEQAQ&url=https%3A%2F%2Fdl.acm.org%2Fdoi%2F10.1145%2F362686.362692&usg=AOvVaw1ki8O_wp0JyqNAObHMFWn1) 
- [Partitioned Learned Bloom Filter](https://arxiv.org/abs/2006.03176)
- [Adaptive Learned Bloom Filter](https://arxiv.org/abs/1910.09131)
- [Daisy Bloom Filter](https://arxiv.org/abs/2205.14894)

### Data sets 
Two different data sets are provided. 

#### URL data set 
A data set containing labeled 450,176 URLs. 345,738 beneign and 104,438 malicous. The data set is provided in `/data/raw/url_data` [(source)](https://www.kaggle.com/code/siddharthkumar25/detect-malicious-url-using-ml).

To vectorize the URL data run: 

```
make vectorize
```

#### Synthetic Zipfean data set 
A synthetic Zipfean data set is provided. This can be regenerated:

```
make zipf
```

## Installation ⚙️

### 1. Clone the repository
```
git clone git@github.com:BSc-learned-indexes/daisy-bf.git
cd daisy
```

### 2. Recommended: Creating a virtual environment 
We recommend that you install this project's dependencies in an isolated enviroment. If you are unfamiliar with this concept you can read more about it [here](https://docs.python.org/3/library/venv.html).
#### Create the environment 
```
python venv -m ~/.virtualenvs/daisy
```

#### Source the environment 
```
source ~/.virtualenvs/daisy/bin/activate
```

### 3. Installing dependencies
```
pip install -r requirements.txt 
```

## Usage 📈

### Benchmarking the Bloom Filters 
We have provided a template to run a benchmarking experiment with the following settings:
- Large Random Forest Classifier as model
- 1 - px as the query distribution
- URL data set 
- Full key set 

A series of `make` commands are provided to build the filters:

#### Build Adaptive Learned Bloom Filter 
```
make adabf 
```

#### Build Partitioned Learned Bloom Filter 
```
make plbf 
```

#### Build Daisy Bloom Filter 
```
make daisy 
```

#### Build all Bloom Filters 
Note: this command takes a while 🐌
```
make all 
```

#### Plot all Bloom Filters
```
make plot_all 
```

#### Plot all the Learned Bloom Filters (excludes the regular Bloom Filter)
```
make plot_learned_bf
```


## Extra 🤓

The directory `/experiments` contains all the data that is presented in the Bachelor's thesis's Experiments section. Here is a link to the thesis for the interested.





