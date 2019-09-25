## Convert spacenet off nadir annotations to yolt annotations

### Expected directory structure
```
.
├── images
│   ├── Atlanta_nadir7_catid_1030010003D22F00
|   ├── Atlanta_nadir8_catid_10300100023BC100
|   ├── .....................................
|   ├── Atlanta_nadir53_catid_1030010003CD4300
|   
├── summaryData
│   ├── Atlanta_nadir7_catid_1030010003D22F00_Train.csv
|   ├── Atlanta_nadir8_catid_10300100023BC100_Train.csv
|   ├── .....................................
|   ├── Atlanta_nadir53_catid_1030010003CD4300_Train.csv
|
├── spacenet2yolt.py
├── spacenet_ann_extractor.py

```

### Usage
`python spacenet2yolt.py`
