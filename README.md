# Standard project baseline: Tagger–parser pipeline

tagger/parser-pipeline implemented in lab 4 and 5 which acts as a baseline.

## Project members

Danny Tran (dantr670)

David Ångström (davan288)

Stian Lockhart Pedersen (stilo759)

Özgür Kofali (ozgko417)

## Results

| Tagger-parser pipeline       | Tagging accuracy | Unlabelled attachment score (UAS) |
| ---------------------------- | ---------------- | --------------------------------- |
| Baseline                     | 0.8905           | 0.6910                            |
| Arc-Hybrid w/ Dynamic oracle | N/A              | 0.6685                             |

## Remarks from Marco

- [ ] Look into the featurizing model, since arc-hybrid changes the implementation

- [ ] Compare our ZERO_SHIFT with Marco's, look up some transitions sequences that can be used for benchmarking

- [ ] Attempt to benchmark both parsers under similar conditions

## Instructions

```
$ git clone git@gitlab.liu.se:stilo759/nlp-project.git

$ cd nlp-project

$ python baseline.py
```
