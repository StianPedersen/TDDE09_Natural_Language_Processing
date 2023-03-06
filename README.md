# Standard project baseline: Tagger–parser pipeline

tagger/parser-pipeline implemented in lab 4 and 5 which acts as a baseline but with additional features added.

## Project members

Danny Tran (dantr670)

David Ångström (davan288)

Stian Lockhart Pedersen (stilo759)

Özgür Kofali (ozgko417)

## Results English

| Tagger-parser pipeline         | Tags        | Tagging accuracy | Unlabelled attachment score (UAS) |
| ------------------------------ | ----------- | ---------------- | --------------------------------- |
| Arc-Standard w/ Static oracle  | Retagged    | 0.8917           | 0.6751                            |
| Arc-Standard w/ Dynamic oracle | Retagged    | 0.8917           |                                   |
| Arc-Hybrid w/ Static oracle    | Retagged    | 0.8917           | 0.6924                            |
| Arc-Hybrid w/ Dynamic oracle   | Retagged    | 0.8917           | 0.6390\*                          |
| Arc-Standard w/ Static oracle  | Golden tags | N/A              | 0.7174                            |
| Arc-Standard w/ Dynamic oracle | Golden tags | N/A              |                                   |
| Arc-Hybrid w/ Static oracle    | Golden tags | N/A              | 0.7307                            |
| Arc-Hybrid w/ Dynamic oracle   | Golden tags | N/A              | 0.7063                            |

## Results Nynorsk

| Tagger-parser pipeline         | Tags        | Tagging accuracy | Unlabelled attachment score (UAS) |
| ------------------------------ | ----------- | ---------------- | --------------------------------- |
| Arc-Standard w/ Static oracle  | Retagged    | 0.8951           | 0.6657                            |
| Arc-Standard w/ Dynamic oracle | Retagged    | 0.8951           |                                   |
| Arc-Hybrid w/ Static oracle    | Retagged    | 0.8951           | 0.6612                            |
| Arc-Hybrid w/ Dynamic oracle   | Retagged    | 0.8951           | 0.6492                            |
| Arc-Standard w/ Static oracle  | Golden tags | N/A              |                                   |
| Arc-Standard w/ Dynamic oracle | Golden tags | N/A              |                                   |
| Arc-Hybrid w/ Static oracle    | Golden tags | N/A              | 0.7603                            |
| Arc-Hybrid w/ Dynamic oracle   | Golden tags | N/A              | 0.7521                            |

## Remarks from Marco

- [x] Look into the featurizing model, since arc-hybrid changes the implementation

- [x] Compare our ZERO_SHIFT with Marco's, look up some transitions sequences that can be used for benchmarking

- [ ] Attempt to benchmark both parsers under similar conditions

## Instructions

```
$ git clone git@gitlab.liu.se:stilo759/nlp-project.git

$ cd nlp-project

$ python baseline.py
```
