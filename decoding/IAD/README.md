# Input-guided Aggressive Decoding
Codes (originally from https://github.com/AutoTemp/Shallow-Aggressive-Decoding) for Input-guided Aggressive Decoding (IAD) that is originally proposed in the paper "Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding" (ACL-IJCNLP 2021)
![SAD](aggdec.gif)

## Results
<table>
  <caption> The performance and online inference efficiency evaluation of baseline and our approach in CoNLL-14. </caption>
  <tr> 
    <th> Model </th> 
    <th> P </th>
    <th> R </th>
    <th> F<sub>0.5</sub> </th>
    <th> Speedup </th>
  </tr>
  <tr>
    <th> Transformer-big (beam=5) </th>
    <th> 73.0 </th>
    <th> 38.1 </th>
    <th> 61.6 </th>
    <th> 1.0x </th>
  </tr>
  <tr>
    <th> Our approach (9+3) </th>
    <th> 73.3 </th>
    <th> 41.3 </th>
    <th> 63.5 </th>
    <th> 10.3x </th>
  </tr>
  <tr>
    <th> Our approach (12+2 BART-Init) </th>
    <th> 71.0 </th>
    <th> 52.8 </th>
    <th> 66.4 </th>
    <th> 9.6x </th>
  </tr>
</table>

<table>
  <caption> For reference, the beam=1 and beam=5 results of the state-of-the-art 12+2 (BART-Init) are: </caption>
<thead>
  <tr>
    <th>12+2 BART-Init</th>
    <th colspan="3">CoNLL-14</th>
    <th colspan="3">BEA-19</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Beam</th>
    <th>P</th>
    <th>R</th>
    <th>F<sub>0.5</sub></th>
    <th>P</th>
    <th>R</th>
    <th>F<sub>0.5</sub></th>
  </tr>
  <tr>
    <th>1</td>
    <th>71.0</th>
    <th>52.8</th>
    <th>66.4</th>
    <th>74.7</th>
    <th>66.4</th>
    <th>72.9</th>
  </tr>
  <tr>
    <th>5</th>
    <th>71.4</td>
    <th>52.8</td>
    <th>66.7</td>
    <th>75.8</td>
    <th>66.3</td>
    <th>73.7</td>
  </tr>
</tbody>
</table>
The above models are all single models without ensemble.

## Installation

```
conda create -n IAD python=3.6
conda activate IAD
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
cd fairseq
pip install --editable .
```

## Usage
This section explains how to decode in different ways.
```
PTPATH=/to/path/checkpoint*.pt # path to model file
BINDIR=/to/path/bin_data # directory containing src and tgt dictionaries  
INPPATH=/to/path/conll*.bpe.txt # path to eval file
OUTPATH=/to/path/conll*.out.txt # path to output file
BATCH=xxx
BEAM=xxx
```

## Directly use fairseq's interactive.py to decode:

```
bash interactive.sh $PTPATH $BATCH $BEAM $INPPATH $BINDIR $OUTPATH
```

## use Input-guided Aggressive Decoding:

```
python inference.py --checkpoint-path $PTPATH --bin-data $BINDIR --input-path $INPPATH --output-path $OUTPATH --aggressive 
```
