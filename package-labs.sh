#!/bin/bash


$ES_HOME/utils/labs/labs-assembler.sh  lab-assembly.txt  $@

# link data dir for easy testing
this_dir=$(basename `pwd`)
output_dir=${1:-$this_dir}
(cd "labs.out/$output_dir" ; ln -s ../../data  ./data )

