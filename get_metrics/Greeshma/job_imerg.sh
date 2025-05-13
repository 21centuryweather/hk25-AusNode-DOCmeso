#!/bin/bash
 
#PBS -l ncpus=6
#PBS -m ae
#PBS -l mem=100GB
#PBS -l jobfs=190GB
#PBS -q normal
#PBS -j oe
#PBS -l wd
#PBS -P nf33
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/up6+scratch/up6+gdata/nf33+scratch/nf33+gdata/hh5+gdata/ia39
#PBS -M g.surendran@unsw.edu.au
#PBS -l wd
 
  
module use /g/data/hh5/public/modules

module load conda/analysis3-unstable


cd /scratch/up6/gs5098/hackathon2025/hk25-AusNode-DOCmeso/get_metrics/Greeshma/

python3 yearly_mean.py