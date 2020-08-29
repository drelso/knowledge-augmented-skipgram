#!/bin/bash -l
# Batch script to run a GPU job on Legion under SGE.
# 0. Force bash as the executing shell.
#$ -S /bin/bash
# 1. Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=45:00:00
# 3. Request 1 gigabyte of RAM (must be an integer)
#$ -l mem=5G
# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G
# 5. Set the name of the job.
#$ -N Skipgram_BNC
# 6. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/uczcdra/Scratch/knowledge-augmented-skipgram
# 7. Your work *must* be done in $TMPDIR 
cd $TMPDIR
# 8. load the cuda module (in case you are running a CUDA program
module unload compilers mpi
module load compilers/gnu/4.9.2
module load compilers/intel/2020/release
module load cuda/10.1.243/gnu-4.9.2
module load cudnn/7.5.0.56/cuda-10.1
# 9. Run the application - the line below is just a random example.
#virtualenv envs/skipgram_syns_env
echo 'JOB DETAILS:'
echo '$PATH:'
echo $PATH
# echo 'Available compilers'
echo module avail compilers
echo '$HOME'
echo $HOME
#export PYTHONPATH=/home/uczcdra/python_src/Python-3.7.4
#export PATH=/home/uczcdra/python_src/Python-3.7.4:$PATH
export PYTHONPATH=$HOME/Scratch/knowledge-augmented-skipgram/myriad_venv/bin/
export PATH=$HOME/Scratch/knowledge-augmented-skipgram/myriad_venv/bin:$PATH
#source $HOME/sculpting-word-embeddings/envs/skipgram_syns_env/bin/activate
source "$HOME/Scratch/knowledge-augmented-skipgram/myriad_venv/bin/activate" --always-copy
#!/usr/bin/env python3
/usr/bin/time --verbose python3 $HOME/Scratch/knowledge-augmented-skipgram/train_skipgram.py # this prints the usage of the program
# 10. Preferably, tar-up (archive) all output files onto the shared scratch area
tar zcvf $HOME/Scratch/myriad_output/files_from_job_$JOB_ID.tar.gz $TMPDIR
# Make sure you have given enough time for the copy to complete!
