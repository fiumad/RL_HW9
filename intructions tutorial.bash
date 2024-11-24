ssh RNL2nvsp@blp01.cci.rpi.edu
ssh RNL2nvsp@nplfen01.ccni.rpi.edu
ls
cd data
ls
emacs photo.jpeg //just in case
//Copying photos
cd ./Documents/RPI/Teching/RL/F24
mkdir temp
cd ./temp/
sftp RNL2niev@blp01.cci.rpi.edu
ls
get * /// or //get val_label.csv


cd barn-shared/
ls
cd startup-files/
;sftp


//how to run code
cd scratch-shared/
./conda init
//re-login
conda activate pytorch-env-shared
//shedule code
srun -t 1 --gres=gpu:1 python train_buildings.py


