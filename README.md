# PSRCNN


Operation:


1. Place the "PSRCNN" folder into "($Caffe_Dir)/examples/"


1. Create and training data from the video by running the parse_video.py script from the src directory
Command: ./parse_video.py --video-path ../data/img-src/test-video.mp4


2. Create and training, validation, and testing data split by running the split_train_test.py script from the src directory
Command: ./split_train_test.py


3. Extract the features from each image set and create the HDF5 files by running generate_database.py script from the src directory
Run this and set k equal to some integer n based on your system capabilities.
Make sure k is the same for all commands
Command: ./generate_database.py --set train --k <k>
Command: ./generate_database.py --set dev --k <k>
Command: ./generate_database.py --set test --k <k>
Command: ./generate_database.py --set train --k <k> --rgb
Command: ./generate_database.py --set dev --k <k> --rgb
Command: ./generate_database.py --set test --k <k> --rgb


4. Train the model by running these commands from ($Caffe_dir)
exp1: ./build/tools/caffe train --solver examples/PSRCNN/exp1/src/PSRCNN_solver.prototxt
exp2: ./build/tools/caffe train --solver examples/PSRCNN/exp2/src/PSRCNN_solver.prototxt --weights <path to chao's .caffemodel file>
exp3: ./build/tools/caffe train --solver examples/PSRCNN/exp3/src/PSRCNN_solver.prototxt


5. Extract the model weights to test the model by running the save_filters.py script from the src directory
exp1: ./save_filters --model ../exp/exp1/src/PSRCNN_mat.prototxt --weights ../exp/exp1/model/PSRCNN_iter_5000000.caffemodel --filename filters --output-path ../exp/exp1/
exp2: ./save_filters --model ../exp/exp2/src/PSRCNN_mat.prototxt --weights ../exp/exp2/model/PSRCNN_iter_100000.caffemodel --filename filters --output-path ../exp/exp2/
exp3: ./save_filters --model ../exp/exp3/src/PSRCNN_mat.prototxt --weights ../exp/exp3/model/PSRCNN_iter_5000000.caffemodel --filename filters --output-path ../exp/exp3/


6. Test the model by running the test.py script from the src directory
exp1: ./test.py --model-filters ../exp/exp1/filters.csv
exp2: ./test.py --model-filters ../exp/exp2/filters.csv
exp3: ./test.py --model-filters ../exp/exp3/filters.csv --rgb
