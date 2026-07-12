# Remove Conda from PATH (if needed)
export PATH=$(echo $PATH | tr ':' '\n' | grep -v 'miniconda3' | tr '\n' ':')

# Activate the environment
source ~/perch-hoplite/tf-env/bin/activate

# Set environment variables (add to ~/.bashrc to make permanent)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export TF_CPP_VSNPRINTF_OVERRIDE=0
export TF_CPP_MIN_LOG_LEVEL=0

# Run your TensorFlow code
python your_script.py
