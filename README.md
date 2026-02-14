This work studies the parallelization of Logistic Regression training using Mini-Batch Gradient Descent on multi-core CPU systems. The objective is to reduce training time while maintaining prediction accuracy. 

Parallel Logistic Regression Training Setup Guide
This document explains how to properly set up and run the PARALLELFAST.py script on both Windows and Linux systems. The guide covers environment creation, dependency installation, CPU configuration, execution steps, and performance considerations.
1. Project Folder Structure
Create a new folder for the project. Inside that folder, keep the following files:

- PARALLELFAST.py
- requirements.txt
2. Create requirements.txt
Create a file named requirements.txt and include the following libraries:

numpy
matplotlib
scikit-learn
3. Creating a Virtual Environment
For Windows:
Open Command Prompt or PowerShell and navigate to the project folder using the cd command.

Create the virtual environment:
python -m venv venv

Activate the environment:
venv\Scripts\activate

If activation is blocked in PowerShell, run:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
For Linux (Ubuntu):
If virtual environment support is not installed:
sudo apt install python3-venv

Navigate to the project folder and create the environment:
python3 -m venv venv

Activate the environment:
source venv/bin/activate
4. Install Required Libraries
After activating the virtual environment, install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

To verify installation:
pip list
5. Check Available CPU Cores
Windows:
echo %NUMBER_OF_PROCESSORS%

Linux:
nproc

Or inside Python:
import multiprocessing as mp
print(mp.cpu_count())
6. Adjust Core Usage in Script
In the script, locate the cores_list variable. Adjust it according to your CPU core count.

For example, if your system has 6 cores:
cores_list = [1, 2, 4, 6]

Avoid exceeding the actual number of available cores.
7. Running the Script
Windows:
python PARALLELFAST.py

Linux:
python3 PARALLELFAST.py
8. Output Files
After execution, two graph files will be generated:

- time_vs_cores.png
- speedup_vs_cores.png

These files show training time and speedup comparison across different core counts.
9. Performance Notes
- The script uses CPU multiprocessing only. It does not use GPU.
- For better performance, increase batch size if sufficient RAM is available.
- Do not use more cores than physically available.
- On Linux systems, advanced users may use taskset or numactl for CPU binding.
10. Windows-Specific Note
Ensure the script contains the following block to avoid multiprocessing errors:

if __name__ == '__main__':
    mp.freeze_support()
    main()
Conclusion
Following the above steps ensures a clean and controlled execution of the parallel training script. Using a virtual environment helps isolate dependencies, and adjusting core usage allows meaningful performance benchmarking across different hardware configurations.

