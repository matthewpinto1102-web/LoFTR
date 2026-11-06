Step 1: Clone the Repository & Navigate to the Folder
Copy the Repository Link:
Get the HTTPS URL of your GitHub repository (e.g., [https://github.com/YourUsername/LoFTR.git](https://github.com/YourUsername/LoFTR.git)).

Open Terminal / Command Prompt:
Open terminal or Command Prompt on your computer, navigate to your desired working folder (e.g., Downloads or Projects), and clone:

PowerShell
cd C:\Users\matth\Downloads
git clone <YOUR_GITHUB_REPOSITORY_URL>
Open the Project Folder in VS Code:

Open VS Code.

Go to File > Open Folder... and select the main LoFTR directory (e.g., C:\Users\matth\Downloads\cvpr_loftr\LoFTR).

Critical Check: Ensure your VS Code Explorer sidebar shows the root files (README.md, .gitignore, environment.yaml, notebooks/, src/) directly, not nested inside another folder.

Verify Terminal Working Directory in VS Code:

Open the integrated terminal in VS Code using Ctrl + ~ (or Terminal > New Terminal).

Confirm your terminal prompt shows you are in the root directory:

PowerShell
C:\Users\matth\Downloads\cvpr_loftr\LoFTR>
If you are in a parent folder, use cd to enter the exact LoFTR directory before running any commands:

PowerShell
cd LoFTR
Step 2: Conda Environment & Jupyter Kernel Setup
Create and Activate the Conda Environment:
Run the following commands in the VS Code terminal:

Bash
conda env create -f environment.yaml
conda activate loftr
Install and Register the Jupyter Kernel:
Register the loftr environment so VS Code can use it inside notebooks:

Bash
pip install ipykernel
python -m ipykernel install --user --name=loftr --display-name "Python (LoFTR)"
Step 3: Resolve Debugging Issues & Code Adjustments
Issue A: PyYAML & PyTorch-Lightning Conflict
Modern package managers install updated versions of PyYAML that clash with pytorch-lightning 1.3.5.

Fix: Force-reinstall PyYAML 5.4.1 in your activated loftr environment:

Bash
pip install PyYAML==5.4.1 --force-reinstall
Issue B: CPU vs. GPU (.cuda() to .cpu()) Adjustments
If you do not have a CUDA-compatible Nvidia GPU setup, running .cuda() will throw C++/DLL or device allocation errors.

Fix: In notebooks/demo_single_pair.ipynb, ensure tensor initializations and batch extractions use .cpu():

Python
# Loading the checkpoint onto CPU
matcher = LoFTR(config=cfg['loftr'])
matcher.load_state_dict(torch.load("../weights/indoor_ds_new.ckpt", map_location=torch.device("cpu"))['state_dict'])
matcher.eval()

# Converting batch matches to CPU numpy arrays
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
(Note: Teammates with dedicated Nvidia GPUs can keep or revert these to .cuda().)

Step 4: Verify Weights and Run the Demo
Place Checkpoint Weights:
Ensure the pre-trained weights file (indoor_ds_new.ckpt) is downloaded and placed inside the weights/ directory relative to the repository root:

Plaintext
LoFTR/
├── weights/
│   └── indoor_ds_new.ckpt
├── notebooks/
│   └── demo_single_pair.ipynb
Select the Correct Kernel in VS Code:

Open notebooks/demo_single_pair.ipynb in VS Code.

In the upper right corner of the notebook editor, click Select Kernel.

Choose Python Environments... and select Python (LoFTR).

Execute the Notebook:

Run the notebook cells sequentially.

Pylance static warning banners (such as Import "torch" could not be resolved) can be safely ignored as long as the cells execute.

A successful execution will output a visualization showing Matches: 843.