import os
import subprocess

# Assuming you're running the script from the parent directory containing the n_clusters folders.
base_path = os.getcwd()

for n in range(2, 46):
    folder_name = "{}_clusters".format(n)
    script_path = os.path.join(base_path, folder_name, "Mt_logo.py")
    
    # Ensure the script exists before trying to run it
    if os.path.exists(script_path):
        print "Running script in folder:", folder_name
        
        # Changing the directory to the folder because some scripts might be
        # dependent on relative paths within their folder.
        os.chdir(os.path.join(base_path, folder_name))
        
        # Run the script
        subprocess.call(["python", "Mt_logo.py"])
    else:
        print "Script not found in folder:", folder_name

# Change back to the original directory after all scripts have been run
os.chdir(base_path)
