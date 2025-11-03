import subprocess
import os
import sys
import argparse


# --- Configuration ---

OCAML_SOURCE_DIR = "simplex_ocaml/src" 
OCAML_EXE_DIR = "simplex_ocaml"
DEFAULT_EXECUTABLE_NAME = "simplet" 
LOG_FILE_NAME = "ocaml_log.txt"


# --- Helper Functions ---

def convert_to_wsl_path(windows_path: str) -> str:

    """Convert a Windows path (e.g., C:\\...) to a WSL path (e.g., /mnt/c/...)."""

    abs_path = os.path.abspath(windows_path)
    abs_path = abs_path.replace("\\", "/")
    drive = abs_path[0]
    path_no_drive = abs_path[2:]
    return f"/mnt/{drive.lower()}{path_no_drive}"


def check_exe_exists_wsl(exe_dir: str, exe_name: str) -> bool:

    """Checks whether the executable exists within WSL."""

    wsl_exe_dir = convert_to_wsl_path(exe_dir)
    wsl_exe_path = f"{wsl_exe_dir}/{exe_name}"
    
    check_cmd_str = f"test -f {wsl_exe_path}"
    
    wsl_command = ["wsl", "bash", "-c", check_cmd_str]
    
    try:
        subprocess.run(wsl_command, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Executable not found.")
        return False
    except FileNotFoundError:
        print("Error: 'wsl.exe' not found.")
        return False
    


def build_ocaml_project_wsl(source_dir: str) -> bool:

    """
    Function to build the OCaml project using 'make' inside OCAML_SOURCE_DIR.
    """

    print(f"--- Starting project build ---")
    
    wsl_source_dir = convert_to_wsl_path(source_dir)
    
    build_cmd_str = f"cd {wsl_source_dir}; eval $(opam env) make"
    
    wsl_command = ["wsl", "bash", "-c", build_cmd_str]

    try:
        result = subprocess.run(wsl_command, capture_output=True, text=True, check=True, timeout=120)
        print("Build completed successfully.")
        if result.stdout:
            print(f"[WSL STDOUT]:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with exit code {e.returncode}")
        print(f"[WSL STDOUT]:\n{e.stdout}")
        print(f"[WSL STDERR]:\n{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("Build failed: Timeout (exceeded 120 seconds)")
        return False
    except FileNotFoundError:
        print("Error: 'wsl.exe' not found. Make sure WSL is installed and in the PATH.")
        return False


def run_ocaml_solver_wsl(exe_dir: str, exe_name: str, lp_file: str, wsl_log_file_path: str) -> str:

    """
    Runs the compiled OCaml solver on a .lp file.
    
    Args:
        exe_dir: Executable directory
        exe_name: Executable name
        lp_file: Input .lp file
        wsl_log_file_path: ABSOLUTE path for the log
    """

    print(f"\n--- Running OCaml solver ---")
    
    wsl_exe_dir = convert_to_wsl_path(exe_dir)
    wsl_exe_path_relative = exe_name 
    wsl_lp_file = convert_to_wsl_path(lp_file)

    run_cmd_str = (
        f"cd {wsl_exe_dir}; "
        f"eval $(opam env) "
        f"./{wsl_exe_path_relative} {wsl_lp_file} -log {wsl_log_file_path}"
    )
    
    wsl_command = ["wsl", "bash", "-c", run_cmd_str]

    try:
        result = subprocess.run(wsl_command, capture_output=True, text=True, check=True, timeout=30)
        print("Execution completed.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"OCaml execution failed with exit code {e.returncode}")
        print(f"[WSL STDOUT]:\n{e.stdout}")
        print(f"[WSL STDERR]:\n{e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: Executable not found. Command: {' '.join(wsl_command)}")
        raise



# --- Main Script ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Build or Run the OCaml Simplex Solver via WSL.")
    
    parser.add_argument(
        '-b', '--build', 
        action='store_true', 
        help=f'Build project: Compile the OCaml project by executing "make" in OCAML_SOURCE_DIR (Actual directory: {OCAML_SOURCE_DIR}).'
    )
    
    parser.add_argument(
        '-e', '--executable', 
        type=str, 
        default=DEFAULT_EXECUTABLE_NAME, 
        help=f'Executable file path/name (default: {DEFAULT_EXECUTABLE_NAME})',
        metavar='FILE_NAME'
    )
    
    parser.add_argument(
        'input_file', 
        type=str, 
        nargs='?',
        default=None, 
        help='Input .lp file path/name (required if not using -b).'
    )
    
    args = parser.parse_args()

    # --- Check Paths and Config ---
    if not OCAML_SOURCE_DIR or not os.path.exists(OCAML_SOURCE_DIR):
        print(f"Error: OCAML_SOURCE_DIR is invalid or does not exist: {OCAML_SOURCE_DIR}")
        sys.exit(1)
        
    # --- Check the executable directory ---
    if not OCAML_EXE_DIR or not os.path.exists(OCAML_EXE_DIR):
        print(f"Error: OCAML_EXE_DIR is invalid or does not exist: {OCAML_EXE_DIR}")
        sys.exit(1)


    # --- Compute paths for the log ---
    try:
        python_script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        python_script_dir = os.path.abspath(os.getcwd())
        
    windows_log_path = os.path.join(python_script_dir, LOG_FILE_NAME)
    wsl_log_path = convert_to_wsl_path(windows_log_path)

    
    if args.build:
        # --- BUILD MODE ---
        print("Build project.")
        if build_ocaml_project_wsl(OCAML_SOURCE_DIR):
            print("\nBuild completed successfully.")
            sys.exit(0)
        else:
            print("\nBuild failed.")
            sys.exit(1)

    elif args.input_file:
        # --- RUN MODE ---
        print(f"Run project")
        print(f"Target executable: {args.executable}")
        
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            sys.exit(1)

        # --- Check the existence of the executable in OCAML_EXE_DIR ---
        if not check_exe_exists_wsl(OCAML_EXE_DIR, args.executable):
             print(f"\nError: OCaml executable '{args.executable}' not found in '{OCAML_EXE_DIR}'.")
             print("Run this script first with the '-b' flag to build the project:")
             print(f"  python {os.path.basename(__file__)} -b")
             sys.exit(1)

        # --- Run the solver ---
        try:
            output = run_ocaml_solver_wsl(
                OCAML_EXE_DIR,  
                args.executable,
                args.input_file,
                wsl_log_path
            )

            
            print("\n--- Final output from solver (STDOUT) ---")
            print(output)
            print(f"\nLog file generated at: {windows_log_path}")
        
        except Exception as e:
            print(f"Error while running the solver: {e}")
            sys.exit(1)
            
    else:
        print("No operation specified.")
        parser.print_help()
        print("\nUsage examples:")
        print(f"  Build the project: python {os.path.basename(__file__)} -b")
        print(f"  Run a .lp file (with default exe): python {os.path.basename(__file__)} path/to/file.lp")
        print(f"  Run a .lp file (with custom exe):  python {os.path.basename(__file__)} -e custom.exe path/to/file.lp")
        sys.exit(1)