"""Utilities to load ACAS Xu .nnet files and synthesize labeled datasets."""

import glob
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os


TAU_VALUES = {
    1: 0.0, 2: 1.0, 3: 5.0, 4: 10.0, 5: 20.0, 
    6: 40.0, 7: 60.0, 8: 80.0, 9: 100.0
}

PREV_ADVISORY_VALUES = {
    1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0
}


class ACASXuOracle:
    """Thin loader/executor for the ACAS Xu feed-forward networks."""
    def __init__(self, filename):
        self.filename = filename
        self.weights = []
        self.biases = []
        self.means = []
        self.ranges = []
        self.min_inputs = []
        self.max_inputs = []
        self._load_nnet()

    def _to_float_vector(self, seq, name: str):
        """Convert to a 1D float vector and raise if the shape is invalid."""
        cleaned = [x for x in seq if str(x).strip() != ""]
        arr = np.asarray(cleaned, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D vector")
        return arr

    def _load_nnet(self):
        """Parse the custom .nnet format."""
        with open(self.filename, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("//")]

        # Header parsing
        # Line 0: num_layers, input_size, output_size, max_layer_size
        try:
            header = [int(x) for x in lines[0].split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid header in the first .nnet line: '{lines[0]}'") from exc
        if len(header) < 2:
            raise ValueError(f"Incomplete header: expected at least num_layers and input_size, got {header}")
        num_layers = header[0]
        input_size = header[1]
        

        self.min_inputs = self._to_float_vector(lines[3].split(","), "min_inputs")
        
 
        self.max_inputs = self._to_float_vector(lines[4].split(","), "max_inputs")
        
 
        self.means = self._to_float_vector(lines[5].split(","), "means")
        

        self.ranges = self._to_float_vector(lines[6].split(","), "ranges")


        current_line = 7
        for i in range(num_layers):
            pass
    
        all_floats = []
        with open(self.filename, 'r') as f:
            for line in f:
                if line.strip().startswith("//") or not line.strip(): continue
                parts = line.split(",")[:-1] 
                if not parts: parts = line.split(",")
                try:
                    vals = [float(x) for x in parts if x.strip() != '']
                    all_floats.extend(vals)
                except:
                    pass
        

        layer_sizes = [5, 50, 50, 50, 50, 50, 50, 5]
    
        
        idx = 0

        raw_data = lines[7:] 
        
        val_idx = 0
        flat_data = []
        for l in raw_data:
            flat_data.extend([float(x) for x in l.split(",") if x.strip()])
            
        idx = 0
        for i in range(len(layer_sizes) - 1):
            rows = layer_sizes[i+1]
            cols = layer_sizes[i]
            

            w_size = rows * cols
            w = np.array(flat_data[idx : idx + w_size]).reshape(rows, cols)
            self.weights.append(w)
            idx += w_size
            

            b_size = rows
            b = np.array(flat_data[idx : idx + b_size])
            self.biases.append(b)
            idx += b_size

    def predict(self, inputs):
        """
        inputs: numpy array (N, 5) not normalized
        """
        
        inp_norm = (inputs - self.means[:-1]) / self.ranges[:-1]
        
        
        activation = inp_norm.T 
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
           
            z = np.dot(W, activation) + b[:, np.newaxis]
            
           
            if i < len(self.weights) - 1:
                activation = np.maximum(0, z) 
            else:
                activation = z 
                
        return activation.T 

    def generate_dataset(self, num_samples=10000):
        
        random_inputs = np.random.uniform(0, 1, (num_samples, 5))
        
        delta = np.subtract(self.max_inputs, self.min_inputs, dtype=float)
        real_inputs = self.min_inputs + random_inputs * delta
        
        print("Computing oracle outputs...")
        labels = self.predict(real_inputs)
        
        return real_inputs, labels  



# --- EXECUTION ---
if __name__ == "__main__":
    default_output_dir = os.path.join(os.getcwd(), "ACAS_Xu_datasets")
    parser = argparse.ArgumentParser(description="Generate dataset from ACAS Xu .nnet network")

    parser.add_argument(
        'input_file', 
        nargs='?',
        default=None,
        type=str, 
        help='.nnet file path'
        )
    
    parser.add_argument(
        '-d', '--directory',
        type = str,
        default=default_output_dir,
        help=f"Output directory (default: {default_output_dir})",
        metavar='DIR'
    )
    args = parser.parse_args()

    nnet_path = args.input_file
    output_dir = args.directory


    if nnet_path is None:
        print("Error: you must specify a single input file.")
        parser.print_usage()
        raise SystemExit(1)

    os.makedirs(output_dir, exist_ok=True)


    if not os.path.isfile(nnet_path):
        print(f"Error: file {nnet_path} not found")
        SystemExit(1)
        
    print(f"Generating single dataset from file: {nnet_path}")
    
    N_TOTAL = 1000000
    oracle = ACASXuOracle(nnet_path)
    X_all, Y_all = oracle.generate_dataset(N_TOTAL)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    base_name = os.path.splitext(os.path.basename(nnet_path))[0]
    out_filename = f"{base_name}"
    dataset_dir = f"{base_name}_dataset"
    out_path = os.path.join(output_dir, dataset_dir)
    os.makedirs(out_path, exist_ok=True)

    np.savez(os.path.join(out_path, f"{base_name}_train.npz"), X=X_train, Y=Y_train)
    np.savez(os.path.join(out_path, f"{base_name}_test.npz"), X=X_test, Y=Y_test)
    print(f"Dataset saved under {out_path}")