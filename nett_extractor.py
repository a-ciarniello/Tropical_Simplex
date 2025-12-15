import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

class ACASXuOracle:
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
        """Convert to 1D float vector, raise if not numeric."""
        cleaned = [x for x in seq if str(x).strip() != ""]
        arr = np.asarray(cleaned, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} deve essere un vettore 1D")
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
            raise ValueError(f"Header non valido nella prima linea del file .nnet: '{lines[0]}'") from exc
        if len(header) < 2:
            raise ValueError(f"Header incompleto: attesi almeno num_layers e input_size, ottenuto {header}")
        num_layers = header[0]
        input_size = header[1]
        
        # Line 1: Layer sizes (ignored here, inferred from weights)
        # Line 2: Symmetric flag (unused)
        
        # Line 3: Input minimums
        self.min_inputs = self._to_float_vector(lines[3].split(","), "min_inputs")
        
        # Line 4: Input maximums
        self.max_inputs = self._to_float_vector(lines[4].split(","), "max_inputs")
        
        # Line 5: Means for normalization
        self.means = self._to_float_vector(lines[5].split(","), "means")
        
        # Line 6: Ranges for normalization
        self.ranges = self._to_float_vector(lines[6].split(","), "ranges")

        # Load weights and biases
        current_line = 7
        for i in range(num_layers):
            # Dimensions for the current layer would be read here; kept for completeness
            # In the ACAS Xu schema layers are sequential. We assume the standard
            # structure (5 -> 50 -> ... -> 5) and parse weights and biases below.
            pass
        
        # Robust pass to extract weight matrices and bias vectors
        all_floats = []
        with open(self.filename, 'r') as f:
            for line in f:
                if line.strip().startswith("//") or not line.strip(): continue
                parts = line.split(",")[:-1] # Drop trailing empty if comma at end
                if not parts: parts = line.split(",") # Fallback
                try:
                    vals = [float(x) for x in parts if x.strip() != '']
                    all_floats.extend(vals)
                except:
                    pass
        
        # ACAS Xu networks have fixed structure: 5 inputs, 6 hidden layers of 50, 5 outputs
        layer_sizes = [5, 50, 50, 50, 50, 50, 50, 5]
        
        # Skip header metadata, then read weights/biases in order
        
        idx = 0
        # Reload only weights and biases, skipping the first 7 metadata lines
        raw_data = lines[7:] 
        
        val_idx = 0
        flat_data = []
        for l in raw_data:
            flat_data.extend([float(x) for x in l.split(",") if x.strip()])
            
        idx = 0
        for i in range(len(layer_sizes) - 1):
            rows = layer_sizes[i+1]
            cols = layer_sizes[i]
            
            # Weights (matrix rows x cols)
            w_size = rows * cols
            w = np.array(flat_data[idx : idx + w_size]).reshape(rows, cols)
            self.weights.append(w)
            idx += w_size
            
            # Bias (vector rows)
            b_size = rows
            b = np.array(flat_data[idx : idx + b_size])
            self.biases.append(b)
            idx += b_size

    def predict(self, inputs):
        """
        inputs: numpy array (N, 5) not normalized
        """
        # 1. Normalization (critical for ACAS Xu)
        # Standardize inputs: (x - mean) / range
        # Last value of means/ranges is for output; use only the first 5
        inp_norm = (inputs - self.means[:-1]) / self.ranges[:-1]
        
        # 2. Forward pass
        activation = inp_norm.T # Transpose for matrix multiplication (5, N)
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # W is (50, 5), activation is (5, N) -> z is (50, N)
            z = np.dot(W, activation) + b[:, np.newaxis]
            
            # Apply ReLU to all but last layer
            if i < len(self.weights) - 1:
                activation = np.maximum(0, z) # ReLU
            else:
                activation = z # Last layer linear
                
        return activation.T # Return (N, 5)

    def generate_dataset(self, num_samples=10000):
        # Generate random inputs uniformly within min/max from the header
        
        # Create random matrix (N, 5)
        random_inputs = np.random.uniform(0, 1, (num_samples, 5))
        
        # Scale to real ranges: min + rand * (max - min)
        delta = np.subtract(self.max_inputs, self.min_inputs, dtype=float)
        real_inputs = self.min_inputs + random_inputs * delta
        
        # Get labels from the oracle
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
        print("Error: specify the .nnet file to use")
        parser.print_usage()
        raise SystemExit(1)

    os.makedirs(output_dir, exist_ok=True)


    if not os.path.exists(nnet_path):
        print(f"Error: file {nnet_path} not found.")
    else:
        
        N_TOTAL = 100000 
        oracle = ACASXuOracle(nnet_path)
        X_all, Y_all = oracle.generate_dataset(N_TOTAL)
        
        # Split with shuffle: 20% test, 80% train
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all, Y_all, test_size=0.2, random_state=42
        )
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Save splits for later training
        base_name = os.path.splitext(os.path.basename(nnet_path))[0]

        out_filename = f"{base_name}"
        dataset_dir = f"{base_name}_dataset"
        out_path = os.path.join(output_dir, dataset_dir)

        os.makedirs(out_path, exist_ok=True)


        np.savez(os.path.join(out_path, f"{base_name}_train.npz"), X=X_train, Y=Y_train)
        np.savez(os.path.join(out_path, f"{base_name}_test.npz"), X=X_test, Y=Y_test)
        print(f"Salvato in {out_path}")